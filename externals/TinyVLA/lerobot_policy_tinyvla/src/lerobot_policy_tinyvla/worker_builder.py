"""Reuse the verified loader without importing old TinyVLA dependencies in LeRobot."""
import ast
import hashlib


TAIL = r'''
    import hashlib
    import numpy as np
    import time
    import traceback
    from importlib.metadata import version, PackageNotFoundError

    channel = args.output.parent
    head_file = args.model / 'action_head.safetensors'
    action_head = 'new_random_initialization'
    if head_file.is_file():
        from safetensors.torch import load_file
        parameters = dict(model.named_parameters())
        names = {name for name in parameters if name.startswith(('embed_out.', 'proj_to_action.'))}
        head = load_file(str(head_file))
        if not names or set(head) != names:
            raise ValueError('Unexpected action-head keys in ' + str(head_file))
        for name, tensor in head.items():
            loaded = parameters[name].detach().cpu()
            if not torch.isfinite(tensor).all() or not torch.equal(loaded, tensor.to(loaded.dtype)):
                raise ValueError('Trained action-head tensor did not reload: ' + name)
        action_head = 'verified_checkpoint'
    versions = {}
    for package in ('torch', 'transformers', 'diffusers', 'numpy'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    ready = {
        'status': 'TINYVLA_READY', 'versions': versions, 'seed': args.seed,
        'repo': str(args.repo), 'repo_commit': revision(args.repo),
        'model': str(args.model), 'action_head': action_head,
        'state_dim': 8, 'action_dim': 7, 'chunk_size': 16,
        'scheduler_dtype': str(model.noise_scheduler.alphas_cumprod.dtype),
        'model_config_sha256': hashlib.sha256((args.model / 'config.json').read_bytes()).hexdigest(),
        'loading_info': info,
    }
    if model.noise_scheduler.alphas_cumprod.dtype != torch.float32:
        raise RuntimeError('Apply the successful step-7 scheduler fix first.')
    ready_tmp = args.output.with_suffix('.tmp')
    ready_tmp.write_text(json.dumps(ready, default=str), encoding='utf-8')
    ready_tmp.replace(args.output)
    print('TINYVLA_READY', flush=True)
    last_id = -1
    while not (channel / 'stop').exists():
        request_path = channel / 'request.json'
        if not request_path.exists():
            time.sleep(0.01)
            continue
        request = json.loads(request_path.read_text(encoding='utf-8'))
        if request['id'] == last_id:
            time.sleep(0.01)
            continue
        last_id = request['id']
        try:
            started = time.perf_counter()
            with np.load(channel / 'input.npz', allow_pickle=False) as data:
                external = data['external'].copy()
                wrist = data['wrist'].copy()
                state_values = data['state'].copy()
            instructions = request['instructions']
            size = len(instructions)
            if not size or state_values.shape != (size, 8) or not np.isfinite(state_values).all():
                raise ValueError('Expected one finite eight-value robot state per batch element.')
            camera_tensors = []
            for array in (external, wrist):
                if array.dtype != np.uint8 or array.ndim != 4 or array.shape[0] != size or array.shape[-1] != 3:
                    raise ValueError('Expected one channel-last uint8 camera image per batch element.')
                processed = process_images([Image.fromarray(frame) for frame in array], processor, cfg)
                if not isinstance(processed, torch.Tensor) or processed.shape[0] != size:
                    raise ValueError('Expected one processed image per batch element.')
                camera_tensors.append(processed.to(device='cuda', dtype=torch.bfloat16))
            states = torch.from_numpy(state_values).to(device='cuda', dtype=torch.bfloat16)
            image_marker = DEFAULT_IMAGE_TOKEN
            if getattr(cfg, 'mm_use_im_start_end', False):
                image_marker = DEFAULT_IM_START_TOKEN + image_marker + DEFAULT_IM_END_TOKEN
            # The action head pools the token sequence, so batch equal lengths without padding.
            groups = {}
            for index, instruction in enumerate(instructions):
                conv = conv_templates['pythia'].copy()
                conv.append_message(conv.roles[0], image_marker + '\n' + instruction)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt() + ' <|endoftext|>'
                tokens = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
                groups.setdefault(len(tokens), []).append((index, tokens))

            actions = torch.empty((size, 16, 7), dtype=torch.float32, device='cpu')
            with torch.inference_mode():
                for group in groups.values():
                    indices, tokens = zip(*group)
                    indices = list(indices)
                    input_ids = torch.stack(tokens).to('cuda')
                    batch_actions = model(
                        input_ids=input_ids, attention_mask=input_ids.ne(tokenizer.pad_token_id),
                        images=camera_tensors[0][indices], images_r=camera_tensors[1][indices],
                        states=states[indices], eval=True,
                    )
                    if tuple(batch_actions.shape) != (len(indices), 16, 7):
                        raise RuntimeError(f'TinyVLA returned invalid action shape: {batch_actions.shape}')
                    actions[indices] = batch_actions.float().cpu()
            if not torch.isfinite(actions).all().item():
                raise RuntimeError('TinyVLA returned invalid actions.')
            response = {'id': last_id, 'ok': True, 'actions': actions.float().cpu().tolist(),
                        'seconds': time.perf_counter() - started}
        except Exception:
            error = traceback.format_exc()
            print(error, flush=True)
            response = {'id': last_id, 'ok': False, 'error': error}
        temporary = channel / 'response.tmp'
        temporary.write_text(json.dumps(response, allow_nan=False), encoding='utf-8')
        temporary.replace(channel / 'response.json')


if __name__ == '__main__':
    main()
'''


def build_worker(source):
    ast.parse(source)
    setup_end = source.index("    print('[3/5]")
    tokenizer_start = source.index('    tokenizer = AutoTokenizer.from_pretrained', setup_end)
    processor_start = source.index('    processor = processor_type.from_pretrained', tokenizer_start)
    tokenizer_end = source.index('\n', processor_start) + 1
    loader = source[:setup_end]
    if 'noise_scheduler' not in loader or 'torch.float32' not in loader:
        raise ValueError('The step-7 script must include the successful scheduler fix.')
    generated = loader + source[tokenizer_start:tokenizer_end] + TAIL
    ast.parse(generated)
    return generated, hashlib.sha256(source.encode()).hexdigest()
