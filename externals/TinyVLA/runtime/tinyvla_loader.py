#!/usr/bin/env python3
"""Check standalone TinyVLA inference with synthetic inputs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path,
                        default=os.environ.get('TINYVLA_WORKSPACE'))
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.workspace:
        args.workspace = args.workspace.expanduser().resolve()
    if not args.workspace and not (args.repo and args.model and args.output):
        parser.error('Set TINYVLA_WORKSPACE or provide all of --repo, --model, and --output.')
    if args.workspace:
        args.repo = args.repo or args.workspace / 'sources' / 'TinyVLA'
        args.model = args.model or args.workspace / 'models' / 'Llava-Pythia-700M'
        args.output = args.output or args.workspace / 'logs' / 'step7_result.json'
    for name in ('repo', 'model', 'output'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def revision(repo):
    if not (repo / '.git').exists():
        return 'unknown'
    result = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else 'unknown'


def action_head_loading(parameter_names, loading_info):
    """Describe active action-head restoration without asserting it was trained.

    Compare loading diagnostics only against parameters of the configured head.
    Older snapshots can carry an unused language embed_out.weight; that unexpected
    tensor says nothing about whether the current diffusion head was restored.
    """
    expected = {name for name in parameter_names if name.startswith(('embed_out.', 'proj_to_action.'))}
    missing = set(loading_info.get('missing_keys') or ())
    mismatched = {item[0] for item in loading_info.get('mismatched_keys') or ()
                  if isinstance(item, (tuple, list)) and item and isinstance(item[0], str)}
    initialized = expected & (missing | mismatched)
    if not expected:
        status = 'unknown'
    elif initialized == expected:
        status = 'new_random_initialization'
    elif initialized:
        status = 'partially_initialized'
    else:
        status = 'loaded_from_checkpoint'
    return {'status': status, 'expected_parameter_keys': len(expected),
            'newly_initialized_keys': sorted(initialized)}


def main():
    args = arguments()
    implementation = args.repo / 'llava-pythia' / 'llava_pythia'
    if not implementation.is_dir():
        raise FileNotFoundError(f'TinyVLA source directory not found: {implementation}')
    if not (args.model / 'config.json').is_file():
        raise FileNotFoundError(f'Model configuration not found: {args.model}')
    if not any((args.model / name).is_file() for name in
               ('model.safetensors', 'model.safetensors.index.json')):
        raise FileNotFoundError(f'Model weights not found: {args.model}')
    sys.path[:0] = [str(args.repo), str(args.repo / 'llava-pythia')]

    print('[1/5] Check runtime dependencies and compute device', flush=True)
    import torch
    from PIL import Image
    from transformers import AutoTokenizer, CLIPImageProcessor, SiglipImageProcessor
    from llava_pythia.constants import (
        DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX,
    )
    from llava_pythia.conversation import conv_templates
    from llava_pythia.mm_utils import process_images, tokenizer_image_token
    from llava_pythia.model.language_model.pythia.configuration_llava_pythia import LlavaPythiaConfig
    from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA device is available in this Python environment.')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('The action head requires a device that supports bfloat16.')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print('torch:', torch.__version__, 'device:', torch.cuda.get_device_name(0), flush=True)

    print('[2/5] Load the model weights and configure the action head', flush=True)
    cfg = LlavaPythiaConfig.from_pretrained(str(args.model), local_files_only=True)
    cfg.action_head_type = 'droid_diffusion'
    cfg.state_dim = 8
    cfg.action_dim = 7
    cfg.chunk_size = 16
    cfg.concat = 'token_cat'
    cfg.use_cache = False
    cfg.image_aspect_ratio = 'pad'
    model, info = LlavaPythiaForCausalLM.from_pretrained(
        str(args.model), config=cfg, torch_dtype=torch.bfloat16,
        local_files_only=True, use_safetensors=True,
        ignore_mismatched_sizes=True, output_loading_info=True,
        _fast_init=False,
    )
    permitted = ('embed_out.', 'proj_to_action.')
    unexpected_missing = [k for k in info.get('missing_keys', [])
                          if not k.startswith(permitted)]
    unexpected_mismatch = [item for item in info.get('mismatched_keys', [])
                           if not item[0].startswith(permitted)]
    if unexpected_missing or unexpected_mismatch or info.get('error_msgs'):
        raise RuntimeError('The vision-language weights did not load as expected: ' + json.dumps(info, default=str))
    # Detect incompatible backbone keys that indicate an incorrect module path.
    for prefix in ('gpt_neox.',):
        if any(k.startswith(prefix) for k in info.get('unexpected_keys', [])):
            raise RuntimeError('Unexpected keys in the base model: ' + json.dumps(info, default=str))
    head_loading = action_head_loading((name for name, _ in model.named_parameters()), info)
    model = model.to(device='cuda', dtype=torch.bfloat16).eval()
    # scheduler_float32_fix
    scheduler = model.noise_scheduler
    print('scheduler_before:', scheduler.alphas_cumprod.dtype,
          'alpha0:', scheduler.alphas_cumprod[0].item(), flush=True)

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        model.noise_scheduler = type(scheduler).from_config(scheduler.config)
    finally:
        torch.set_default_dtype(previous_dtype)

    schedule = model.noise_scheduler.alphas_cumprod
    print('scheduler_after:', schedule.dtype,
          'alpha0:', schedule[0].item(), flush=True)
    print('new_action_head_keys:', len(head_loading['newly_initialized_keys']), flush=True)

    print('[3/5] Prepare two test images and a robot state', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError('The tokenizer has neither an end-of-sequence token nor a padding token.')
        tokenizer.pad_token = tokenizer.eos_token
    tower = cfg.vision_config['vision_tower']['vision_model_name_or_path'].lower()
    processor_type = CLIPImageProcessor if 'clip' in tower else SiglipImageProcessor if 'siglip' in tower else None
    if processor_type is None:
        raise ValueError(f'Unsupported vision encoder for image preprocessing: {tower}')
    processor = processor_type.from_pretrained(str(args.model), local_files_only=True)
    # Use synthetic inputs for this standalone inference check.
    size = int(cfg.vision_config['vision_tower']['image_size'])
    images = [Image.new('RGB', (size, size), (128, 128, 128)),
              Image.new('RGB', (size, size), (96, 96, 96))]
    image_batch = process_images(images, processor, cfg)
    if not isinstance(image_batch, torch.Tensor) or image_batch.shape[0] != 2:
        raise RuntimeError('Expected a tensor containing two processed images.')
    image_batch = image_batch.to(device='cuda', dtype=torch.bfloat16)
    states = torch.zeros((1, 8), device='cuda', dtype=torch.bfloat16)
    image_marker = DEFAULT_IMAGE_TOKEN
    if getattr(cfg, 'mm_use_im_start_end', False):
        image_marker = DEFAULT_IM_START_TOKEN + image_marker + DEFAULT_IM_END_TOKEN
    conv = conv_templates['pythia'].copy()
    conv.append_message(conv.roles[0], image_marker + '\nPick up the object.')
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() + ' <|endoftext|>'
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to('cuda')
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    print('[4/5] Run one TinyVLA inference call', flush=True)
    with torch.inference_mode():
        actions = model(
            input_ids=input_ids, attention_mask=attention_mask,
            images=image_batch[:1], images_r=image_batch[1:2],
            states=states, eval=True,
        )
    torch.cuda.synchronize()
    if not isinstance(actions, torch.Tensor) or tuple(actions.shape) != (1, 16, 7):
        raise RuntimeError(f'Expected an action shape of (1, 16, 7), got {getattr(actions, "shape", type(actions))}.')
    if not torch.isfinite(actions).all().item():
        raise RuntimeError('The predicted actions contain non-finite values.')

    print('[5/5] Save the inference report', flush=True)
    report = {
        'status': 'MODEL_CALL_OK', 'input_kind': 'synthetic_images_and_zero_state',
        'simulator_connected': False, 'action_head': head_loading['status'],
        'action_head_loading': head_loading,
        'repo': str(args.repo), 'repo_commit': revision(args.repo),
        'model': str(args.model), 'torch': torch.__version__,
        'device': torch.cuda.get_device_name(0), 'seed': args.seed,
        'action_shape': list(actions.shape), 'all_finite': True,
        'first_action': actions[0, 0].float().cpu().tolist(),
        'loading_info': info,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print('MODEL_CALL_OK', flush=True)
    print('action_shape:', tuple(actions.shape), flush=True)
    print('all_finite: True', flush=True)
    print('result:', args.output, flush=True)


if __name__ == '__main__':
    main()
