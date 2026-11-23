"""AI-written reuse of immutable language and existing standard LIBERO pixels."""

import ast
from contextlib import ExitStack, contextmanager
from functools import wraps
from inspect import getsource, signature
from textwrap import dedent
from unittest.mock import patch


_RENDER_SOURCE = '''def render(self):
    self._ensure_env()
    raw_obs = self._env.env._get_observations()
    pixels = self._format_raw_obs(raw_obs)["pixels"]
    image = next(iter(pixels.values()))
    image = image[::-1, ::-1]
    return image
'''


def observation_render_supported(render):
    """Fall back if a future or decorated renderer changes the verified semantics."""
    if getattr(render, "__wrapped__", None) is not None:
        return False
    try:
        current = ast.parse(dedent(getsource(render))).body[0]
        expected = ast.parse(_RENDER_SOURCE).body[0]
        return (isinstance(current, ast.FunctionDef) and not current.decorator_list
                and ast.dump(ast.Module(body=current.body, type_ignores=[])) ==
                ast.dump(ast.Module(body=expected.body, type_ignores=[])))
    except (OSError, TypeError, SyntaxError, UnicodeError):
        return False


def standard_libero_camera(config):
    """Identify the native fixed-task environment, excluding custom Hub modules."""
    try:
        from lerobot.envs.configs import LiberoEnv
    except ImportError:
        return False, None
    if (type(config) is not LiberoEnv or getattr(config, "hub_path", None)
            or getattr(config, "is_libero_plus", False)):
        return False, None
    cameras = getattr(config, "camera_name", ())
    cameras = [name.strip() for name in cameras.split(",") if name.strip()] if isinstance(cameras, str) else cameras
    mapping = getattr(config, "camera_name_mapping", None)
    if mapping is None:
        mapping = {"agentview_image": "image", "robot0_eye_in_hand_image": "image2"}
    key = mapping.get(cameras[0]) if cameras and isinstance(mapping, dict) else None
    if isinstance(key, str):
        try:
            from lerobot.envs.libero import LiberoEnv as Implementation

            if not observation_render_supported(Implementation.render):
                key = None
        except ImportError:
            key = None
    return True, key if isinstance(key, str) else None


@contextmanager
def existing_observations(env, camera_key):
    """Reuse only immutable strings and exact uint8 render pixels within a rollout.

    A genuine reset clears both caches. Other methods and calls with arguments
    retain their native behavior. Render replies are independent contiguous
    copies, matching transport semantics without mutating policy observations.
    """
    original_call, original_reset, original_step = env.call, env.reset, env.step
    language, pixels = None, None

    def remember(result):
        nonlocal pixels
        pixels = None
        if camera_key is None or not isinstance(result, (tuple, list)) or not result:
            return result
        observation = result[0]
        images = observation.get("pixels") if isinstance(observation, dict) else None
        frame = images.get(camera_key) if isinstance(images, dict) else None
        try:
            import numpy as np
        except ImportError:
            return result
        if (isinstance(frame, np.ndarray) and frame.dtype == np.uint8 and frame.ndim == 4
                and frame.shape[0] == env.num_envs and frame.shape[-1] == 3):
            pixels = frame
        return result

    @wraps(original_reset)
    def reset(*args, **kwargs):
        nonlocal language, pixels
        language, pixels = None, None
        return remember(original_reset(*args, **kwargs))

    @wraps(original_step)
    def step(*args, **kwargs):
        nonlocal pixels
        pixels = None
        return remember(original_step(*args, **kwargs))

    def frame(index):
        return pixels[index, ::-1, ::-1].copy()

    @wraps(original_call)
    def call(name, *args, **kwargs):
        nonlocal language
        if not args and not kwargs:
            if name == "task_description":
                if language is not None:
                    return language
                result = original_call(name)
                if type(result) is tuple and all(isinstance(value, str) for value in result):
                    language = result
                return result
            if name == "render" and pixels is not None:
                return tuple(frame(index) for index in range(env.num_envs))
        return original_call(name, *args, **kwargs)

    with ExitStack() as hooks:
        hooks.enter_context(patch.object(env, "call", call))
        hooks.enter_context(patch.object(env, "reset", reset))
        hooks.enter_context(patch.object(env, "step", step))
        # SyncVectorEnv's native render callback calls its sub-envs directly.
        for index, child in enumerate(getattr(env, "envs", ())):
            original = child.render

            def render(*args, _index=index, _original=original, **kwargs):
                if pixels is not None and not args and not kwargs:
                    return frame(_index)
                return _original(*args, **kwargs)

            hooks.enter_context(patch.object(child, "render", render))
        yield


@contextmanager
def standard_libero_rollouts(evaluator):
    """Apply the safe reuse to every policy only for certified native environments."""
    original_factory = getattr(evaluator, "make_env", None)
    original_rollout = getattr(evaluator, "rollout", None)
    try:
        factory_signature, rollout_signature = signature(original_factory), signature(original_rollout)
    except (TypeError, ValueError):
        yield
        return
    if (not {"cfg", "env_cfg"} & factory_signature.parameters.keys()
            or "env" not in rollout_signature.parameters):
        yield
        return
    eligible = {}

    @wraps(original_factory)
    def make_env(*args, **kwargs):
        bound = factory_signature.bind(*args, **kwargs)
        config = bound.arguments.get("cfg", bound.arguments.get("env_cfg"))
        enabled, camera_key = standard_libero_camera(config)
        environments = original_factory(*args, **kwargs)
        if enabled and isinstance(environments, dict):
            for group in environments.values():
                if isinstance(group, dict):
                    eligible.update((id(env), camera_key) for env in group.values())
        return environments

    @wraps(original_rollout)
    def rollout(*args, **kwargs):
        bound = rollout_signature.bind(*args, **kwargs)
        env = bound.arguments.get("env")
        if id(env) not in eligible:
            return original_rollout(*args, **kwargs)
        with existing_observations(env, eligible[id(env)]):
            return original_rollout(*args, **kwargs)

    with patch.object(evaluator, "make_env", make_env), patch.object(evaluator, "rollout", rollout):
        yield
