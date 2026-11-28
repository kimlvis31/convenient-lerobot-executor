"""AI-written bounded frame storage for the native evaluator's video writer.

Only the evaluator's recognized frame-list, stacking and encoder-thread sites
are adapted in memory. The native rollout and encoder remain unchanged.
"""

import ast
from collections.abc import Sequence
from contextlib import contextmanager
from functools import wraps
from inspect import getsource
import logging
from tempfile import TemporaryFile
from textwrap import dedent
from threading import Lock, Thread, local
from types import FunctionType
from unittest.mock import patch


class FrameSpool:
    """Store step-major frames in an anonymous file, never an episode list."""

    def __init__(self):
        self.file = TemporaryFile(prefix="convlex-frames-")
        self.lock = Lock()
        self.count = 0
        self.frame_shape = None
        self.dtype = None
        self.frame_bytes = None
        self.readers = 0

    def __len__(self):
        return self.count

    def append(self, frames):
        import numpy as np

        frames = np.ascontiguousarray(frames)
        if frames.ndim != 4 or frames.dtype != np.uint8:
            raise ValueError("Evaluation video frames must be batched uint8 RGB images.")
        if self.frame_shape is None:
            self.frame_shape, self.dtype = frames.shape, frames.dtype
            self.frame_bytes = frames[0].nbytes
        elif frames.shape != self.frame_shape or frames.dtype != self.dtype:
            raise ValueError("Evaluation video frame shape or dtype changed during a rollout.")
        self.file.write(memoryview(frames).cast("B"))
        self.count += 1

    def stack(self):
        """Return lazy B,T,H,W,C sequences instead of a second full array."""
        if not self.count:
            raise ValueError("Cannot stack an empty evaluation video.")
        self.file.flush()
        self.readers = self.frame_shape[0]
        return [EpisodeFrames(self, batch, range(self.count)) for batch in range(self.frame_shape[0])]

    def read(self, batch, step):
        import numpy as np

        offset = (step * self.frame_shape[0] + batch) * self.frame_bytes
        with self.lock:
            self.file.seek(offset)
            frame = self.file.read(self.frame_bytes)
        if len(frame) != self.frame_bytes:
            raise OSError("Incomplete evaluation video frame in temporary storage.")
        return np.frombuffer(frame, dtype=self.dtype).reshape(self.frame_shape[1:])

    def close(self):
        self.file.close()

    def reader_finished(self):
        with self.lock:
            self.readers -= 1
            if self.readers == 0:
                self.close()


class EpisodeFrames(Sequence):
    """Native write_video needs indexing and iteration, not a full ndarray."""

    def __init__(self, spool, batch, indices):
        self.spool, self.batch, self.indices = spool, batch, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return EpisodeFrames(self.spool, self.batch, self.indices[index])
        return self.spool.read(self.batch, self.indices[index])


def frame_storage_code(original):
    """Recognize the native list/stack contract; leave unknown versions alone."""
    if original.__closure__ or getattr(original, "__wrapped__", None) is not None:
        return None
    try:
        tree = ast.parse(dedent(getsource(original)))
    except (OSError, TypeError, SyntaxError):
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return None
    function = tree.body[0]
    if function.decorator_list:
        return None
    counts = {"list": 0, "stack": 0, "threads": 0}

    class Adapt(ast.NodeTransformer):
        def visit_AnnAssign(self, node):
            self.generic_visit(node)
            if (isinstance(node.target, ast.Name) and node.target.id == "ep_frames"
                    and isinstance(node.value, ast.List) and not node.value.elts):
                counts["list"] += 1
                node.value = ast.Call(func=ast.Name(id="_convlex_frame_spool", ctx=ast.Load()), args=[], keywords=[])
            return node

        def visit_Call(self, node):
            self.generic_visit(node)
            if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "np" and node.func.attr == "stack"
                    and len(node.args) == 1 and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == "ep_frames" and len(node.keywords) == 1
                    and node.keywords[0].arg == "axis" and isinstance(node.keywords[0].value, ast.Constant)
                    and node.keywords[0].value.value == 1):
                counts["stack"] += 1
                node = ast.copy_location(ast.Call(
                    func=ast.Attribute(value=node.args[0], attr="stack", ctx=ast.Load()), args=[], keywords=[]), node)
            elif (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "threading" and node.func.attr == "Thread"):
                counts["threads"] += 1
                node.func = ast.Name(id="_convlex_video_thread", ctx=ast.Load())
            return node

    # The stacked result must only feed native per-episode video iteration.
    uses = [node for node in ast.walk(function) if isinstance(node, ast.Name)
            and node.id == "batch_stacked_frames" and isinstance(node.ctx, ast.Load)]
    zipped = [node for node in ast.walk(function) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Name) and node.func.id == "zip" and node.args
              and isinstance(node.args[0], ast.Name) and node.args[0].id == "batch_stacked_frames"]
    frame_methods = {node.attr for node in ast.walk(function) if isinstance(node, ast.Attribute)
                     and isinstance(node.value, ast.Name) and node.value.id == "ep_frames"}
    joined = any(isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "thread"
                 and isinstance(node.iter, ast.Name) and node.iter.id == "threads"
                 and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                         and isinstance(call.func.value, ast.Name) and call.func.value.id == "thread"
                         and call.func.attr == "join" for call in ast.walk(node)) for node in ast.walk(function))
    if len(uses) != 1 or len(zipped) != 1 or frame_methods != {"append"} or not joined:
        return None
    tree = Adapt().visit(tree)
    if counts != {"list": 1, "stack": 1, "threads": 2}:
        return None
    # Extract the new code object without freezing the evaluator's globals:
    # later rollout/progress/camera hooks must remain visible to it.
    namespace = dict(original.__globals__)
    ast.increment_lineno(tree, original.__code__.co_firstlineno - 1)
    exec(compile(ast.fix_missing_locations(tree), original.__code__.co_filename, "exec"), namespace)
    return namespace[function.name].__code__


@contextmanager
def bounded_evaluation_videos(evaluator):
    original = getattr(evaluator, "eval_policy", None)
    code = frame_storage_code(original) if isinstance(original, FunctionType) else None
    if code is None:
        logging.info("Native evaluator video-storage contract is unknown; retaining its video writer.")
        yield
        return
    state = local()

    def spool():
        frames = FrameSpool()
        state.frames.append(frames)
        return frames

    def video_thread(*args, **kwargs):
        # Native sites use keyword targets. Keep their exact encoder/options.
        target = kwargs["target"]
        errors = state.errors

        @wraps(target)
        def encode(*parameters, **options):
            try:
                return target(*parameters, **options)
            except BaseException as error:
                errors.append(error)
            finally:
                if len(parameters) > 1 and isinstance(parameters[1], EpisodeFrames):
                    parameters[1].spool.reader_finished()

        thread = Thread(*args, **{**kwargs, "target": encode})
        state.threads.append(thread)
        return thread

    adapted = FunctionType(code, original.__globals__, original.__name__, original.__defaults__)
    adapted.__kwdefaults__ = original.__kwdefaults__

    @wraps(original)
    def evaluate(*args, **kwargs):
        previous = getattr(state, "frames", None), getattr(state, "threads", None), getattr(state, "errors", None)
        frames, threads, errors = state.frames, state.threads, state.errors = [], [], []
        try:
            result = adapted(*args, **kwargs)
            if errors:
                raise RuntimeError(f"Evaluation video encoding failed: {errors[0]}") from errors[0]
            return result
        finally:
            # Also join already-started encoders on a later rollout failure.
            # They may still read the anonymous files until they finish.
            try:
                for thread in threads:
                    if thread.ident is not None:
                        thread.join()
            finally:
                for item in frames:
                    item.close()
                state.frames, state.threads, state.errors = previous

    with patch.dict(original.__globals__, {"_convlex_frame_spool": spool, "_convlex_video_thread": video_thread}), \
         patch.object(evaluator, "eval_policy", evaluate):
        logging.info("Evaluation videos use temporary frame storage and bounded encoder reads.")
        yield
