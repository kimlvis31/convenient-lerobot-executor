"""AI-written frame ordering, bounded storage, encoder and cleanup checks."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluation_video import FrameSpool, bounded_evaluation_videos, frame_storage_code

try:
    import numpy as np
except ImportError:
    np = None


received = {}


def write_video(path, frames, fps):
    received[path] = (np.stack(list(frames)), fps)


def rollout(env):
    return env.frames


def reference_eval(env, max_episodes_rendered=2, save_predicted_video=False):
    threads = []
    ep_frames: list[np.ndarray] = []
    for frame in rollout(env):
        ep_frames.append(frame)
    if max_episodes_rendered > 0 and len(ep_frames) > 0:
        batch_stacked_frames = np.stack(ep_frames, axis=1)
        for index, (stacked_frames, done_index) in enumerate(zip(batch_stacked_frames, env.done_indices)):
            thread = threading.Thread(target=write_video, args=(f"{env.name}/{index}", stacked_frames[:done_index+1], 30))
            thread.start()
            threads.append(thread)
            if getattr(env, "fail_after_start", False):
                raise RuntimeError("later rollout failed")
    if save_predicted_video:
        thread = threading.Thread(target=write_video, args=(f"{env.name}/predicted", env.predicted, 30))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    return {"completed": True}


class VideoContractTests(unittest.TestCase):
    def test_unknown_or_decorated_storage_contract_keeps_original_function(self):
        original = lambda env: "native"
        evaluator = SimpleNamespace(eval_policy=original)
        with bounded_evaluation_videos(evaluator):
            self.assertIs(evaluator.eval_policy, original)
        self.assertIsNone(frame_storage_code(original))

    def test_installed_native_evaluator_is_recognized(self):
        try:
            from lerobot.scripts import lerobot_eval
        except ImportError:
            self.skipTest("LeRobot is unavailable")
        self.assertIsNotNone(frame_storage_code(lerobot_eval.eval_policy))


@unittest.skipIf(np is None, "NumPy is unavailable")
class VideoStorageTests(unittest.TestCase):
    def make_env(self, name="video", fail=False):
        frames = [np.full((2, 5, 7, 3), step, np.uint8) + np.arange(2, dtype=np.uint8)[:, None, None, None]
                  for step in range(9)]
        return SimpleNamespace(name=name, frames=frames, done_indices=[3, 7], fail_after_start=fail)

    def setUp(self):
        received.clear()

    def test_spool_copies_mutable_and_noncontiguous_input_and_reads_exact_slices(self):
        spool = FrameSpool()
        self.addCleanup(spool.close)
        source = np.arange(2*5*7*3, dtype=np.uint8).reshape(2,5,7,3)
        expected = []
        for step in range(8):
            source[:] += step
            frame = source[:, ::-1]
            expected.append(frame.copy())
            spool.append(frame)
        source.fill(0)
        stacked = np.stack(expected, axis=1)
        episodes = spool.stack()
        self.assertEqual(len(episodes), 2)
        for batch, episode in enumerate(episodes):
            np.testing.assert_array_equal(np.stack(list(episode)), stacked[batch])
            np.testing.assert_array_equal(episode[-1], stacked[batch, -1])
            np.testing.assert_array_equal(np.stack(list(episode[1:7:2])), stacked[batch, 1:7:2])
        self.assertEqual(os.fstat(spool.file.fileno()).st_size, 8*source.nbytes)
        self.assertFalse(any(isinstance(value, (list, np.ndarray)) for value in vars(spool).values()))

    def test_shape_changes_fail_and_readers_release_storage_after_the_last_video(self):
        spool = FrameSpool()
        self.addCleanup(spool.close)
        spool.append(np.zeros((2,4,4,3), np.uint8))
        with self.assertRaises(ValueError):
            spool.append(np.zeros((2,5,4,3), np.uint8))
        spool.stack()
        spool.reader_finished()
        self.assertFalse(spool.file.closed)
        spool.reader_finished()
        self.assertTrue(spool.file.closed)

    def test_native_order_trimming_and_encoder_options_are_preserved(self):
        env = self.make_env()
        expected_result = reference_eval(env)
        expected = {key: (value[0].copy(), value[1]) for key,value in received.items()}
        received.clear()
        evaluator = SimpleNamespace(eval_policy=reference_eval)
        with bounded_evaluation_videos(evaluator):
            self.assertEqual(evaluator.eval_policy(env), expected_result)
        self.assertIs(evaluator.eval_policy, reference_eval)
        self.assertEqual(set(received), set(expected))
        for key in expected:
            np.testing.assert_array_equal(received[key][0], expected[key][0])
            self.assertEqual(received[key][1], expected[key][1])
        self.assertNotIn("_convlex_frame_spool", reference_eval.__globals__)

    def test_threads_and_later_rollout_hooks_remain_independent(self):
        evaluator = SimpleNamespace(eval_policy=reference_eval)
        with bounded_evaluation_videos(evaluator), patch(__name__ + ".rollout", side_effect=rollout) as observed:
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(evaluator.eval_policy, [self.make_env("a"), self.make_env("b")]))
            self.assertEqual(observed.call_count, 2)
        self.assertEqual(set(received), {"a/0", "a/1", "b/0", "b/1"})

    def test_failure_waits_for_started_encoders_before_closing_files(self):
        tracked, encoded = [], []

        class ObservedSpool(FrameSpool):
            def __init__(self):
                super().__init__()
                tracked.append(self)

        def slow_writer(path, frames, fps):
            time.sleep(.03)
            encoded.append(np.stack(list(frames)))

        evaluator = SimpleNamespace(eval_policy=reference_eval)
        with patch("evaluation_video.FrameSpool", ObservedSpool), patch(__name__ + ".write_video", slow_writer), \
             bounded_evaluation_videos(evaluator), self.assertRaisesRegex(RuntimeError, "later rollout failed"):
            evaluator.eval_policy(self.make_env(fail=True))
        self.assertEqual(len(encoded), 1)
        self.assertTrue(all(item.file.closed for item in tracked))

    def test_encoder_failure_is_reported_and_storage_is_released(self):
        evaluator = SimpleNamespace(eval_policy=reference_eval)
        with patch(__name__ + ".write_video", side_effect=OSError("disk full")), \
             bounded_evaluation_videos(evaluator), self.assertRaisesRegex(RuntimeError, "disk full"):
            evaluator.eval_policy(self.make_env())

    def test_real_native_encoder_decodes_the_same_frames(self):
        try:
            import av
            from lerobot.utils.io_utils import write_video as native_writer
        except ImportError:
            self.skipTest("Native video encoder is unavailable")
        frames = np.random.default_rng(42).integers(0,256,(13,2,17,19,3),dtype=np.uint8)
        spool = FrameSpool()
        self.addCleanup(spool.close)
        for frame in frames:
            spool.append(frame)
        lazy = spool.stack()[1][:10]
        with TemporaryDirectory() as directory:
            a,b = Path(directory)/"native.mp4", Path(directory)/"spooled.mp4"
            native_writer(a, frames[:10,1], 20)
            native_writer(b, lazy, 20)
            def decode(path):
                with av.open(str(path)) as container:
                    return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
            np.testing.assert_array_equal(np.stack(decode(a)), np.stack(decode(b)))


if __name__ == "__main__":
    unittest.main()
