"""Transducer continuity across windows, including zero duration and overshoot."""
from types import SimpleNamespace
import unittest

import torch

from unittest.mock import Mock, patch

from app.bark_buffered import BufferedBark, DecodeState, decode_frames, encoder_windows, frame_end


class ToyTransducer:
    config = SimpleNamespace(vocab_size=5, blank_token_id=4, max_symbols_per_step=10,
                             durations=[0, 1, 2, 3, 4])
    generation_config = SimpleNamespace(decoder_start_token_id=4, eos_token_id=3)

    def __call__(self, encoder_outputs, decoder_input_ids, decoder_cache):
        frame = int(encoder_outputs.pooler_output.item())
        previous = int(decoder_input_ids.item())
        history = getattr(decoder_cache, "test_history", [])
        history.append((frame, previous))
        decoder_cache.test_history = history
        if frame == 0:
            token, duration = 0, 2
        elif frame == 2 and previous != 1:
            token, duration = 1, 0
        elif frame == 2:
            token, duration = 4, 3
        else:
            token, duration = 2, 1
        logits = torch.full((1, 1, 10), -100.0)
        logits[0, 0, token] = 1.0
        logits[0, 0, 5 + duration] = 10.0  # Must never be selected as a text token.
        return SimpleNamespace(logits=logits)


class BufferedDecoderTests(unittest.TestCase):
    def test_encoder_windows_cover_core_and_preserve_lookaround(self):
        spec = dict(core_start=461312, core_end=870144, window_start=300800, window_end=950144)
        windows = list(encoder_windows(spec))
        self.assertEqual([end - start for start, end, _, _ in windows], [160000, 160000, 88832])
        self.assertEqual(windows[0][0], spec["core_start"])
        self.assertEqual(windows[-1][1], spec["core_end"])
        for start, end, left, right in windows:
            self.assertGreaterEqual(start - left, 160000)
            self.assertEqual(right - end, 80000)
            self.assertEqual(left % 1280, 0)

    def test_split_matches_one_pass_and_carries_duration_overshoot(self):
        encoded = torch.arange(6).reshape(1, 6, 1)
        whole, split = DecodeState(), DecodeState()
        model = ToyTransducer()
        decode_frames(model, encoded, whole, 0, 6)
        decode_frames(model, encoded[:, :5], split, 0, 3)
        self.assertEqual(split.next_frame, 5)
        decode_frames(model, encoded[:, 1:], split, 1, 6)
        self.assertEqual(split.tokens, [0, 1, 2])
        self.assertEqual(split.tokens, whole.tokens)
        self.assertEqual(split.cache.test_history, whole.cache.test_history)
        self.assertEqual(split.next_frame, whole.next_frame)

    def test_adjacent_unaligned_sample_boundaries_share_one_encoder_grid(self):
        for cut in (461312, 870144, 2894336):
            left_start = max(0, cut - 160000) // 1280 * 1280
            self.assertEqual(frame_end(cut) - left_start // 1280, frame_end(cut - left_start))

    def test_lost_state_and_invalid_coordinates_fail_before_inference(self):
        import numpy as np
        worker = BufferedBark(SimpleNamespace())
        spec = dict(stream_id="new", index=1, window_start=0, window_end=16000,
                    core_start=8000, core_end=16000, final=True)
        with self.assertRaisesRegex(ValueError, "missing or out of order"):
            worker.transcribe(np.zeros(16000, dtype=np.float32), 16000, spec)
        spec.update(index=0, core_start=0, window_start=1)
        with self.assertRaisesRegex(ValueError, "coordinates"):
            worker.transcribe(np.zeros(15999, dtype=np.float32), 16000, spec)

    def test_session_commit_is_transactional_and_final_retry_is_idempotent(self):
        import numpy as np

        class Features(dict):
            def to(self, *args):
                return self

        class Processor:
            def __call__(self, audio, **kwargs):
                return Features(input_features=torch.zeros(1, len(audio) // 160, 1))

            def batch_decode(self, tokens, **kwargs):
                return [" ".join(map(str, row)) for row in tokens]

        model = ToyTransducer()
        def encode(input_features, **kwargs):
            count = (input_features.shape[1] + 7) // 8
            return SimpleNamespace(pooler_output=torch.arange(count).reshape(1, count, 1),
                                   attention_mask=torch.ones(1, count))
        model.get_audio_features = encode
        engine = SimpleNamespace(model=model, processor=Processor(), device="cpu", dtype=torch.float32,
                                 synchronize=Mock(), torch=SimpleNamespace(
            inference_mode=torch.inference_mode, bfloat16=torch.bfloat16,
            cuda=SimpleNamespace(synchronize=Mock())))
        worker = BufferedBark(engine)
        audio = np.full(160000, 0.1, dtype=np.float32)
        first = dict(stream_id="one", index=0, window_start=0, window_end=160000,
                     core_start=0, core_end=80000, final=False)
        result = worker.transcribe(audio, 16000, first)
        self.assertEqual(worker.transcribe(audio, 16000, first), result)
        original_tokens = list(worker.state.tokens)
        final = dict(first, index=1, core_start=80000, core_end=160000, final=True)
        def fail_after_mutation(model, encoded, state, *args):
            state.tokens.append(999)
            raise RuntimeError("Injected decode failure")
        with patch("app.bark_buffered.decode_frames", side_effect=fail_after_mutation):
            with self.assertRaisesRegex(RuntimeError, "Injected"):
                worker.transcribe(audio, 16000, final)
        self.assertEqual(worker.state.tokens, original_tokens)
        self.assertEqual(worker.index, 0)
        finished = worker.transcribe(audio, 16000, final)
        self.assertIsNone(worker.state)  # Release all GPU decoder state on completion.
        self.assertEqual(worker.transcribe(audio, 16000, final), finished)
        with self.assertRaisesRegex(ValueError, "Changed buffered retry"):
            worker.transcribe(audio * 2, 16000, final)
        with self.assertRaisesRegex(ValueError, "out of order"):
            worker.transcribe(audio, 16000, dict(final, stream_id="lost"))
        self.assertEqual(worker.transcribe(audio, 16000, final), finished)
        restarted = worker.transcribe(audio, 16000, dict(first, stream_id="two"))
        self.assertEqual(restarted["text"], result["text"])
