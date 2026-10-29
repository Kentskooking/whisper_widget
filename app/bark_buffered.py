"""Buffered TDT inference with audio lookaround and carried prediction state.

Applies NVIDIA NeMo's left/core/right + stateful-decoding approach to the
Transformers Bark model. It does not import NeMo or modify
the installed library, model weights, tokenizer, or feature normalization.
"""
import copy
from dataclasses import dataclass, field
import hashlib
import json


FRAME_SAMPLES = 1280  # 160-sample mel hop * encoder subsampling factor 8
PREFIX_SAMPLES = 4000  # Bark's published 0.25-second leading silence
CORE_SAMPLES = 160000  # NeMo's recommended 10 s core / 10 s left / 5 s right
LEFT_SAMPLES = 160000
RIGHT_SAMPLES = 80000


def frame_end(sample):
    return (sample + PREFIX_SAMPLES + FRAME_SAMPLES - 1) // FRAME_SAMPLES


def encoder_windows(spec):
    """Subdivide a capture chunk without changing its saved pause boundaries."""
    start = spec["core_start"]
    while start < spec["core_end"]:
        end = min(spec["core_end"], start + CORE_SAMPLES)
        left = max(0, start - LEFT_SAMPLES) // FRAME_SAMPLES * FRAME_SAMPLES
        right = min(spec["window_end"], end + RIGHT_SAMPLES)
        if left < spec["window_start"]:
            raise ValueError("Buffered request is missing left context")
        yield start, end, left, right
        start = end


@dataclass
class DecodeState:
    next_frame: int = 0
    last_token: object = None
    cache: object = None
    tokens: list = field(default_factory=list)


def decode_frames(model, encoded, state, window_frame, stop_frame):
    """Decode owned encoder frames only, retaining duration overshoot and LSTM.

    ``next_frame`` and ``stop_frame`` are on the recording-global padded grid.
    Advancing across a cut must not reset a token's predicted duration. The
    last predicted token is fed exactly once on the following decoding step.
    """
    import torch
    from transformers.models.parakeet.generation_parakeet import ParakeetRNNTDecoderCache
    from transformers.models.parakeet.modeling_parakeet import ParakeetEncoderModelOutput

    if state.cache is None:
        state.cache = ParakeetRNNTDecoderCache(model.config)
        state.last_token = torch.tensor([[model.generation_config.decoder_start_token_id]], device=encoded.device)
    if state.next_frame < window_frame or stop_frame > window_frame + encoded.shape[1]:
        raise ValueError("Decoder frame range is outside the encoded context")
    limit = max(1, stop_frame - state.next_frame) * model.config.max_symbols_per_step
    steps = 0
    while state.next_frame < stop_frame:
        if steps >= limit:
            raise RuntimeError("Buffered TDT failed to advance through its audio")
        frame = state.next_frame - window_frame
        output = model(
            encoder_outputs=ParakeetEncoderModelOutput(pooler_output=encoded[:, frame:frame + 1]),
            decoder_input_ids=state.last_token, decoder_cache=state.cache,
        )
        logits = output.logits[0, -1]
        # Match stock generate: duration classes are not eligible text tokens.
        token, duration_index = torch.stack((logits[:model.config.vocab_size].argmax(),
                                            logits[model.config.vocab_size:].argmax())).tolist()
        duration = model.config.durations[duration_index]
        if token == model.config.blank_token_id:
            duration = max(1, duration)
        else:
            state.tokens.append(token)
        if token == model.generation_config.eos_token_id:
            raise RuntimeError("Bark emitted EOS before exhausting the buffered audio")
        state.last_token = output.logits[..., :model.config.vocab_size].argmax(dim=-1)
        state.next_frame += duration
        steps += 1
    return steps


class BufferedBark:
    """One active microphone session; stateless file requests remain independent.

    State commits only after successful inference. Duplicate last requests are
    idempotent; gaps, lost state, changed retries and cross-session continuations
    fail explicitly so the widget can recover from its original recording.
    """

    def __init__(self, engine):
        self.engine = engine
        self.stream_id = None
        self.index = -1
        self.core_end = 0
        self.state = None
        self.last_key = None
        self.last_result = None

    def transcribe(self, wave, rate, spec):
        import numpy as np

        required = {"stream_id", "index", "window_start", "window_end", "core_start", "core_end", "final"}
        if set(spec) != required or rate != 16000 or wave.ndim != 1:
            raise ValueError("Invalid buffered audio request")
        start, end = spec["window_start"], spec["window_end"]
        core_start, core_end, index = spec["core_start"], spec["core_end"], spec["index"]
        if (not isinstance(spec["stream_id"], str) or not spec["stream_id"] or
                any(type(spec[key]) is not int for key in required - {"stream_id", "final"}) or
                type(spec["final"]) is not bool or index < 0 or
                not 0 <= start <= core_start < core_end <= end or
                start % FRAME_SAMPLES or len(wave) != end - start or
                (spec["final"] and end != core_end)):
            raise ValueError("Inconsistent buffered audio coordinates")
        key = json.dumps(spec, sort_keys=True) + hashlib.sha256(wave.tobytes()).hexdigest()
        same_stream = spec["stream_id"] == self.stream_id
        if same_stream and index == self.index:
            if key != self.last_key:
                raise ValueError("Changed buffered retry")
            return copy.deepcopy(self.last_result)
        if index == 0 and not same_stream:
            if core_start != 0 or start != 0:
                raise ValueError("A new stream must start at sample zero")
            state = DecodeState()
        else:
            if not same_stream or index != self.index + 1 or core_start != self.core_end or self.state is None:
                raise ValueError("Buffered decoder state missing or out of order; use recording fallback")
            state = copy.deepcopy(self.state)
        engine = self.engine
        first_token = len(state.tokens)
        windows = []
        steps = 0
        with engine.torch.inference_mode():
            for owned_start, owned_end, left, right in encoder_windows(spec):
                audio = wave[left - start:right - start]
                padded = np.concatenate([np.zeros(PREFIX_SAMPLES, np.float32), audio])
                inputs = engine.processor(padded, sampling_rate=rate, return_tensors="pt").to(
                    engine.device, engine.dtype)
                encoded = engine.model.get_audio_features(**inputs, output_attention_mask=True)
                window_frame = left // FRAME_SAMPLES
                valid_frames = int(encoded.attention_mask.sum().item())
                final_window = spec["final"] and owned_end == core_end
                stop_frame = window_frame + valid_frames if final_window else frame_end(owned_end)
                # In an active session silence advances the transducer normally.
                silent_start = not state.tokens and float(np.max(np.abs(audio))) < 0.0001
                if silent_start:
                    state.next_frame = max(state.next_frame, stop_frame)
                else:
                    steps += decode_frames(engine.model, encoded.pooler_output, state, window_frame, stop_frame)
                windows.append(dict(core_start=owned_start, core_end=owned_end, window_start=left,
                                    window_end=right, stop_frame=stop_frame, next_frame=state.next_frame))
            text = engine.processor.batch_decode([state.tokens[first_token:]], skip_special_tokens=True)[0].strip()
            cumulative = engine.processor.batch_decode([state.tokens], skip_special_tokens=True)[0].strip()
            engine.synchronize()
        result = {"text": text, "buffered": dict(spec, cumulative_text=cumulative,
                  token_count=len(state.tokens), new_tokens=len(state.tokens) - first_token,
                  next_frame=state.next_frame, stop_frame=stop_frame, decode_steps=steps, windows=windows)}
        self.stream_id, self.index, self.core_end = spec["stream_id"], index, core_end
        self.state = None if spec["final"] else state
        self.last_key, self.last_result = key, copy.deepcopy(result)
        return result
