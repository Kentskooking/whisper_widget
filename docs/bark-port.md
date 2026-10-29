# Bark backend port

This branch replaces Whisper inference with Underdog Bark 3.0 Pro while keeping
the existing Windows widget, supervisor, storage layout, and embedded HTTP API.
It is source for users to install on their own machines.

## Source reuse

- Windows capture, original-audio handling, and buffered decoder behavior derive
  from the existing widget adaptation through commit `06d02a2`.
- Remote upload reconstruction, session persistence/reset/cancellation, and
  bounded file decoding derive from the Linux integration at `d4db2ff`.
- The original widget base is `baa9365`. Public launchers, dependency ranges,
  device/dtype selection, model cache resolution, and documentation are adapted
  here. Hardware-specific launch/setup scripts and deployment receipts are not
  part of the port.

## Boundaries

`app/bark_backend.py` owns model loading and device selection. Workers retain
the JSON request/response protocol and add a decoder reset request.
`app/bark_buffered.py` implements the carried TDT state. Desktop capture uses
`app/chunking.py` and `app/buffered_audio.py`; remote recordings use
`app/web/recording_audio.py` to reconstruct the original PCM timeline from
declared upload overlap before selecting internal boundaries.

Capture/transport chunks and internal decoder cores have different sizes.
Each core owns its samples exactly once. Five seconds of acoustic right context
does not guarantee five seconds of wall-clock delay: browser transport uploads
can arrive less frequently. Finalization flushes whatever context is available.

The inference worker loads cached files only. The explicit download command uses
the public Hugging Face model repository. Silero is installed as a package;
its sound-file input/output uses SoundFile and soxr rather than a machine-local
Torch Hub cache or a particular TorchAudio decoder version.

## Compatibility

The Windows desktop and JSON/HTTP entry points retain legacy `Whisper` class,
module, launcher, and environment names where they are compatibility surfaces.
UI labels and model metadata identify Bark. English recognition replaces
Whisper's multilingual capability. Whisper-specific decoding options are
rejected by the backend instead of silently ignored.

The original integration used Transformers 5.19. This port allows compatible
patch releases in that family because it directly uses Parakeet decoder/cache
interfaces. Other dependencies use ranges; PyTorch's hardware build is selected
by the installer. No particular GPU model is required or claimed to be supported.

## Validation scope

This is a source port, not a deployment receipt. Per the requested scope, this
branch was not installed, launched, benchmarked, or subjected to the regression
verifier or model/microphone acceptance on the author's workstation. Prior
Windows/Linux evaluation results are not new validation of this portable branch.
The retained/adapted regression tests and `tools/check_all.ps1` are available to
maintainers. Source review and Git whitespace checks are separate from execution.

Model files, caches, recordings, transcripts, runtime environments, and private
service configuration are excluded. The model's own NOTICE/license accompanies
its download; this branch does not bundle or relicense its weights.
