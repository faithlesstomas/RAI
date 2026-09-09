# Local voice profiles

Stage 4.3 is in progress. The current slice defines synthesis contracts,
selection profiles, an actuator, lazy Piper and sounddevice adapters, shared
capability-service composition and deterministic test substitutes. It does not
yet load profiles into the default runtime, replace the legacy facade, capture a
microphone or transcribe speech.

## Why profiles are separate from voices

A synthesis profile answers *how* to speak: acceptable backends, whether remote
execution is possible, and the latency, cost, RAM and VRAM envelope. A voice
profile answers *which voice* to use by mapping one user-facing identity to the
voice identifier understood by each backend. Keeping those concerns separate
lets a user switch between fast conversation and slower document narration
without rewriting application code or silently changing privacy boundaries.

The built-in defaults are:

| Profile | Ordered backends | Intended use | Remote permitted |
| --- | --- | --- | --- |
| `realtime_local` | Piper | interactive responses on CPU | no |
| `quality_local` | Chatterbox, XTTS, Piper | narration where quality matters more than latency | no |
| `premium_remote` | ElevenLabs, Gemini, then local backends | explicitly authorized premium output | possible, but disabled by default |

These are conservative defaults returned by
`rai.speech.profiles.default_synthesis_profiles()`, not fixed architecture.
Runtime configuration loading is part of GitLab #22. A deployment may replace
the ordering and budgets, and unavailable optional backends are skipped without
import-time process termination.

The new Piper path resolves only an existing `.onnx`/`.onnx.json` pair. Model
provisioning is an explicit operation and synthesis never downloads artifacts.
Both Piper model loading and sounddevice imports are lazy, so a base RAI install
can import and test the speech package without either optional dependency. The
old `rai.tts` facade is now import-safe but remains transitional and must not be
used for new runtime composition.

## Selection and completion contract

Selection is deterministic. A backend must have an explicit voice binding and
must satisfy availability, language, locality, streaming, first-audio latency,
provider-cost, RAM and VRAM constraints. Local profiles never widen themselves
to cloud providers. Remote selection requires agreement from the profile and an
authorized request; the actuator also requires trusted composition to enable
remote execution. Sensitive `SECRET` and `BLOCKED` text cannot be routed
remotely.

`SpeechSynthesisActuator.act()` performs these steps:

1. Validate the typed request, profile and voice.
2. Select one eligible backend.
3. Synthesize ephemeral signed 16-bit PCM chunks.
4. Wait for playback or the deterministic player receipt.
5. Return a terminal `ActionResult` containing only metadata and completion
   evidence.

Cancellation, missing profiles or voices, unavailable backends, unsupported
languages and exceeded budgets are terminal typed failures. Input text and raw
audio are deliberately absent from success and failure records.

`register_speech_synthesis()` attaches the actuator to a chosen
`CapabilityRegistry`. This is intentionally explicit until configuration owns
the selected profile, voice artifacts and output device. Calls through that
registry receive the normal policy decision and two-stage decision/terminal
audit. A remote target must also be visible as an HTTP(S) `target_resource`;
private text is escalated before any backend is invoked.

## Remaining work

GitLab #21 is the Stage 4.3 umbrella. #22 completes default runtime composition,
legacy-facade migration, text normalization and configuration. #23 adds
push-to-talk VAD/STT and transcript provenance. #24 adds quality-local and
remote adapters without hidden cloud fallback. #25 covers an optional,
consented Piper voice-enrollment workflow. Existing issue #1 remains the
Markdown normalization acceptance case.
