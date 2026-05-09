# Voice Output Sanitization (OpenClaw mode)

## Why this exists

Some OpenAI-compatible streaming adapters can interleave tagged reasoning output
with user-visible content when a model emits both internal thinking and a tagged
final answer.

OpenClaw orchestrator prompts follow a strict contract:

- internal reasoning may appear in `<think>...</think>`
- user-visible content must be in `<final>...</final>`

If a transport leaks text outside `<final>`, voice clients can read internal
reasoning aloud. This module prevents that.

## Current behavior

When the active backend model starts with `openclaw:`, the server enables
`StreamSanitizer(strict_final=True)`.

In strict mode:

- stream chunks are buffered and filtered
- only content inside `<final>...</final>` is emitted as `response_chunk`
- internal reasoning/planning text is discarded for UI + TTS output

If a delayed/background response is needed, the server can also read the
authoritative `<final>` block from appended OpenClaw session events and deliver
that value instead of trusting raw stream output.

## Related durability behavior

For long-running tasks and reconnect safety:

- delivery state files track session offsets, pending email-copy requests, and
  last spoken answer
- outbox files queue completed responses that must replay after reconnect

Together, strict final filtering plus durable replay prevents malformed spoken
responses and reduces dropped follow-up deliveries.
