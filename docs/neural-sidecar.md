# NCSI/J-lens neural sidecar

RAI provides an optional, separately started neural process for GAIA's
`gcas.ncsi.v1` interface. It owns the Transformers model, accelerator and
J-lens tensors. It exposes only token events and bounded concept observations;
raw tensors never leave the process. The sidecar is read-only: activation
steering and other interventions are not advertised or accepted.

Install the optional runtime and start it with an immutable model revision:

```bash
uv sync --extra inference-transformers
uv run rai neural serve \
  --model HuggingFaceTB/SmolLM2-135M \
  --revision MODEL_COMMIT_SHA \
  --uds "$XDG_RUNTIME_DIR/rai/neural.sock"
```

The socket is protected by the same token as the ordinary RAI API. Clients
send either `Authorization: Bearer TOKEN` or `X-RAI-Token: TOKEN`. A local GAIA
client can use an HTTP transport configured for the Unix socket and these
endpoints:

- `GET /api/v1/neural/capabilities`
- `GET /api/v1/neural/models`
- `GET /api/v1/neural/lenses`
- `GET /api/v1/neural/telemetry`
- `POST /api/v1/neural/generate`
- `POST /api/v1/neural/requests/{request-id}/cancel`

Generation uses newline-delimited JSON (`application/x-ndjson`). Example body:

```json
{
  "request-id": "gaia-run-1",
  "prompt": "Explain the result",
  "model-id": "HuggingFaceTB/SmolLM2-135M",
  "lens-id": "smollm2-jlens-v1",
  "max-new-tokens": 64,
  "timeout-seconds": 30,
  "top-k": 8,
  "layers": [4, 8]
}
```

Requests without `lens-id` provide the Transformers token stream but no neural
observations. Requests with a lens fail explicitly if the artifact is absent
or incompatible; they never silently downgrade to text-only mode.

## Lens artifacts

Lens weights live below `$XDG_DATA_HOME/rai/neural/lenses`, one directory per
artifact. RAI accepts only a checksummed `safetensors` payload with a
`rai.jlens-artifact.v1` `manifest.json`. Pickled Torch objects are not loaded.
Each tensor is named `layer.N.jacobian` and has shape
`[hidden_size, hidden_size]`. RAI computes `unembed(J_N @ residual)` using the
loaded model's final normalization and output embedding, matching the pinned
reference implementation. The manifest contains:

```json
{
  "schema-version": "rai.jlens-artifact.v1",
  "lens-id": "smollm2-jlens-v1",
  "lens-revision": "artifact-revision",
  "model": {
    "id": "HuggingFaceTB/SmolLM2-135M",
    "revision": "MODEL_COMMIT_SHA",
    "tokenizer-revision": "TOKENIZER_COMMIT_SHA",
    "architecture": "LlamaForCausalLM",
    "dtype": "float32",
    "quantization": null
  },
  "layers": [4, 8],
  "implementation-revision": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
  "fitting-parameters": {},
  "calibration-corpus": "corpus-id",
  "calibration-license": "license-id",
  "artifact-file": "lens.safetensors",
  "artifact-sha256": "64-lowercase-hex-characters",
  "created-at": "2026-08-18T00:00:00Z"
}
```

Model revision, tokenizer revision, architecture, dtype and quantization must
all match. Checksum or identity failures terminate the stream with
`GenerationFailed` and `LENS_INCOMPATIBLE`. Cancellation, timeout, model-load,
out-of-memory and concurrency failures are likewise terminal typed events.

The ordinary `rai serve` process does not import Transformers or Torch, and it
continues to operate when this optional process is absent or fails.

## Fitting a lens

Fitting is an offline, accelerator-intensive operation and is not exposed over
the sidecar API. RAI pins the Apache-2.0 reference implementation to commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e`. Install it separately so ordinary
sidecar installations do not need the fitting code:

```bash
uv sync --extra inference-jlens
uv run rai neural fit-lens \
  --model org/model \
  --revision MODEL_COMMIT_SHA \
  --lens-id model-jlens-v1 \
  --lens-revision fit-001 \
  --prompts calibration-prompts.json \
  --corpus-id corpus-version \
  --corpus-license corpus-license \
  --output "$XDG_DATA_HOME/rai/neural/lenses/model-jlens-v1"
```

The prompts file is a non-empty JSON array of strings. Options such as
`--layer`, `--dim-batch`, `--max-seq-len`, and `--skip-first` control the
reference fitter and are recorded in the manifest. A resumable pickle
checkpoint may exist while fitting, but successful output is converted to
checksummed `safetensors` and the checkpoint is removed. The serving process
never loads pickle.
