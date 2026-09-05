# Rich AI documentation

**Rich AI (RAI)** is the secure, local-first integration layer between AI
assistants and Linux. It gives replaceable local or external agents access to
explicitly allowed perception and typed, policy-controlled capabilities while
keeping durable state outside model contexts.

RAI is designed to reduce both operating cost and data exposure: deterministic
operations and small local models handle routine work first, while larger local
or external agents receive only the minimum context needed for harder tasks.
Versioned budget and usage contracts make token, provider-cost and local-resource
accounting consistent across local-only, hybrid and external modes.

The longer-term product combines private activity history, bounded local AI and
safe Linux actions behind one provider-neutral interface. This lets users change
models or agent harnesses without giving each of them separate ambient desktop
access or losing policy, approval, provenance and audit controls. The roadmap
distinguishes these planned product outcomes from the kernel capabilities that
are already implemented.

Start with the repository-root `README.md`, `ROADMAP.md`, `SECURITY.md` and
`CONTRIBUTING.md`, then read the [architecture](architecture.md) document.

The API reference describes the transitional implementation. Features described
in the roadmap are not assumed to exist until their acceptance gate is complete.

```{toctree}
:maxdepth: 2
:caption: Contents:

architecture
kernel-contracts
neural-sidecar
reference/modules
```
