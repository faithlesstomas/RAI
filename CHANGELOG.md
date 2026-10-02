# Changelog

<!-- version list -->

## v0.11.0 (2026-10-02)

### Bug Fixes

- **ci**: Restore security tests across supported Python versions
  ([`1a689b1`](https://gitlab.com/tk-lab1/ai/rai/-/commit/1a689b1a19a4618b40db6039e192aa0e824cfddb))

- **release**: Synchronize lockfile before tagging releases
  ([`ce3e549`](https://gitlab.com/tk-lab1/ai/rai/-/commit/ce3e54996d2d0814bcead8a2b92c4557b2f72fd0))

- **security**: Harden hybrid inference boundaries
  ([`02eac8e`](https://gitlab.com/tk-lab1/ai/rai/-/commit/02eac8e0acb022853336f1dc97e30c4cb78f330b))

### Features

- **assistant**: Implement AntigravityAssistantModelBackend and Lemonade worker integration
  ([`3a07c98`](https://gitlab.com/tk-lab1/ai/rai/-/commit/3a07c98a5ee0e8acb32eba679c900167f2a6be27))

- **governor**: Implement InferenceBudgetGovernor, EgressFirewall, and usage accounting
  ([`7dc8a00`](https://gitlab.com/tk-lab1/ai/rai/-/commit/7dc8a00d30ebab83a92796f05d95b5f72d26bb6b))

- **routing**: Implement DecisionBackend, HybridRouter, and adversarial red-teaming test suite
  ([`3166b5b`](https://gitlab.com/tk-lab1/ai/rai/-/commit/3166b5be9f68885c718d5bf1bb6d958c877ce031))

- **security**: Harden tool execution, register gitlab/client capabilities, and wrap untrusted
  content
  ([`b126132`](https://gitlab.com/tk-lab1/ai/rai/-/commit/b126132c61d59bb93af1f36a8a52c80baee3c97a))


## v0.10.0 (2026-10-02)

### Bug Fixes

- **assistant**: Catch inference timeout on python 3.10
  ([`29a42e7`](https://gitlab.com/tk-lab1/ai/rai/-/commit/29a42e72aa510c33f3bfa598ac072a564dd2cd20))

- **assistant**: Complete chat template conformance
  ([`8eecf12`](https://gitlab.com/tk-lab1/ai/rai/-/commit/8eecf12747be1d0028afdd0e13f7b6695e98bf0a))

- **assistant**: Complete M4-M6 memory audit fixes and update v2 benchmark evidence
  ([`b6e0e0b`](https://gitlab.com/tk-lab1/ai/rai/-/commit/b6e0e0b229808c726fb4492e09fcae372346a6eb))

- **assistant**: Complete M4-M6 memory audit fixes and update v2 benchmark evidence
  ([`ede282b`](https://gitlab.com/tk-lab1/ai/rai/-/commit/ede282b0755ab60cb2c2349cd5fc77181e9a6a9b))

- **assistant**: Expand judge abstention markers for natural Polish phrasing
  ([`a707da9`](https://gitlab.com/tk-lab1/ai/rai/-/commit/a707da9b57439162f485ce1f3ed32f254ae48949))

- **assistant**: Harden context rot benchmark protocol
  ([`c1a88c0`](https://gitlab.com/tk-lab1/ai/rai/-/commit/c1a88c05d50ea3b94ef593973fb4569de3f00006))

- **assistant**: Harden local backend boundaries
  ([`a24cad7`](https://gitlab.com/tk-lab1/ai/rai/-/commit/a24cad76137bc92c3d4006ee86730eaa254e3a25))

- **client**: Validate Emacs package in CI
  ([`1418f11`](https://gitlab.com/tk-lab1/ai/rai/-/commit/1418f11b95a042b1e7294dcd79cbfe5a27b1a029))

- **emacs**: Support context snapshots on Emacs 28
  ([`8c44cde`](https://gitlab.com/tk-lab1/ai/rai/-/commit/8c44cde5e8c569d4813ee301150bc9059bc67e7b))

- **inference**: Handle reasoning_content and display tokens_out in context rot report
  ([`1a63b35`](https://gitlab.com/tk-lab1/ai/rai/-/commit/1a63b35773745e2b4b02846ce658eb314388d750))

- **inference**: Prioritize chat completions in LemonadeEngine and update conversational memory
  roadmap
  ([`3cd67b7`](https://gitlab.com/tk-lab1/ai/rai/-/commit/3cd67b7a1cdba2db2ffe7f443702b99808b80970))

### Features

- **assistant**: Add context rot and long-context needle-in-a-haystack benchmark
  ([`c55a661`](https://gitlab.com/tk-lab1/ai/rai/-/commit/c55a661b22dd664c3baf01ec4eea337384e36e37))

- **assistant**: Add evidence-first general memory
  ([`af761bf`](https://gitlab.com/tk-lab1/ai/rai/-/commit/af761bf104e404d5108446ce65cadef5a46351ce))

- **assistant**: Add grounded summary evaluation
  ([`641736b`](https://gitlab.com/tk-lab1/ai/rai/-/commit/641736b8fb0319bf434a55f1f6484ae87edd5c2c))

- **assistant**: Add versioned memory benchmark
  ([`e55c7f4`](https://gitlab.com/tk-lab1/ai/rai/-/commit/e55c7f4d789c3b10f2610e2980ab312d11316afd))

- **assistant**: Complete memory evaluation through M6
  ([`9312db7`](https://gitlab.com/tk-lab1/ai/rai/-/commit/9312db7a8bb6d8b26a189aba72d47a99b099370f))

- **assistant**: Deliver usable memory-backed MVP
  ([`16e40b6`](https://gitlab.com/tk-lab1/ai/rai/-/commit/16e40b6b1907b5e9e7fc9d4a8099a352ba83b81e))

- **assistant**: Evaluate memory sufficiency and hierarchical domain scopes
  ([`f7ca053`](https://gitlab.com/tk-lab1/ai/rai/-/commit/f7ca053083f441bb73300041117f192b4d42bf7c))

- **assistant**: Implement M7 structured chat templating and conversational benchmark
  ([`2dc795e`](https://gitlab.com/tk-lab1/ai/rai/-/commit/2dc795e743eed85bef3ede7e31c3e6e1c3b72c9e))

- **assistant**: Scope and evaluate memory retrieval
  ([`ba6f8e5`](https://gitlab.com/tk-lab1/ai/rai/-/commit/ba6f8e5bb145c6b23a4c98f4da6666be923c839f))

- **assistant**: Unify CoT reasoning architecture and update benchmarks up to 4B
  ([`e8a8f2e`](https://gitlab.com/tk-lab1/ai/rai/-/commit/e8a8f2e0a2a509f86f1d92f4784aa9718555f2a7))

- **client**: Add provider-neutral Emacs assistant package
  ([`84ba90e`](https://gitlab.com/tk-lab1/ai/rai/-/commit/84ba90e3ed32ea69c183aa0863ab7a984f2629ab))

- **inference**: Add Lemonade Server multimodal backend, speech adapters, and neural retrieval
  ([`17abfce`](https://gitlab.com/tk-lab1/ai/rai/-/commit/17abfcedb4f8bbfd8ff2f07ade5d812acf3a6630))

### Breaking Changes

- **assistant**: Remove the ambiguous local backend alias; use llama, ollama, or lemonade
  explicitly.


## v0.9.0 (2026-09-15)

### Bug Fixes

- **assistant**: Resolve ruff security issues S101 and S608 in service and store
  ([`3605fc2`](https://gitlab.com/tk-lab1/ai/rai/-/commit/3605fc21c537bb9c804d13cc92472cf88e52c637))

### Features

- **assistant**: Complete local graph-memory MVP
  ([`922c5f0`](https://gitlab.com/tk-lab1/ai/rai/-/commit/922c5f053313a38b0ae3b67da200338c4923747c))

- **assistant**: Freeze schemas, records, and conformance fixtures for graph memory
  ([`5be3bf7`](https://gitlab.com/tk-lab1/ai/rai/-/commit/5be3bf7ae5b65083b27b1c555550717d630dc145))

- **assistant**: Implement deterministic graph-memory vertical path and service
  ([`691a1cb`](https://gitlab.com/tk-lab1/ai/rai/-/commit/691a1cb7cf87742d6cecf90369fbb982baf5fc6d))

- **assistant**: Implement local backend, CLI surface, and 8-step acceptance test
  ([`93c3aea`](https://gitlab.com/tk-lab1/ai/rai/-/commit/93c3aead361576a130e9bf4498546cc58d0571f3))

- **assistant**: Make local chat usable with personal memory
  ([`3f7b726`](https://gitlab.com/tk-lab1/ai/rai/-/commit/3f7b72691d5ba2d9e59f386db1e3112d2f1bc32e))

- **inference**: Cache bounded local task results
  ([`44b9219`](https://gitlab.com/tk-lab1/ai/rai/-/commit/44b9219ca67310e5945ac0268a64d2329db4a759))


## v0.8.1 (2026-09-10)

### Bug Fixes

- **release**: Authenticate glab with CI job token
  ([`694a5c0`](https://gitlab.com/tk-lab1/ai/rai/-/commit/694a5c0100da39b4ee61bce5989ad9b71fecc8f1))


## v0.8.0 (2026-09-10)

### Bug Fixes

- **history**: Reconnect GNOME monitor across sessions
  ([`122b2ba`](https://gitlab.com/tk-lab1/ai/rai/-/commit/122b2baf523790d44550801df2fe02f33629cfce))

- **history**: Validate live GNOME collectors
  ([`137202d`](https://gitlab.com/tk-lab1/ai/rai/-/commit/137202dc6ca189708155e3446bcac4f84b87f147))

- **release**: Attach tag publishing to release branch
  ([`9499026`](https://gitlab.com/tk-lab1/ai/rai/-/commit/9499026fbedfbcfb6f4b5708a8d4ddc08a2b2e94))

- **release**: Publish GitLab assets with glab
  ([`dfaff19`](https://gitlab.com/tk-lab1/ai/rai/-/commit/dfaff19b1b62a5e593ea4e5a92e8a36ed95c845d))

- **tests**: Restore speech synthesis lint
  ([`5aaa720`](https://gitlab.com/tk-lab1/ai/rai/-/commit/5aaa7206c5a430537e628daaa2e3fffb07a9d1dd))

### Features

- **speech**: Expose speech.synthesize capability via MCP and default runtime
  ([`c26d666`](https://gitlab.com/tk-lab1/ai/rai/-/commit/c26d666d2fdefd961ca9a0974dc890938a9fc251))


## v0.7.0 (2026-09-09)

### Bug Fixes

- **ci**: Restore Python 3.10 compatibility
  ([`70b4a55`](https://gitlab.com/tk-lab1/ai/rai/-/commit/70b4a55b42579a9e77e13e695eb8b39b69ee8018))

- **release**: Repair changelog generation and simplify pipeline
  ([`1ae74ff`](https://gitlab.com/tk-lab1/ai/rai/-/commit/1ae74ffb65988767fe643a909e79a990e985b1ab))

### Features

- **voice**: Add profile-driven speech synthesis
  ([`01ef1a9`](https://gitlab.com/tk-lab1/ai/rai/-/commit/01ef1a9b6efac9442a94b1852f95f621fc0619f2))


## v0.6.0 (2026-09-08)

### Added
- Added a lifecycle-managed local processor supervisor with bounded concurrency,
  cancellation, non-blocking llama.cpp execution, Ollama support, idle model
  unloading, typed failures and provenance-preserving episode processing.
- Added the first Stage 4.2 bounded local-task contracts for validated episode
  summarization and non-authoritative intent classification.
- Completed the Stage 4.2 bounded-task catalog with provenance-bound entity
  extraction, deterministic salience estimation, monotonic privacy-risk
  elevation and non-authoritative routing hints.

### Security
- Replaced the compatibility calculator's Python evaluator with a bounded
  arithmetic parser and made Ruff security checks blocking in CI.
- Removed shell-based screenshot delays, secured temporary screenshot paths,
  bounded compatibility-client connection timeouts and documented narrowly
  reviewed security-rule exceptions at trusted process and SQL boundaries.

## [0.5.0] - 2026-09-07

### Added
- Added the Stage 3 Rich History acceptance slice: opt-in supervised semantic
  collectors, a deterministic pre-persistence privacy firewall, event fusion,
  reproducible episodes, AES-GCM local storage, provenance queries and verified
  cascading deletion through an authenticated local API.
- Added bundled GNOME Shell/D-Bus, AT-SPI, foreground-process and approved-root
  filesystem sidecars, including persistent opt-in configuration and an
  installer for the GNOME extension.
- Added coordinated periodic retention, bounded ephemeral raw-event buffering,
  deletion presets and Episode schema `1.1.0` compatibility coverage.
- Added the Stage 2 durable local event plane with transactional ordering,
  idempotent ingest, replay cursors, consumer acknowledgements, bounded
  subscriptions, authenticated HTTP/Unix-socket transports and the
  provider-neutral deterministic acceptance subscriber.
- Added reproducible wheel and source-distribution validation, a clean-wheel
  smoke test, Python 3.10-3.12 CI coverage, an explicit alpha-preview release
  job, and GitLab OIDC trusted-publishing jobs for TestPyPI and PyPI.

### Changed
- Selected `rich-ai` as the PyPI distribution name while retaining `rai` for
  the repository, Python namespace, CLI, configuration and protocol identity.
- Repositioned RAI as the secure, local-first integration layer between AI
  assistants and Linux, with GAIA, GCAS compatibility, Antigravity and J-lens
  explicitly outside the standalone product's required core.
- Moved Google Antigravity and GitLab tooling out of the base installation and
  corrected optional extras so they cannot resolve to the unrelated PyPI
  project named `rai`.

### Fixed
- Closed browser origin/URL and AT-SPI `file://` policy bypasses, removed
  private observation bodies from the plaintext event journal, bounded sidecar
  stream buffering and made collector health reflect persistence failures.
- Made GitLab tooling initialize without network access and handle authentication
  timeouts without making the test suite depend on GitLab availability.
- Simplified GitLab releases to one calculated preview job and one stable job,
  with a preflight guard that rejects version regressions and releases outside
  the active pre-release line.
- Removed timing sensitivity from the Rich History retention acceptance test.

## [0.3.1] - 2026-09-04

### Fixed
- Made GitLab CI install locked dependencies through `uv`, kept Pages
  independent of the manual release job, removed the unused Sphinx static path
  and updated Antigravity conversation fixtures for the current SDK contract.

## [0.3.0] - 2026-09-04

### Changed
- Reframed RAI as a local-first agent runtime for Linux and made Antigravity a
  transitional compatibility backend.
- Expanded the implementation roadmap for Rich History, local voice and AI,
  safe desktop actions, hybrid backends, token budgets and privacy controls.
- Documented the SemVer release, testing and Sphinx/GitLab Pages workflow.
- Updated the semantic-release configuration for current version stamping,
  pre-1.0 development and GitLab `vX.Y.Z` releases.
- Adopted XDG-compliant configuration, data, cache and runtime directories.
- Made conversation history independent of thread-based database helpers.
- Moved heavyweight inference dependencies into optional extras.

### Security
- Removed package-import monkeypatches of Antigravity and `subprocess.Popen`.
- Changed sandbox selection to fail closed when Bubblewrap/Guix is unavailable.
- Added per-user token authentication and disabled CORS by default.
- Added ignore rules for loose credential files and local model weights.

### Added
- Added the Stage 1 embodiment kernel: immutable versioned domain records,
  structured provenance, language-neutral JSON Schema and conformance fixtures.
- Added provider-neutral runtime ports, cancellation/lifecycle contracts and
  deterministic synthetic reference implementations.
- Added a typed capability registry, four-outcome policy engine, shared
  CLI/REST/MCP invocation envelope and durable capability audit ledger.
- Added an application container and FastAPI app factory.
- `--debug` flag to enable debug logging for the application and the `python-gitlab` library.
- `GitlabTools` to interact with the GitLab API.
- Debug logging to `GitlabTools` methods.

### Fixed
- Made `rai --version` use the package version instead of a stale hard-coded
  value.
- `TypeError: 'Function' object is not callable` by correctly implementing `GitlabTools` as a `Toolkit`.
- `requests.exceptions.ChunkedEncodingError: Response ended prematurely` in `list_projects` by using an iterator.
- Issue with loading `GITLAB_BASE_URL` from `.env` file by stripping quotes and trailing slashes.
- Logic for checking required environment variables for `GitlabTools`.
