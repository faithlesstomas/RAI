# Changelog

## [Unreleased]

### Changed
- Reframed RAI as a local-first agent runtime for Linux and made Antigravity a
  transitional compatibility backend.
- Adopted XDG-compliant configuration, data, cache and runtime directories.
- Made conversation history independent of thread-based database helpers.
- Moved heavyweight inference dependencies into optional extras.

### Security
- Removed package-import monkeypatches of Antigravity and `subprocess.Popen`.
- Changed sandbox selection to fail closed when Bubblewrap/Guix is unavailable.
- Added per-user token authentication and disabled CORS by default.
- Added ignore rules for loose credential files and local model weights.

### Added
- `--debug` flag to enable debug logging for the application and the `python-gitlab` library.
- `GitlabTools` to interact with the GitLab API.
- Debug logging to `GitlabTools` methods.

### Fixed
- `TypeError: 'Function' object is not callable` by correctly implementing `GitlabTools` as a `Toolkit`.
- `requests.exceptions.ChunkedEncodingError: Response ended prematurely` in `list_projects` by using an iterator.
- Issue with loading `GITLAB_BASE_URL` from `.env` file by stripping quotes and trailing slashes.
- Logic for checking required environment variables for `GitlabTools`.
