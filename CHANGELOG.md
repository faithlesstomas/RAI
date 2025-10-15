# Changelog

## [Unreleased]

### Added
- `--debug` flag to enable debug logging for the application and the `python-gitlab` library.
- `GitlabTools` to interact with the GitLab API.
- Debug logging to `GitlabTools` methods.

### Fixed
- `TypeError: 'Function' object is not callable` by correctly implementing `GitlabTools` as a `Toolkit`.
- `requests.exceptions.ChunkedEncodingError: Response ended prematurely` in `list_projects` by using an iterator.
- Issue with loading `GITLAB_BASE_URL` from `.env` file by stripping quotes and trailing slashes.
- Logic for checking required environment variables for `GitlabTools`.
