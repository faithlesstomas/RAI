# TODO

- [ ] Diagnose why debug logs in `GitlabTools` methods are not showing up, even when the `--debug` flag is used.
- [ ] Investigate and fix `ResourceWarning: unclosed database` and `ResourceWarning: unclosed transport` that appear on application exit. This might be related to `prompt_toolkit` history or other network clients not being closed properly.