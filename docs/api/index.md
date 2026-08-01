# API reference

This reference is generated from `pylibfreenect3.__all__` and
`pylibfreenect3.lowlevel.__all__` at build time, then rendered from the source
annotations and docstrings with mkdocstrings. The generator reads the Python
source statically, so building the documentation does not require compiling or
loading the native extension.

Every backend class is importable on every supported platform. Constructing an
uncompiled or runtime-unusable backend raises `BackendUnavailableError`.

`Stream` and `Pipeline` are string enums. Canonical string values remain
accepted at API boundaries and are normalized to their enum member.

Packet parser/processor hooks, decoder-thread callbacks into Python, and a
Python logger callback are intentionally excluded for thread safety. Native
console logging is available through `set_global_log_level`.

{{ public_api_reference() }}
