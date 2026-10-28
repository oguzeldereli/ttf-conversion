# Legacy Qwen checks

`qwen_check.py`, `qwen_stream_check.py` and `qwen_calib_domain.py` were in ttf-core's `scripts/quant` up to
d5f6256 (27 Oct 2026). They call ttf-core's old in-core conversion: `ttf.qwen.load_qwen`, `calib_hessians_qwen`
and `convert_qwen_streaming`. That code left the core on 28 Oct and lives here as `ttfconv.stream`.

The scripts are kept unchanged as the record of those measurements. To run them, use a ttf-core checkout at d5f6256:

    PYTHONPATH=/path/to/ttf-core@d5f6256 python scripts/legacy/qwen_check.py ...

For new work, use `python -m ttfconv.convert --eval wiki@FILE@N@STRIDE` instead: it converts and scores
fixed windows in one run (`qwen_check`, `qwen_calib_domain`). `qwen_stream_check` compared the in-memory path
against the streamed one. Only the streamed path exists now, so that check has nothing left to compare.
