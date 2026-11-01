import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _export() -> object:
    spec = importlib.util.spec_from_file_location(
        "export_openapi", ROOT / "tools/export_openapi.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OpenApiExport()


def test_committed_spec_matches_the_code() -> None:
    export = _export()
    assert export.is_current(), (
        "docs/openapi.json is out of date: run .venv/bin/python tools/export_openapi.py"
    )


def test_spec_covers_the_main_routes() -> None:
    import json

    paths = json.loads(_export().text())["paths"]
    for route in ("/api/chat", "/api/generate", "/v1/chat/completions", "/v1/models", "/metrics"):
        assert route in paths, route
