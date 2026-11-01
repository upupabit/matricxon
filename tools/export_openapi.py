"""Writes the API spec (OpenAPI 3) to docs/openapi.json, the file published with each release.

    .venv/bin/python tools/export_openapi.py          # rewrite it
    .venv/bin/python tools/export_openapi.py --check  # exit 1 if it is out of date

tests/unit/test_openapi_spec.py fails while the committed file differs from the code.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.main import MatricxonApp  # noqa: E402

SPEC_PATH = ROOT / "docs" / "openapi.json"


class OpenApiExport:
    def __init__(self, path: Path = SPEC_PATH) -> None:
        self.path = path

    @staticmethod
    def text() -> str:
        spec = MatricxonApp().build().openapi()
        return json.dumps(spec, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    def is_current(self) -> bool:
        return self.path.exists() and self.path.read_text() == self.text()

    def write(self) -> None:
        self.path.write_text(self.text())


if __name__ == "__main__":
    export = OpenApiExport()
    if "--check" in sys.argv:
        sys.exit(0 if export.is_current() else 1)
    export.write()
    print(f"wrote {export.path}")
