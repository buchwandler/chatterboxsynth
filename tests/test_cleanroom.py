from __future__ import annotations

import ast
from pathlib import Path

_FORBIDDEN = {"chatterbox", "chatterbox_tts", "torch", "torchaudio", "transformers"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".", 1)[0].casefold().replace("-", "_"))
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".", 1)[0].casefold().replace("-", "_"))
        elif isinstance(node, ast.Call):
            function = node.func
            is_dynamic = isinstance(function, ast.Name) and function.id == "__import__"
            is_dynamic = is_dynamic or (
                isinstance(function, ast.Attribute) and function.attr == "import_module"
            )
            if is_dynamic and node.args and isinstance(node.args[0], ast.Constant):
                if isinstance(node.args[0].value, str):
                    found.add(node.args[0].value.split(".", 1)[0].casefold().replace("-", "_"))
    return found & _FORBIDDEN


def test_runtime_and_onnxvoice_chatterbox_modules_have_no_forbidden_imports() -> None:
    package_root = Path(__file__).resolve().parents[1]
    files = list((package_root / "chatterboxsynth").rglob("*.py"))
    onnxvoice_root = package_root.parent / "onnxvoice" / "onnxvoice"
    for relative in ("catalog_tools/chatterbox.py", "systems/chatterbox.py"):
        candidate = onnxvoice_root / relative
        if candidate.is_file():
            files.append(candidate)
    violations = {}
    for path in files:
        imports = _imports(path)
        if imports:
            violations[str(path)] = sorted(imports)
    assert not violations, f"forbidden runtime imports found: {violations}"
