"""Write docs/code_reference.md from the module and function docstrings of the rsfilter package."""
import importlib
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ["config", "data", "preprocess", "filters_spatial", "filters_freq", "bank", "metrics", "classify",
           "experiments", "viz", "cli"]


def first_paragraph(doc: str | None) -> str:
    return (inspect.cleandoc(doc).split("\n\n")[0].replace("\n", " ") if doc else "(no docstring)")


def main():
    lines = ["# Code reference", "",
             "Generated from the docstrings by `scripts/make_code_reference.py`. "
             "Read the source for the full explanation and formulas.", "",
             "| Module | Purpose |", "|---|---|"]
    mods = {m: importlib.import_module(f"rsfilter.{m}") for m in MODULES}
    for name, mod in mods.items():
        lines.append(f"| `{name}.py` | {first_paragraph(mod.__doc__)} |")
    for name, mod in mods.items():
        lines += ["", f"## `rsfilter/{name}.py`", "", first_paragraph(mod.__doc__), ""]
        members = [(n, o) for n, o in inspect.getmembers(mod, lambda o: inspect.isfunction(o) or inspect.isclass(o))
                   if o.__module__ == mod.__name__ and not n.startswith("_")]
        members.sort(key=lambda m: inspect.getsourcelines(m[1])[1])
        for n, obj in members:
            sig = "" if inspect.isclass(obj) else str(inspect.signature(obj))
            lines.append(f"- **`{n}{sig}`** {first_paragraph(obj.__doc__)}")
    out = ROOT / "docs" / "code_reference.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
