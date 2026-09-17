"""Rewrite README.md for the PyPI project page: relative links become absolute GitHub URLs.

GitHub resolves ``docs/img/x.png`` and ``[guide](docs/USER_GUIDE.md)`` against the repository; PyPI
renders the same README as the project description but leaves relative targets as they are, so every
image and link would be dead there. The release workflow runs this on its throw-away checkout just
before ``python -m build``, pinned to the tag being released:

    python tools/absolutize_readme.py v0.3.2

Images go through raw.githubusercontent.com, everything else through github.com/.../blob/<ref>/.
GitHub's ``> [!TIP]`` alert syntax, which PyPI prints literally, becomes a bold lead-in. The file in
the repository stays relative; only the built sdist and wheel carry the rewritten copy.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = "georgkislinger/feabas-workbench"
README = Path(__file__).resolve().parents[1] / "README.md"

_RELATIVE = re.compile(r"^(?![a-zA-Z][a-zA-Z0-9+.-]*:|#|//)")   # not a scheme, not an anchor, not protocol-relative
_ALERT = re.compile(r"^> \[!(\w+)\]\n> ", re.MULTILINE)


def absolutize(text: str, ref: str, repo: str = REPO) -> str:
    blob = f"https://github.com/{repo}/blob/{ref}/"
    raw = f"https://raw.githubusercontent.com/{repo}/{ref}/"

    def html_attr(m: re.Match) -> str:
        name, target = m.group(1), m.group(2)
        if not _RELATIVE.match(target):
            return m.group(0)
        return f'{name}="{raw if name == "src" else blob}{target}"'

    def md_link(m: re.Match) -> str:
        bang, label, target = m.group(1), m.group(2), m.group(3)
        if not _RELATIVE.match(target):
            return m.group(0)
        return f"{bang}[{label}]({raw if bang else blob}{target})"

    text = re.sub(r'\b(src|href)="([^"]+)"', html_attr, text)
    text = re.sub(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)", md_link, text)
    text = _ALERT.sub(lambda m: f"> **{m.group(1).capitalize()}:** ", text)
    return text


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    README.write_text(absolutize(README.read_text(encoding="utf-8"), argv[1]), encoding="utf-8")
    print(f"README.md links now point at {REPO}@{argv[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
