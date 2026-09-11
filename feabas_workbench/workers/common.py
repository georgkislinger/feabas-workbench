from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def progress(done: int, total: int, msg: str = "") -> None:
    print("##PROGRESS " + json.dumps({"done": int(done), "total": int(total), "msg": str(msg)}), flush=True)


def result(payload: dict) -> None:
    print("##RESULT " + json.dumps(payload, default=str), flush=True)


def log(msg: str) -> None:
    print(str(msg), flush=True)


def load_spec(description: str = "") -> dict:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--spec", required=True, help="JSON spec file")
    args = ap.parse_args()
    return json.loads(Path(args.spec).read_text(encoding="utf-8"))


def run(main_func) -> None:
    try:
        code = main_func()
        sys.exit(int(code or 0))
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)
