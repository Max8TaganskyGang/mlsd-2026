"""Stage entry point: python -m hw3pipeline <stage> --execution-id ID [--request-path PATH]."""

from __future__ import annotations

import argparse
import sys

from .common import EXECUTION_ID, StageError

STAGES = ("stage_input", "build_examples", "train_select", "evaluate", "publish")
EXIT_REFUSED = 3


def record_refusal(execution_id: str, stage: str, error: StageError) -> None:
    from .storage import RunStore

    try:
        RunStore(execution_id).put_json(
            "error.json", {"stage": stage, "error_code": error.code, "message": error.message}
        )
    except Exception as exc:  # the refusal itself is the result; storage loss is reported only
        print(f"could not store error.json: {type(exc).__name__}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(prog="hw3pipeline")
    parser.add_argument("stage", choices=(*STAGES, "ensure_bucket"))
    parser.add_argument("--execution-id", default="")
    parser.add_argument("--request-path", default="")
    args = parser.parse_args()

    if args.stage == "ensure_bucket":
        from .storage import ensure_bucket

        ensure_bucket()
        return 0
    if not EXECUTION_ID.fullmatch(args.execution_id):
        print("INVALID_REQUEST: dag_run_id is not a valid execution_id", file=sys.stderr)
        return EXIT_REFUSED
    try:
        if args.stage == "stage_input":
            from . import stage_input

            stage_input.run(args.execution_id, args.request_path)
        elif args.stage == "build_examples":
            from . import build_examples

            build_examples.run(args.execution_id)
        elif args.stage == "train_select":
            from . import train_select

            train_select.run(args.execution_id)
        elif args.stage == "evaluate":
            from . import evaluate

            evaluate.run(args.execution_id)
        else:
            from . import publish

            publish.run(args.execution_id)
    except StageError as error:
        print(f"{error.code}: {error.message}", file=sys.stderr)
        record_refusal(args.execution_id, args.stage, error)
        return EXIT_REFUSED
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
