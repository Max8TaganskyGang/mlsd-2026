"""stage_input: validate the request, verify the history hash, copy both to S3."""

from __future__ import annotations

from pathlib import Path

from .common import (
    INPUT_ROOT,
    SPLITS,
    StageError,
    require_full_calendar,
    sha256_file,
    strict_loads,
    validate_request,
)
from .storage import RunStore


def resolve_input(relative: str) -> Path:
    path = (INPUT_ROOT / relative).resolve()
    if not path.is_relative_to(INPUT_ROOT.resolve()) or not path.is_file():
        raise StageError("INVALID_REQUEST", f"input file /input/{relative} is not available")
    return path


def load_request_file(request_path: str, execution_id: str) -> tuple[dict, bytes]:
    path = Path(request_path or "").resolve()
    if not request_path or not path.is_relative_to(INPUT_ROOT.resolve()) or not path.is_file():
        raise StageError("INVALID_REQUEST", "conf.request_path must point to a file in /input")
    raw = path.read_bytes()
    try:
        request = strict_loads(raw)
    except (ValueError, UnicodeError) as error:
        raise StageError("INVALID_REQUEST", "request.json is not valid JSON") from error
    return validate_request(request, execution_id), raw


def run(execution_id: str, request_path: str) -> None:
    store = RunStore(execution_id)
    store.ensure_bucket()
    # A retry of this execution must not leave a stale ready result or refusal behind.
    store.delete("result.json")
    store.delete("error.json")
    request, raw = load_request_file(request_path, execution_id)
    if len(request["snapshots"]) == len(SPLITS):
        require_full_calendar(request)
    history = resolve_input(request["input_path"])
    actual = sha256_file(history)
    if actual != request["input_sha256"]:
        raise StageError(
            "INPUT_HASH_MISMATCH",
            f"sha256 of /input/{request['input_path']} is {actual}, request declares "
            f"{request['input_sha256']}",
        )
    store.delete_prefix("input/")
    store.put_file("input/input.csv.gz", history)
    store.put_bytes("input/request.json", raw)
    print(f"staged input {request['input_path']} sha256={actual} for {execution_id}")
