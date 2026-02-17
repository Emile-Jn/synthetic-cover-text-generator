"""
Convert terminal output to json file for analysis.

This module provides convert_txt_to_json which reads a text file where each
line is a Python dict literal (for example:
{"loss": 2.9137, 'grad_norm': 1.36, 'learning_rate': 3.24e-07, 'epoch': 2.0})
and writes either a JSON array or newline-delimited JSON (NDJSON).
"""

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_line_to_obj(line: str) -> Any:
    """Safely parse a single line containing a Python literal using ast.literal_eval.

    Returns the parsed object (commonly a dict). Raises ValueError if parsing fails.
    """
    try:
        # ast.literal_eval safely evaluates Python literals (dicts, lists, numbers, strings)
        return ast.literal_eval(line)
    except Exception as exc:  # keep broad to rewrap as ValueError for callers
        raise ValueError(f"Failed to parse line: {line!r}") from exc


def convert_txt_to_json(
    input_path: str,
    output_path: Optional[str] = None,
    newline_delimited: bool = False,
    encoding: str = "utf-8",
    skip_invalid: bool = True,
) -> List[Dict]:
    """Convert a TXT file where each line is a Python dict to JSON.

    Args:
        input_path: Path to the input .txt file.
        output_path: If provided, the JSON will be written to this path. If not,
            the parsed list is still returned.
        newline_delimited: If True, write NDJSON (one JSON object per line).
            Otherwise write a single JSON array.
        encoding: File encoding when reading/writing.
        skip_invalid: If True, skip lines that fail to parse; otherwise raise.

    Returns:
        A list of parsed Python dicts (or other literals if present).

    Raises:
        ValueError: If a line fails to parse and skip_invalid is False.
    """
    inp = Path(input_path)
    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    parsed: List[Dict] = []
    with inp.open("r", encoding=encoding) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = _parse_line_to_obj(line)
            except ValueError as exc:
                if skip_invalid:
                    # skip and continue
                    continue
                else:
                    raise
            parsed.append(obj)

    if output_path:
        outp = Path(output_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        if newline_delimited:
            # Write NDJSON: one JSON object per line
            with outp.open("w", encoding=encoding) as fh:
                for obj in parsed:
                    fh.write(json.dumps(obj, ensure_ascii=False))
                    fh.write("\n")
        else:
            with outp.open("w", encoding=encoding) as fh:
                json.dump(parsed, fh, indent=2, ensure_ascii=False)

    return parsed


# CLI entrypoint so the module can be run as a script
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert per-line Python-literal txt to JSON")
    parser.add_argument("input", help="Input text file (one Python dict per line)")
    parser.add_argument("output", nargs="?", help="Output JSON file path (if omitted, prints to stdout)")
    parser.add_argument("--ndjson", action="store_true", help="Write newline-delimited JSON (one JSON object per line)")
    parser.add_argument("--encoding", default="utf-8", help="File encoding")
    parser.add_argument("--no-skip-invalid", dest="skip_invalid", action="store_false", help="Do not skip invalid lines (raise instead)")

    args = parser.parse_args()

    results = convert_txt_to_json(args.input, args.output, newline_delimited=args.ndjson, encoding=args.encoding, skip_invalid=args.skip_invalid)

    if not args.output:
        # Print compact JSON to stdout
        print(json.dumps(results, ensure_ascii=False, indent=2))
