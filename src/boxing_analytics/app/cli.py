"""Package CLI entrypoint for diagnostics and phased workflows."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from boxing_analytics.app.positioning import full_disclaimer
from runtime_profile import runtime_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="boxing-analytics",
        description="Boxing analytics decision support CLI.",
    )
    parser.add_argument(
        "--print-disclaimer",
        action="store_true",
        help="Print mandatory decision-support disclaimer.",
    )
    parser.add_argument(
        "--print-runtime-profile",
        action="store_true",
        help="Print detected runtime profile and accelerator settings.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.print_disclaimer:
        print(full_disclaimer())
        return 0
    if args.print_runtime_profile:
        print(runtime_summary())
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
