"""Command line entry point.

    uv run demo                    # list the demos
    uv run demo all                # run every demo in order
    uv run demo agents streaming   # run just these

By default everything runs against the offline `ScriptedChatModel`, so no server
and no API key are needed. To use a real model:

    LC_PROVIDER=ollama LC_MODEL=llama3.1 uv run demo all
"""

from __future__ import annotations

import sys

from langchain_project import config
from langchain_project.console import kv, note, title
from langchain_project.demos import BY_NAME, DEMOS


def _list() -> None:
    title("LANGCHAIN CORE COMPONENTS")
    note(config.describe_provider())
    print()
    for index, demo in enumerate(DEMOS, start=1):
        print(f"   {index}. {demo.name}")
        print(f"      {demo.blurb}")
        print(f"      {demo.doc}")
    print()
    print("   Run one:  uv run demo agents")
    print("   Run all:  uv run demo all")
    print()


def main() -> int:
    """Parse argv and run the requested demos."""
    args = [arg.lower() for arg in sys.argv[1:]]

    if not args or args[0] in {"list", "-h", "--help", "help"}:
        _list()
        return 0

    if args[0] == "all":
        chosen = list(DEMOS)
    else:
        unknown = [name for name in args if name not in BY_NAME]
        if unknown:
            print(f"Unknown demo(s): {', '.join(unknown)}")
            print(f"Available: {', '.join(BY_NAME)}")
            return 1
        chosen = [BY_NAME[name] for name in args]

    for demo in chosen:
        demo.run()

    print()
    kv("done", f"{len(chosen)} demo(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
