"""One scripted `cloudgpu up` against the fake, and what it left behind.

    uv run --frozen --extra dev python -m tests.fake_lambda
    uv run --frozen --extra dev python -m tests.fake_lambda --launch start_then_502
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from . import LAUNCH_MODES, FakeLambda, run_up


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.fake_lambda")
    parser.add_argument("--launch", action="append", default=[], choices=sorted(LAUNCH_MODES),
                        help="what the next launch does; repeat for later ones. After these, launches succeed.")
    parser.add_argument("--then", default="ok", choices=sorted(LAUNCH_MODES), help="what every later launch does")
    parser.add_argument("--boot-polls", type=int, default=0, help="reads a launched instance answers as booting")
    parser.add_argument("--max-hours", type=float, default=1.0, help="how long up may poll")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp, FakeLambda(
        launches=args.launch, then=args.then, boot_polls=args.boot_polls
    ) as fake:
        result = run_up(fake, Path(tmp), max_hours=args.max_hours)
        print(result.output)
        print(f"cloudgpu up exit code: {result.returncode}")
        print(f"fake saw launches:     {len(fake.launches)}")
        print(f"fake's live instances: {fake.alive()}")
        print(f"local state tracks:    {result.state.get('instance_id')}")
        print(f"ssh/rsync shim calls:  {len(result.shim_calls)}, hosts {sorted({c['host'] for c in result.shim_calls})}")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
