"""One scripted `cloudgpu up`, then `cloudgpu down`, against the fake, and what they left behind.

    uv run --frozen --extra dev python -m tests.fake_lambda
    uv run --frozen --extra dev python -m tests.fake_lambda --launch start_then_502
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from . import LAUNCH_MODES, FakeLambda, run_cloudgpu, run_up


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
        if result.returncode != 0:
            return result.returncode
        down = run_cloudgpu(fake, Path(tmp), ["down", "--config", "box", "--yes"])
        print(down.output)
        print(f"cloudgpu down exit code: {down.returncode}")
        print(f"fake saw terminates:     {fake.count('POST', '/terminate')}")
        print(f"fake's live instances:   {fake.alive()}")
        print(f"local state tracks:      {down.state.get('instance_id')}")
        if down.returncode == 0 and fake.alive():
            print("FAILED: down exited 0 with an instance still alive")
            return 1
    return down.returncode


if __name__ == "__main__":
    sys.exit(main())
