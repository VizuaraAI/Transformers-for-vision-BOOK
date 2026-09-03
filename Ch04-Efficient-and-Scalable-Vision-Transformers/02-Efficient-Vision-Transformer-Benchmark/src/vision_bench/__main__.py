"""Allow ``python -m vision_bench`` to behave like ``vision-bench``."""

from vision_bench.cli import main

raise SystemExit(main())
