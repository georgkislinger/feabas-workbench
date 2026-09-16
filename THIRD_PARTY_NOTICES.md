# Third-party notices

The workbench itself is Copyright 2026 Georg Kislinger and licensed under the Apache License 2.0
(`LICENSE`); its author attribution is in `NOTICE`. What follows are the notices of bundled third-party
software.

## FEABAS

This repository vendors the driver scripts, tools and default configuration files of
**FEABAS 3.0.5**, unmodified, under `feabas_workbench/vendor/feabas_3_0_5/`. FEABAS does the actual
stitching and alignment; the workbench is a front end for it.

* Project: <https://github.com/YuelongWu/feabas>
* Copyright (c) 2022 Yuelong Wu, Center for Brain Science, Harvard University
* License: MIT – see [`feabas_workbench/vendor/feabas_3_0_5/LICENSE`](feabas_workbench/vendor/feabas_3_0_5/LICENSE)

FEABAS itself is installed separately, as a Python package, into an environment of your choice; the
workbench never modifies that installation. `feabas_workbench/vendor/winfix/sitecustomize.py` applies two
run-time fixes to FEABAS 3.0.5 from the outside (see the comments in that file).
