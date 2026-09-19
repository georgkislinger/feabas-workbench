# Use your normal Workbench workflow at LRZ

Local build **0.3.3+cluster.3**. Open your project on your PC, choose **Use cluster…**,
complete setup, save cluster settings, then use **Sync project & images**. The normal
Run buttons now submit work to LRZ. The green accents and the **Execution: LRZ cluster**
banner show which computer will do the computation. **Use this PC** restores local execution.

This build passed 133 automated tests. The preceding build also passed a real
13-command synthetic FEABAS pipeline, including rendering, mipmaps and VAST export.
See the accompanying live test report for account-specific checks. Image transfer
and full FEABAS execution at LRZ still require an assigned workspace and environment.
This is a local modification, not an upstream release.

## First live check: sign in and find storage

1. Restart Workbench after updating. Confirm **0.3.3+cluster.3** in the title/About dialog.
   The supplied Windows app includes SSH and the Globus CLI. The existing
   `start_gui.bat` environment has also been updated with Globus support.
2. Open the tutorial project. Click **Use cluster…** (also available as
   **Pipeline → Run on cluster…**).
3. In **1 Cluster setup**, enter your LRZ username and click **1. Sign in and find my storage**.
   Passwords and MFA are entered in private prompts; never paste them into chat.
   Check a new host fingerprint against [LRZ's published fingerprints](https://doku.lrz.de/access-and-login-to-the-linux-cluster-10745974.html).
4. If DSS folders appear, choose the allocation your institute permits you to use and
   click **2. Use this storage folder**. Workbench creates separate images and work
   folders belonging to this project. It verifies that the SSH and Globus paths agree.
5. Click **3. Prepare FEABAS at LRZ**. This creates a private Linux Python environment
   under the chosen allocation and installs FEABAS 3.0.5 and its CPU dependencies.
   If LRZ's default Python is too old, the message asks for a Python module in Advanced;
   we can resolve the available module after signing in.

If sign-in succeeds but there is no DSS allocation, stop at this point. Ask the institute's
LRZ project administrator for Linux Cluster access and DSS project storage with enough
quota for source images, intermediate data, outputs and retained previous results.
Creating an MFA token does not itself grant cluster or storage access.

The filesystem list in `dssusrinfo all` describes what the login node has mounted;
it does not grant you those directories. Workbench now lists only paths from the
**You have access to the following DSS Containers** section. If that section is
empty, no project container is listed for your account. A failed or unrecognized
report is shown as a discovery problem, rather than proof of no allocation.
Mounted roots such as `/dss/dssfs02` cannot be selected as project storage.
A separate tiny access test can use home without enabling production image uploads.

LRZ documents a 100 GB home quota; the proposed 1 TB dataset belongs in an appropriate
DSS allocation. Scratch has a 14-day retention policy and no backup.
See [LRZ storage instructions](https://doku.lrz.de/file-systems-and-io-on-linux-cluster-10745972.html).

## Connect the dataset and local result destination

1. In **2 Data transfer**, install and run **Globus Connect Personal** on this PC using
   the linked [official Windows instructions](https://docs.globus.org/globus-connect-personal/install/windows/).
   Allow access to the tutorial images, project folder and intended export destination.
2. Click **Sign in / grant Globus access in browser** and complete the browser flow.
   The official Globus CLI keeps its authentication in its own credential store;
   Workbench project settings do not contain passwords, MFA codes or access tokens.
3. Click **Find this computer's collection**. The PC UUID is also used as the image
   source unless you choose a different source collection.
4. For the institutional NAS, prefer an institution-managed Globus collection or an
   institute server that can access the NAS. Ask your administrator for its collection
   UUID and the dataset's path within that collection. Workbench does not automatically
   turn a NAS into a Globus server. A mapped Windows path and a Globus path can differ.
5. The LRZ collection is prefilled. Remote folders are filled by the storage step.
   Check the selected paths in **Open these locations in Globus**.
6. Leave **Automatically download after Export** enabled. Keep **Include existing
   project results** checked to resume a local project, or clear it for a fresh run.
7. Save cluster settings, then click **Sync project & images** in the top bar.
   Wait until the verified uploads finish and the banner says the images are synchronized.

Transfers use encryption and checksum verification. They copy changed files without
deleting extra destination files. Keep the source NAS/server and the local Globus service
available. The PC can disconnect after receiving the Slurm job ID; jobs keep running.
Globus transfers can continue independently of Workbench, provided their endpoints remain online.
See [Globus transfer options](https://docs.globus.org/cli/reference/transfer/).

If Globus requests collection permissions, click **Sign in / grant access** again. The
interface remembers the required consent scopes for that session. Failed transfers can
be retried under Advanced. If a request's outcome is unknown, recover its task UUID from
Globus Activity using the exact label; Workbench does not automatically duplicate it.

## Which LRZ resource to use

These are starting configurations in **3 Computing power**, not promises of account
entitlement or immediate capacity. Slurm queues work until the requested resources are available.

| Purpose | Partition | Total cores | RAM | Simultaneous sections | Workers per section |
|---|---|---:|---:|---:|---:|
| Tutorial / first connection test | serial_std | 4 | 16 GiB | 1 | 4 |
| CPU processing that fits in memory | cm4_tiny | 32 | 128 GiB | 2 | 16 |
| Large-memory starting preset | teramem_inter | 32 | 512 GiB | 2 | 16 |

For a first full-resolution memory measurement, change the large-memory preset to
**one simultaneous section**. After measuring peak memory, try two with the same
settings and compare total throughput. The displayed peak is summed process RSS;
shared memory can be counted more than once, so treat it as an estimate.

**Teramem** is the high-memory candidate for your images. It supports batch jobs through
`teramem_inter`, despite that name. It is a shared single node with roughly 6 TB physical
memory. See [LRZ's large-memory batch instructions](https://doku.lrz.de/running-large-memory-jobs-on-the-linux-cluster-1311572409.html).
The interface permits up to 96 cores and conservatively caps a Teramem request at 2900 GiB.
CoolMUC-4 `cm4_tiny` permits up to 112 physical cores on one node; this build caps its RAM
request at 244 GiB to respect LRZ's current per-user restriction. Availability and access
must be checked against [resource limits](https://doku.lrz.de/job-processing-on-the-linux-cluster-10745970.html)
and [current policies](https://doku.lrz.de/policies-on-the-linux-cluster-1307289651.html).

Your 9–24 tiles of 12k–40k pixels per side correspond to roughly 1.2–35.8 GiB of raw
pixels per section at 8 bits, or 2.4–71.5 GiB at 16 bits. Working arrays, image pyramids,
meshes and caches add memory. This is not a RAM requirement prediction. A 512 GiB request
is a benchmark starting point, not a guarantee that every processing stage fits.

Within-section workers and simultaneous sections are bounded together. Matching,
montage optimization/rendering and thumbnail mipmapping can run independent sections
concurrently. Whole-stack operations remain coordinated and can use the full CPU budget.
The implementation does not combine RAM across nodes and does not make FEABAS an MPI
application. A cluster job is not automatically faster than your 64-core/512 GB workstation.

## Run, inspect and export

Use the same preprocessing, stitching, masking, alignment and export pages. Batch workers
run at LRZ; interactive previews and editing remain on the PC. Settings and authored
masks are uploaded with each request. Large masks/checkpoints use Globus before computation
is submitted. Changing scientific settings archives affected remote outputs so FEABAS
does not silently reuse stale results. These archives consume DSS space.

Pipeline state, progress, logs and memory estimates return to the GUI. Small previews
arrive over SSH. Larger preview/model caches download through Globus; wait for that
transfer before editing or selecting those files. Local edits made during a job are
preserved when incoming previews conflict with them.

Export writes a new remote folder for each run, then downloads it into a matching local
subfolder. The GUI reports **downloaded and verified** before listing the local VAST file.
For the built-in full-resolution viewer or Neuroglancer, use **Download rendered stack
from LRZ for local viewing** on the Export page. This is a separate bulk download and
requires adequate local disk space; the app does not stream a TB-sized volume over SSH.

Workstation results and the cluster preview cache are kept separately. Closing Workbench
does not cancel a submitted job. Reopen the project, reconnect, and monitoring resumes.
Slurm is checked at most every ten minutes, following LRZ policy; log/status files and
Globus transfers are checked more often. Queued-job status may therefore be delayed.

## Scope and recovery

- The CPU pipeline and headless workers use the regular Run buttons. Optional deep-learning
  tools require a separately prepared remote environment in Advanced. These presets do not
  allocate GPUs, and deep-learning jobs have not been tested at LRZ.
- Main-project section ranges work remotely. The separate local test-run-folder workflow
  and restoring a saved snapshot are not yet supported in cluster mode; their messages
  explain the limitation. Remote snapshot creation and archiving cleared outputs are supported.
- Standard FEABAS HDF5 image roots, PNG metadata and TensorStore path specifications are
  relocated when an existing Windows project is mirrored. Unusual custom external output
  folders are rejected with a clear message; use the managed project folders.
- If submission is uncertain, reconnect and refresh to recover the saved job receipt.
  Do not create another job until the LRZ queue has been checked. The immutable request
  package and log are retained for diagnosis.
- Synthetic Zeiss-like data still uses filename/grid settings and explicit calibration.
  No changes were made to Zeiss microscope metadata parsing.

The next shared check is deliberately small: sign in, verify DSS, prepare the environment,
transfer the tutorial, run one stage, inspect its log, then check export/download before
attempting a production-sized dataset.
