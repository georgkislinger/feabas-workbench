# Use your normal Workbench workflow at LRZ

Local build **0.3.3+cluster.5**. Open your project on your PC, choose **Use cluster…**,
complete setup, save cluster settings, then use **Sync project & images**. The normal
Run buttons now submit work to LRZ. The green accents and the **Execution: LRZ cluster**
banner show which computer will do the computation. **Leave cluster mode** (top bar,
Pipeline menu or the setup window) switches back to this PC at any time.

The cluster.2 build passed a real 13-command synthetic FEABAS pipeline, including
rendering, mipmaps and VAST export. A live LRZ check (job 5610645) confirmed SSH/MFA
sign-in, Slurm submission, file round trips and download for this account; FEABAS
itself has not run at LRZ yet. This is a local modification, not an upstream release.

## First live check: sign in, choose storage, prepare FEABAS

1. Restart Workbench after updating. Confirm **0.3.3+cluster.5** in the title/About dialog.
2. Open the tutorial project. Click **Use cluster…** (also available as
   **Pipeline → Run on cluster…**).
3. In **1 Cluster setup**, enter your LRZ username and click **1. Sign in and find my storage**.
   Passwords and MFA are entered in private prompts; never paste them into chat.
   Check a new host fingerprint against [LRZ's published fingerprints](https://doku.lrz.de/access-and-login-to-the-linux-cluster-10745974.html).
4. **Storage folder** lists your assigned DSS containers and your **home folder**. Below it,
   Workbench estimates what the project will need at LRZ: about **5× the raw images**
   (the uploaded copy, stitched sections, aligned stack and VAST export with their image
   pyramids), one more copy per preprocessing variant, plus ~3 GB for the FEABAS
   environment. For the home folder it compares that with the free quota from
   `dssusrinfo` and warns before you commit to a folder that is too small.
   Click **2. Use this storage folder**. In the home folder, Workbench keeps everything in
   `~/feabas-workbench/` (one folder per project, plus the shared environment).
5. Click **3. Prepare FEABAS at LRZ** (5–15 minutes). LRZ's `python` modules stop at 3.8,
   so Workbench loads LRZ's recommended **Miniforge** module, creates a private
   Python 3.11 environment next to the project folders and installs FEABAS 3.0.5 with
   headless OpenCV (compute nodes may lack the graphics libraries of the default OpenCV).
   Jobs call that environment's Python directly; nothing needs activating.

The home folder (100 GB, backed up nightly) is enough for the tutorial and projects up
to roughly 15–20 GB of raw images. The proposed 1 TB dataset needs a DSS container: ask
the institute's LRZ project administrator for Linux Cluster DSS project storage with
enough quota for source images, intermediate data, outputs and retained previous results.
Scratch has a 14-day retention policy and no backup, so it is not offered.
See [LRZ storage instructions](https://doku.lrz.de/file-systems-and-io-on-linux-cluster-10745972.html).

The filesystem list in `dssusrinfo all` describes what the login node has mounted;
it does not grant you those directories. Workbench lists only paths from the
**You have access to the following DSS Containers** section, plus your own home folder.
A failed or unrecognized report is shown as a discovery problem, rather than proof of no
allocation. Mounted roots such as `/dss/dssfs02` and other users' homes cannot be selected.

## Connect the dataset and local result destination

**2 Data transfer** offers two ways for images, results and exports to travel:

- **Directly over SSH** (default for the home folder and for datasets below ~200 GB): no
  extra software. Files go through your existing SSH login, encrypted, and are renamed
  into place only after their size is confirmed. Unchanged files are skipped and nothing
  is deleted at either end, so an interrupted upload continues with the next **Sync**.
  Workbench must stay open while it copies; meanwhile the Sync button reads **Stop transfer**.
- **Globus** (large datasets, institute NAS): transfers run without Workbench and survive
  disconnections, but need Globus Connect Personal on this PC (or a NAS collection).
  LRZ serves DSS containers through Globus, not home folders.

For SSH: save cluster settings, click **Sync project & images** in the top bar and wait for
**Images synchronized over SSH · ready to run**. Keep **Include existing project results**
checked to resume a local project, or clear it for a fresh run.

For Globus:

1. Install and run **Globus Connect Personal** on this PC using
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
6. Leave **Automatically download after Export** enabled.
7. Save cluster settings, then click **Sync project & images** in the top bar.
   Wait until the verified uploads finish and the banner says the images are synchronized.

Globus transfers use encryption and checksum verification. They copy changed files without
deleting extra destination files. Keep the source NAS/server and the local Globus service
available. The PC can disconnect after receiving the Slurm job ID; jobs keep running.
Globus transfers can continue independently of Workbench, provided their endpoints remain online.
See [Globus transfer options](https://docs.globus.org/cli/reference/transfer/).

If Globus requests collection permissions, click **Sign in / grant access** again. The
interface remembers the required consent scopes for that session. Failed transfers can
be retried under Advanced. If a request's outcome is unknown, recover its task UUID from
Globus Activity using the exact label; Workbench does not automatically duplicate it.

## Leaving cluster mode

**Leave cluster mode** is in the top bar, in the **Pipeline** menu and at the bottom of the
setup window. The Run buttons then compute on this PC again and the colours return to blue.
Jobs already submitted to LRZ and running Globus transfers keep going; Workbench stops
monitoring them (the banner says so) until you choose **Use cluster…** and save again.
Only an operation that is talking to LRZ at that moment (sign-in, an SSH copy, a
submission) has to finish first; an SSH copy can be stopped with **Stop transfer**.
Workstation results and the cluster preview cache stay separate either way.

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
masks are uploaded with each request. Large masks/checkpoints travel with the chosen
transfer method before computation is submitted. Changing scientific settings archives
affected remote outputs so FEABAS does not silently reuse stale results. These archives
consume storage space.

Pipeline state, progress, logs and memory estimates return to the GUI. Small previews
arrive over SSH. Larger preview/model caches follow over SSH or through Globus; wait for
that transfer before editing or selecting those files. Local edits made during a job are
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

The next shared check is deliberately small: sign in, choose the home folder, prepare the
environment, sync the tutorial over SSH, run one stage, inspect its log, then check
export/download before attempting a production-sized dataset.
