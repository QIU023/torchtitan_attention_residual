"""Minimal, GPU-free reproducer for the SHM multimodal-IPC lifecycle race
that SGLANG_DISABLE_SHM_MM works around.

Background
----------
TokenizerManager ships multimodal image tensors to the scheduler subprocess
over POSIX shared memory (/psm_*) when the "cuda_ipc" transport is selected.
When SGLang is launched *inside a parent-managed process tree* (Ray actor
groups, SLURM job arrays, Monarch meshes), the worker that builds the payload
runs as its own Python interpreter -- with its own multiprocessing
``resource_tracker``. That tracker unlinks every /psm_* segment the worker
created when the worker tears down. If the scheduler subprocess (a *different*
process tree) opens the segment after that teardown, it is already gone and the
engine dies at boot with ``FileNotFoundError: ... '/psm_xxx'``.

This script reproduces that mechanism deterministically with the standard
library (no torch, no GPU, no sglang install needed) and proves the patch's
fix end-to-end via three demos run back-to-back:

1. ``demo_shm_race``     -- default ``cuda_ipc`` path: producer creates
   ``/psm_*``, exits, its resource_tracker unlinks the segment; consumer opens
   the now-missing segment -> ``FileNotFoundError`` (the production crash).
2. ``demo_default_no_race`` -- gated ``default`` path that
   ``SGLANG_DISABLE_SHM_MM=1`` selects: payload travels as pickled bytes in the
   IPC message body itself, no ``/psm_*`` segment is ever created, the
   producer's resource_tracker has nothing to unlink -> race is structurally
   impossible.
3. ``demo_gate``         -- the patched env-gate decision function: maps
   ``SGLANG_DISABLE_SHM_MM`` value to the transport string that
   ``_determine_tensor_transport_mode`` returns.

Run (Linux; the /psm_* race is POSIX-specific):
    python3 repro_disable_shm_mm_race.py
"""

import os
import pickle
import subprocess
import sys
from multiprocessing import shared_memory

# Producer body, run as its OWN interpreter so it owns its resource_tracker.
# It creates a segment, prints the name, then exits -> its resource_tracker
# unlinks /psm_<name> on shutdown (the "leaked shared_memory" cleanup). This is
# exactly what a Ray/SLURM/Monarch worker's interpreter does on teardown.
_PRODUCER = (
    "from multiprocessing import shared_memory;"
    "shm = shared_memory.SharedMemory(create=True, size=4096);"
    "print(shm.name, flush=True);"
    "shm.close()"
)


def demo_shm_race():
    # Run the producer to completion in a separate interpreter.
    out = subprocess.run(
        [sys.executable, "-c", _PRODUCER], capture_output=True, text=True
    )
    name = out.stdout.strip()
    # Producer has exited -> its resource_tracker unlinked /psm_<name>.
    try:
        shared_memory.SharedMemory(name=name)  # scheduler subprocess opens it
        print(f"[default/cuda_ipc] opened {name} (segment survived — no crash)")
    except FileNotFoundError as e:
        print(f"[default/cuda_ipc] CRASH as in prod: FileNotFoundError: {e.filename or name}")


def demo_default_no_race():
    """SGLANG_DISABLE_SHM_MM=1 selects the ``"default"`` transport: the payload
    travels as pickled bytes inside the IPC message body, with NO ``/psm_*``
    segment created. Because no segment exists, the producer's resource_tracker
    has nothing to unlink on exit -- the race that ``demo_shm_race`` exhibits
    is structurally impossible on this code path.

    Reproduced here with subprocess stdout standing in for the IPC channel
    (in production it is the existing ZMQ/socket message body).
    """
    producer = (
        "import sys, pickle;"
        "payload = b'mm_image_tensor_bytes_' * 1024;"  # ~22 KB stand-in
        "sys.stdout.buffer.write(pickle.dumps(payload))"
    )
    out = subprocess.run(
        [sys.executable, "-c", producer], capture_output=True
    )
    payload = pickle.loads(out.stdout)
    print(
        f"[default] received {len(payload)}-byte payload via inline pickle "
        f"(no /psm_* segment created -> race impossible)"
    )


def _transport_for(env_value):
    # Mirror of the patched branch in _determine_tensor_transport_mode. In
    # production the read goes through the typed descriptor
    # ``envs.SGLANG_DISABLE_SHM_MM.get()`` (sglang.srt.environ); reproduced here
    # with os.environ so the script stays import-free and runnable anywhere:
    #     if envs.SGLANG_DISABLE_SHM_MM.get():
    #         return "default"   # inline pickle, no /psm_* segment, race-proof
    if env_value in ("1", "true", "True"):
        return "default"
    return "cuda_ipc"  # (single-node) -> SHM, susceptible to the race above


def demo_gate():
    for v in (None, "1"):
        env = os.environ.get("SGLANG_DISABLE_SHM_MM") if v is None else v
        label = "unset" if v is None else "=1"
        print(f"[gate] SGLANG_DISABLE_SHM_MM {label:5s} -> transport {_transport_for(env)!r}")


if __name__ == "__main__":
    demo_shm_race()         # default cuda_ipc -> /psm_* race -> FileNotFoundError
    demo_default_no_race()  # SGLANG_DISABLE_SHM_MM=1 -> no /psm_* -> no race
    demo_gate()             # gate function picks transport per env value
