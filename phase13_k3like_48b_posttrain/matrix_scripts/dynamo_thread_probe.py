"""Does `torch.compile(fullgraph=True)` work off the main thread?

The veRL colocated worker runs its forward on Ray's async actor thread, and the
context-parallel `allgather_kv` backend fails there with "found no compiled frames"
while dynamo reports itself enabled and supported. This compares the same compile
on the main thread, a plain worker thread and an asyncio event-loop thread, in one
process and with no GPU, so the thread is the only variable.

    python dynamo_thread_probe.py
"""

from __future__ import annotations

import asyncio
import threading

import torch


def payload(x: torch.Tensor) -> torch.Tensor:
    return x * 2 + 1


def try_compile(tag: str) -> str:
    torch._dynamo.reset()
    compiled = torch.compile(payload, fullgraph=True)
    try:
        out = compiled(torch.ones(4))
    except Exception as exc:  # noqa: BLE001
        return f"{tag:22s} FAIL {type(exc).__name__}: {str(exc).splitlines()[0][:110]}"
    return f"{tag:22s} OK   {out.tolist()} (main thread: {threading.current_thread() is threading.main_thread()})"


def main() -> None:
    print(try_compile("main thread"), flush=True)

    out: list[str] = []
    worker = threading.Thread(target=lambda: out.append(try_compile("worker thread")))
    worker.start()
    worker.join()
    print(out[0], flush=True)

    async def inside_loop() -> str:
        return try_compile("asyncio thread")

    loop_out: list[str] = []

    def run_loop() -> None:
        loop = asyncio.new_event_loop()
        try:
            loop_out.append(loop.run_until_complete(inside_loop()))
        finally:
            loop.close()

    looper = threading.Thread(target=run_loop)
    looper.start()
    looper.join()
    print(loop_out[0], flush=True)


if __name__ == "__main__":
    main()
