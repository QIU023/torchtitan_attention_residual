"""The image 5D matrix's results as the logbook's table.

    python img5d_table.py /workspace/.img5d_0917_A /workspace/.img5d_0917_B

Per cell: the parallelism shape, the exit code, the step count, the rollout-vs-actor
log-prob diff, and the mean prompt length (111 is the expanded media block, so a cell
that silently dropped its images reads shorter).
"""

import pathlib
import re
import sys

ROW = re.compile(r"^(\S+)\s+(.*?)\s+rc=(\S+)\s+steps=(\S+)\s+(?:rollout_logprobs_diff_mean:([0-9.]+))?")


def prompt_length(cell: str) -> str:
    log = pathlib.Path(f"/workspace/img5d-{cell}.log")
    if not log.is_file():
        return "n/a"
    text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="ignore"))
    seen = sorted(set(re.findall(r"prompt_length/mean:([0-9.]+)", text)))
    return seen[0] if len(seen) == 1 else (", ".join(seen) if seen else "n/a")


def main() -> None:
    print("| cell | shape | rc | steps | log-prob diff mean | prompt length |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for d in sys.argv[1:]:
        results = pathlib.Path(d) / "results.txt"
        if not results.is_file():
            continue
        for line in results.read_text().splitlines():
            m = ROW.match(line)
            if not m or line.startswith("DONE"):
                continue
            cell, shape, rc, steps, diff = m.groups()
            shape = shape.replace("FSDP_SIZE=", "fsdp ").replace("TP_SIZE=", "tp ")
            shape = shape.replace("CP_SIZE=", "cp ").replace("PP_SIZE=", "pp ").replace("EP_SIZE=", "ep ")
            d_txt = f"{float(diff):.5f}" if diff else "n/a"
            print(f"| {cell} | {shape} | {rc} | {int(steps) // 2 if steps.isdigit() else steps} | {d_txt} | {prompt_length(cell)} |")


if __name__ == "__main__":
    main()
