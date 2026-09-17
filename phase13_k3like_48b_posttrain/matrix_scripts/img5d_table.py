"""The image 5D matrix's results as the logbook's table.

    python img5d_table.py /workspace/.img5d_0917_A /workspace/.img5d_0917_B

Per cell: the parallelism shape, the step count, the rollout-vs-actor log-prob diff and
the mean prompt length (111 is the expanded media block, so a cell that silently dropped
its images reads shorter). The exit code is not reported: the cell script ends with an
echo, so it always exits 0. A cell relaunched by hand has a log but no results row and is
picked up from its log.
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


def from_log(cell: str) -> tuple[str, str, str]:
    """Steps, log-prob diff and prompt length read from a cell's own log."""
    log = pathlib.Path(f"/workspace/img5d-{cell}.log")
    if not log.is_file():
        return "n/a", "n/a", "n/a"
    text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="ignore"))
    steps = len(re.findall(r"step:\d+ - ", text)) // 2
    diffs = re.findall(r"rollout_logprobs_diff_mean:([0-9.]+)", text)
    return str(steps), (f"{float(diffs[-1]):.5f}" if diffs else "n/a"), prompt_length(cell)


def main() -> None:
    rows = []
    covered = set()
    for d in sys.argv[1:]:
        results = pathlib.Path(d) / "results.txt"
        if not results.is_file():
            continue
        for line in results.read_text().splitlines():
            m = ROW.match(line)
            if not m or line.startswith("DONE"):
                continue
            cell, shape, _rc, steps, diff = m.groups()
            shape = shape.replace("FSDP_SIZE=", "fsdp ").replace("TP_SIZE=", "tp ")
            shape = shape.replace("CP_SIZE=", "cp ").replace("PP_SIZE=", "pp ").replace("EP_SIZE=", "ep ")
            covered.add(cell)
            rows.append((cell, shape.strip(), str(int(steps) // 2 if steps.isdigit() else steps),
                         f"{float(diff):.5f}" if diff else "n/a", prompt_length(cell)))
    # Cells relaunched by hand write a log but no results row; take them from the log.
    for log in sorted(pathlib.Path("/workspace").glob("img5d-*.log")):
        cell = log.name[len("img5d-"):-len(".log")]
        if cell in covered:
            continue
        steps, diff, plen = from_log(cell)
        rows.append((cell, "(relaunched by hand)", steps, diff, plen))
    # A cell rerun by hand carries a "_fix" suffix and supersedes the row it replaces,
    # whose shape was wrong: report it under the original name and drop the void row.
    fixed = {c[: -len("_fix")] for c, *_ in rows if c.endswith("_fix")}
    shapes = {c: sh for c, sh, *_ in rows}
    merged = []
    for cell, shape, steps, diff, plen in rows:
        if cell in fixed:
            continue
        if cell.endswith("_fix"):
            base = cell[: -len("_fix")]
            merged.append((base, shapes.get(base, shape), steps, diff, plen))
            continue
        merged.append((cell, shape, steps, diff, plen))
    print("| cell | shape | steps | log-prob diff mean | prompt length |")
    print("| --- | --- | ---: | ---: | ---: |")
    for cell, shape, steps, diff, plen in sorted(merged):
        print(f"| {cell} | {shape} | {steps} | {diff} | {plen} |")


if __name__ == "__main__":
    main()
