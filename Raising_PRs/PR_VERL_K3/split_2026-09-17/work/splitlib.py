"""Line-item model for splitting a two-tree diff into ordered patches."""
import subprocess, os, sys

REPO = "/tmp/wt_verl_new"
BASE = "1a8a0f5f"
HEAD = "409d059a"
# order in which the patches are applied; index in this list is the "ordinal"
ORDER = ["01", "02", "03", "04", "05", "06", "00"]
OIDX = {name: i for i, name in enumerate(ORDER)}


def git(*args, binary=False):
    out = subprocess.run(["git", "-C", REPO, *args], capture_output=True, check=True)
    return out.stdout if binary else out.stdout.decode()


def show(rev, path):
    try:
        return git("show", f"{rev}:{path}")
    except subprocess.CalledProcessError:
        return None


class Item:
    __slots__ = ("kind", "revs", "idx")

    def __init__(self, kind, revs):
        self.kind = kind
        self.revs = revs  # list of (ordinal, text|None), ordinal -1 = initial
        self.idx = None

    def at(self, k):
        val = None
        for o, t in self.revs:
            if o <= k:
                val = t
        return val


def build_items(path):
    """Items for one file: every base line plus every added line, in diff order."""
    base = show(BASE, path)
    head = show(HEAD, path)
    base_lines = base.splitlines(keepends=True) if base is not None else []
    diff = git("diff", "-U3", BASE, HEAD, "--", path)
    lines = diff.splitlines(keepends=True)
    i = 0
    while i < len(lines) and not lines[i].startswith("@@"):
        i += 1
    items = []
    base_pos = 0  # number of base lines already consumed
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("@@"):
            head_part = ln.split("@@")[1]
            old = head_part.strip().split()[0]  # -start,count
            start = int(old[1:].split(",")[0])
            # emit untouched base lines before this hunk
            while base_pos < start - 1:
                items.append(Item("ctx", [(-1, base_lines[base_pos])]))
                base_pos += 1
            i += 1
            continue
        if ln.startswith("\\"):
            i += 1
            continue
        if ln[0] == " ":
            items.append(Item("ctx", [(-1, base_lines[base_pos])]))
            base_pos += 1
        elif ln[0] == "-":
            items.append(Item("del", [(-1, base_lines[base_pos]), (None, None)]))
            base_pos += 1
        elif ln[0] == "+":
            items.append(Item("add", [(-1, None), (None, ln[1:])]))
        else:
            raise SystemExit(f"bad diff line {ln!r}")
        i += 1
    while base_pos < len(base_lines):
        items.append(Item("ctx", [(-1, base_lines[base_pos])]))
        base_pos += 1
    for n, it in enumerate(items):
        it.idx = n
    return items, base, head


def changed(items):
    return [it for it in items if it.kind != "ctx"]


def listing(path, out=sys.stdout):
    items, _, _ = build_items(path)
    for it in items:
        if it.kind == "ctx":
            continue
        mark = "-" if it.kind == "del" else "+"
        txt = it.revs[0][1] if it.kind == "del" else it.revs[1][1]
        out.write(f"{it.idx:5d} {mark} {txt.rstrip()}\n")
