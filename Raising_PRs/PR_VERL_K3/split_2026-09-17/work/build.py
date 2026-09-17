import os, subprocess, sys
import splitlib as S
import assign as A

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
ORDER = S.ORDER
OIDX = S.OIDX


def apply_moves(items, moves):
    by_idx = {it.idx: it for it in items}
    order = list(items)
    for (lo, hi), where, anchor in moves:
        block = [by_idx[i] for i in range(lo, hi + 1) if i in by_idx]
        for b in block:
            order.remove(b)
        pos = order.index(by_idx[anchor])
        if where == "after":
            pos += 1
        order[pos:pos] = block
    return order


def stage_contents(path):
    items, base, head = S.build_items(path)
    spec = A.ASSIGN.get(path)
    if spec is None and path not in A.WHOLE:
        raise SystemExit(f"no assignment for {path}")
    if spec:
        covered = set()
        for rng, name in spec:
            o = OIDX[name]
            if rng == "*":
                for it in items:
                    if it.kind != "ctx":
                        it.revs[-1] = (o, it.revs[-1][1])
                        covered.add(it.idx)
            else:
                lo, hi = rng
                for i in range(lo, hi + 1):
                    it = items[i]
                    if it.kind == "ctx":
                        continue
                    it.revs[-1] = (o, it.revs[-1][1])
                    covered.add(it.idx)
        missing = [it.idx for it in items if it.kind != "ctx" and it.idx not in covered]
        if missing:
            raise SystemExit(f"{path}: unassigned items {missing[:20]} ({len(missing)} total)")
    for idx, revs in A.VARIANTS.get(path, {}).items():
        it = items[idx]
        for name, text in revs:
            it.revs.append((OIDX[name], text))
        it.revs.sort(key=lambda r: (r[0] if r[0] is not None else -1))
    order = apply_moves(items, A.MOVES.get(path, []))
    out = {}
    for k in range(len(ORDER)):
        whole = A.WHOLE.get(path)
        if whole is not None:
            content = None
            for name in ORDER[: k + 1]:
                if name in whole:
                    v = whole[name]
                    content = head if v is None else open(os.path.join(HERE, v)).read()
            out[k] = content
            continue
        parts = [it.at(k) for it in order]
        parts = [p for p in parts if p is not None]
        out[k] = "".join(parts) if parts else None
    return out, head


def main():
    paths = sorted(set(list(A.ASSIGN) + list(A.WHOLE)))
    changed = S.git("diff", "--name-only", S.BASE, S.HEAD).split()
    assert sorted(changed) == paths, (set(changed) ^ set(paths))
    per_stage = {}
    for p in paths:
        stages, head = stage_contents(p)
        if stages[len(ORDER) - 1] != head:
            open(os.path.join(HERE, "MISMATCH.txt"), "w").write(stages[len(ORDER) - 1] or "")
            raise SystemExit(f"{p}: final stage != HEAD (see MISMATCH.txt)")
        per_stage[p] = stages

    idx = os.path.join(HERE, "build.idx")
    env = dict(os.environ, GIT_INDEX_FILE=idx)

    def g(*args):
        return subprocess.run(["git", "-C", S.REPO, *args], capture_output=True,
                              check=True, env=env).stdout.decode()

    trees = []
    for k in range(-1, len(ORDER)):
        if os.path.exists(idx):
            os.remove(idx)
        g("read-tree", S.BASE)
        if k >= 0:
            for p, stages in per_stage.items():
                content = stages[k]
                if content is None:
                    g("update-index", "--force-remove", p)
                    continue
                tmp = os.path.join(HERE, "blob.tmp")
                with open(tmp, "w") as f:
                    f.write(content)
                blob = g("hash-object", "-w", "--path", p, tmp).strip()
                g("update-index", "--add", "--cacheinfo", f"100644,{blob},{p}")
        trees.append(g("write-tree").strip())
    os.remove(idx)

    names = {
        "01": "01_engine_tp_packed_and_compat.patch",
        "02": "02_engine_pipeline.patch",
        "03": "03_engine_ep_lora_qat_sync.patch",
        "04": "04_engine_context_parallel.patch",
        "05": "05_kimi_k3.patch",
        "06": "06_metrics_logprob_diff.patch",
        "00": "local_env_and_diagnostics.patch",
    }
    for k, name in enumerate(ORDER):
        d = subprocess.run(["git", "-C", S.REPO, "diff", "--no-color", "--binary",
                            trees[k], trees[k + 1]], capture_output=True, check=True).stdout
        with open(os.path.join(OUT, names[name]), "wb") as f:
            f.write(d)
        print(name, names[name], len(d.splitlines()), "lines")
    print("final tree", trees[-1], "HEAD tree", S.git("rev-parse", f"{S.HEAD}^{{tree}}").strip())


if __name__ == "__main__":
    main()
