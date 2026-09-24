"""Per micro-batch block transport, one block committed per stage, stage s on rank s % p.

ours:   every block rides the schedule's adjacent hop s -> s+1; a hop carries what the
        receiving rank has not cached (the PR's delta_to_send).
hybrid: jinsooihm's rule, generalised to v > 1: block b reaches the first stage after b on
        each other rank at offset d = 1..p-1; d == 1 or d even rides the adjacent hop into
        that stage, d odd >= 3 is a direct send from b's rank (non-adjacent).
"""
from collections import Counter


def ours(p, v):
    S = p * v
    held = {r: set() for r in range(p)}
    acc, hops = set(), []
    for s in range(S):
        acc.add(s)
        held[s % p] |= acc
        if s + 1 < S:
            hops.append(sorted(acc - held[(s + 1) % p]))
    return hops, []


def hybrid(p, v):
    S = p * v
    relay = {s: [] for s in range(S - 1)}      # hop s -> s+1
    direct = []                               # (src stage, dst stage, block)
    for b in range(S):
        for d in range(1, p):
            t = b + d                         # first stage after b on rank (b + d) % p
            if t >= S:
                break
            if d == 1 or d % 2 == 0:
                relay[t - 1].append(b)
            else:
                direct.append((b, t, b))
    return [sorted(relay[s]) for s in range(S - 1)], direct


def check(p, v, hops, direct):
    S = p * v
    got = {r: Counter() for r in range(p)}
    for s, blocks in enumerate(hops):
        for b in blocks:
            got[(s + 1) % p][b] += 1
    for src, dst, b in direct:
        got[dst % p][b] += 1
    for t in range(S):                        # every stage must see every earlier block
        r = t % p
        for b in range(t):
            if b % p != r and got[r][b] != 1:
                return False
    return True


def summary(p, v, scheme):
    hops, direct = scheme(p, v)
    assert check(p, v, hops, direct), (p, v, scheme.__name__)
    msgs = [len(h) for h in hops if h] + [1] * len(direct)
    return dict(peak=max(msgs), volume=sum(msgs), adjacent=sum(1 for h in hops if h), direct=len(direct))


print("| p | v | ours: peak | ours: volume | ours: adjacent sends (extra) | hybrid: peak | hybrid: volume | hybrid: adjacent sends | hybrid: direct non-adjacent sends |")
print("|---:|---:|---:|---:|---|---:|---:|---:|---:|")
for p in (2, 4, 8, 16):
    for v in (1, 2, 4, 8):
        o, h = summary(p, v, ours), summary(p, v, hybrid)
        S = p * v
        # closed forms
        assert o["peak"] == (p - 1 if p > 1 else 0), (p, v, o)
        assert o["volume"] == p * (p - 1) * (2 * v - 1) // 2, (p, v, o)
        assert o["adjacent"] == S - 1 and o["direct"] == 0
        assert h["volume"] == o["volume"]
        assert h["peak"] == 1 + (p - 1) // 2 or p == 2, (p, v, h)
        direct_closed = (S - p + 1) * ((p - 2) // 2) + sum(max(0, (j - 1) // 2) for j in range(3, p - 1))
        assert h["direct"] == direct_closed, (p, v, h, direct_closed)
        print(f"| {p} | {v} | {o['peak']} | {o['volume']} | {o['adjacent']} (0) | {h['peak']} | {h['volume']} | {h['adjacent']} | {h['direct']} |")
print("closed forms verified for p in 2..16, v in 1..8")
