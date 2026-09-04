"""Resolve model.py conflicts between the SP splice fix (HEAD) and the CP splice branch (theirs)."""
import pathlib, re, sys
p = pathlib.Path('torchtitan/models/kimi_k3/model.py'); s = p.read_text()
out = []; i = 0; n = 0
while True:
    a = s.find('<<<<<<< ', i)
    if a == -1:
        out.append(s[i:]); break
    out.append(s[i:a]); m = s.index('=======\n', a); b = s.index('>>>>>>> ', m); e = s.index('\n', b) + 1
    head = s[s.index('\n', a) + 1:m]; theirs = s[m + len('=======\n'):b]; n += 1
    if '_sp_group: dist.ProcessGroup | None = None' in head and 'def preprocess_inputs(' in theirs:
        out.append(head)                      # the attribute plus the annotated def
    elif '_splice_under_sequence_parallel(' in head and 'self._cp_group is not None' in theirs:
        guard_old = '''            if embeddings_TD.shape[0] != tokens.shape[0]:
                # Under spmd_types the stream is a plain local tensor, so a
                # sequence-parallel shard shows up as a row count that no
                # longer matches the token shard the mask describes.
                raise NotImplementedError(
                    "Kimi K3 context parallel with sequence parallel is not "
                    "supported: the vision splice needs the whole sequence."
                )
'''
        assert guard_old in theirs, "cp guard"
        cp = theirs.replace(guard_old, '')
        cp = cp.replace('''        if self._cp_group is not None and dist.get_world_size(self._cp_group) > 1:
''', '''        if self._cp_group is not None and dist.get_world_size(self._cp_group) > 1:
            if self._sp_group is not None:
                raise NotImplementedError(
                    "Kimi K3 context parallel with sequence parallel is not "
                    "supported: the vision splice needs the whole sequence."
                )
''', 1)
        out.append(cp.rstrip('\n') + '\n\n' + head)
    else:
        raise SystemExit(f"unexpected hunk {n}:\n{head[:300]}\n=====\n{theirs[:300]}")
    i = e
s = ''.join(out); p.write_text(s)
print(f"resolved {n} hunk(s); markers left {s.count('<<<<<<<')}; sp block {s.count('_splice_under_sequence_parallel(')}; cp guard {s.count('if self._sp_group is not None:')}")
