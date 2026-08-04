# Short comment for the eager Kimi K3 PR — draft, not posted

One comment, deliberately small. The substance lives on RFC #3029; this is a
pointer plus the two architecture questions. Do not restate the matrix here.

---

Thanks for this -- having the model folder exist upstream unblocks a lot.

We have been building the parallelism side against the full K3 architecture
since release, and have posted the current state on RFC #3029 as early
evidence: TP, EP, CP and PP all running across a 13-configuration matrix on a
multimodal debug model, eager 13/13 at 3, 10 and 100 steps. Branches are on our
fork and will be rebased onto this folder once it lands; nothing to review yet.

Two architecture questions while reading through:

**1.** Should `full_attention_layers` include the last layer? Sec 2.1 says an
additional Gated MLA is placed at the end of the backbone "ensuring that the
final layer always performs global attention", and the released 93 = 23*4 + 1
lines up with that. As written, `{4, 8, 12}` over 13 layers ends the stack on
KDA. Raising it at config level rather than as a debug-model nit, because a
2.8T config would need `[4, 8, ..., 88, 92, 93]` with 92 and 93 both global,
which the current every-(ratio+1)-th-layer pattern cannot express without a
special case.

**2.** Is the final aggregation over block representations intended to be out
of scope here? Sec 2.2 has the per-layer attention over block representations
(eq. 10) and then a separate final output-layer aggregation over all N of them;
I see the former but not the latter. Happy to contribute it if wanted.

For what it is worth, the rest lines up with the report where we checked:
SiTU-GLU with beta 4 / 25, the full-rank Gated MLA gate from eq. 7, the
AttnRes pseudo-query as a scalar projection with RMSNorm on the keys, the 3:1
ratio and block size 12.
