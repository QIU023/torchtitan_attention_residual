# Architecture comments for the upstream Kimi K3 eager PR

Draft, not posted. Two points, both checked against the tech report, both
config-level rather than design-level -- which matters, because the stated goal
is that the same module code scales to the released 2.8T config. A debug model
that cannot express the released layer pattern is a problem for that goal, not
just for the debug model.

Keep the tone as questions. We are wrong often enough that "did you intend
this?" costs nothing and being wrong loudly costs a lot.

## Comment 1: the final layer is not global attention

Report sec 2.1:

> Each block contains 3 KDA layers followed by 1 Gated MLA layer, giving a 3:1
> mixing ratio. This pattern is repeated throughout the backbone. An additional
> Gated MLA layer is placed at the end of the backbone, ensuring that the final
> layer always performs global attention.

`_debugmodel` sets `full_attention_layers = {4, 8, 12}` over 13 layers, so
layer 13 is KDA and the backbone ends on linear attention.

The released shape corroborates the reading: 93 layers is 23 x 4 + 1, i.e. 23
blocks plus that one extra Gated MLA.

Draft text:

> Should `full_attention_layers` include the last layer? Sec 2.1 says an extra
> Gated MLA is placed at the end of the backbone "ensuring that the final layer
> always performs global attention", and 93 = 23*4 + 1 lines up with that. As
> written, `{4, 8, 12}` over 13 layers ends the stack on KDA.
>
> Raising it at config level because a 2.8T config would need
> `[4, 8, ..., 88, 92, 93]` with 92 and 93 both global, which the current
> "every (ratio+1)-th layer" pattern cannot express without a special case.

## Comment 2: no final aggregation over block representations

Report sec 2.2, immediately after eq. 10:

> The final output layer then aggregates all N block representations.

and:

> for Kimi K3, we partition its layers into 8 blocks with 12-layer size, giving
> a partial final block and 9 total blocks when counting the embedding layer.

`KimiK3TransformerBlock` carries `attention_res_norm/proj` and
`ffn_res_norm/proj`, which is the per-layer attention of eq. 10. There is no
module corresponding to the final aggregation.

Draft text:

> Is the final aggregation over the N block representations intended to be out
> of scope for this PR? Sec 2.2 has the per-layer attention over block
> representations (eq. 10) and then a separate final output-layer aggregation
> over all N of them; I only see the former here. Happy to contribute it if
> it is wanted.

## What we should NOT raise

Both implementations attach a residual attention to the attention and FFN
sublayers separately (two per layer), while eq. 8 and eq. 10 describe one per
layer over `f_i(h_i)`, "the output of layer i". This may be compressed phrasing
in the report rather than a deviation, and we have not settled it. Raising it
would be presenting our own unverified design as a question about theirs.

## Things we checked and that are fine

Say so if the thread gets adversarial; it makes the two above read as review
rather than as a list of complaints.

* SiTU-GLU, both branches, `beta=4.0` / `linear_beta=25.0` -- matches fig. 4.
* Gated MLA gate is full rank (`_linear(dim, num_heads * v_head_dim)`) --
  matches eq. 7's "the gate projection Wg is full rank".
* AttnRes pseudo-query as `Linear(dim, 1)` with an RMSNorm on the keys --
  matches eq. 8's `phi(q,k) = exp(q^T RMSNorm(k))`.
* 3:1 KDA:MLA ratio, block size 12.
* Loss is core torchtitan `CrossEntropyLoss`, not a bespoke one.

## Note on cross-references

Do not put the PR number in commit messages. Each one becomes a
cross-reference on their thread, and we have already generated far too many.
Reference issues only in the comment text itself, where it is deliberate.
