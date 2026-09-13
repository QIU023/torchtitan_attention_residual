# PR 4312: the empty stack payload on a cached hop (2026-09-13)

Checked against `k3_pp_text` = `pp_review4` = `dbc425403` and the torch nightly in `venv_bfx9`.

## What happens today

- Our `AttnResPipelineStage` sends nothing itself (no `dist.` call in `kimi_k3/pipeline_stage.py`): it shapes the stage's outputs -- the hidden state and a stack payload holding only the entries the next rank has not seen -- and reassembles the stack on receipt. `PPRankLocalCache` is a rank-local dict of tensor views, no communication.
- The P2P is torch.distributed.pipelining's: the schedule asks `get_fwd_send_ops` / `get_fwd_recv_ops` for one isend / irecv per stage output, with no zero-size check.
- When the receiving rank already holds every entry, `route_payload` returns `new_zeros(T, 0, D)`: a 0-byte isend / irecv is still posted. `_batch_p2p` does not coalesce an all-send (or all-recv) batch, so it is its own call, next to the hidden state's.
- Backward: the empty payload needs no gradient, the receiver allocates no gradient buffer, `_get_recv_ops` skips it -- no backward op.
- Frequency: only hops into the rank that already holds everything; 1 of 3 hops in the 2 x 4 pp2 x vp2 example, 4 of 15 in 4 x 4 pp4 x vp4.

## Dropping the payload on those hops (option 2): not now

Cost it would save: one 0-byte forward send / recv per empty hop per micro-batch -- launch-level, no bandwidth, no memory; not measured.

Risks:
1. Both ends must agree on the stage's output count from the same routing tables; a rank that disagrees leaves an unmatched P2P and hangs the first hop with no message pointing at the cause. A fixed arity removes that failure mode.
2. pipelining's init paths (output metadata inference via `_compute_outputs`, receive-buffer creation, the runtime's P2P warm-up / inference-mode vote) must handle per-stage output counts; they are torch's and move with torch versions.
3. Our backward (`backward_one_chunk`, `_retrieve_recv_grads`, `split_stack_grad`) indexes (hidden, payload) and would need the missing-payload case.
4. Everything wrapping `forward_one_chunk` assumes two outputs: the neighbor P2P transport (`k3_pp_transport`), pp_balance, DEP (`vit_dep`); the regression set grows to pp2 x vp2, pp8 x vp4, vit_dep, transport, with bitwise and hang checks.

Decision (user, 2026-09-13): the reply states the empty payload honestly ("sN -> [] (empty payload)"); option 2 only if a profile shows the 0-byte ops matter.
