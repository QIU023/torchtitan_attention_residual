# How wrong conclusions got produced here, by mechanism

Written after a session in which I stated seven things confidently and had to retract
six of them, several after they had already been written into other documents. This is
not a list of mistakes; it is a list of the MECHANISMS that produced them, because the
mistakes were not independent -- the same three shapes recur.

Read this before asserting anything about the parallelism stack.

---

## Mechanism 1: asserting the content of a grep that returned nothing

**Instance.** "Upstream's proper axis for expert tensor parallelism is `etp`, and we never
configured it." `etp` does not exist anywhere in this upstream version. The grep I ran
returned no output at all, and I read that absence as "our config does not set it" rather
than as "the thing does not exist".

**Instance.** "The maxdeg cells pass on the merged tree, byte-identical." `run_maxdeg.sh`
hard-coded `TITAN=` while its sibling honoured `${TITAN:-}`, so those five cells ran
against the DEV tree. They were byte-identical because they were literally the same tree.
I read the numbers without checking which tree produced them.

**The shape.** An empty or unsurprising result is evidence about the QUESTION, not only
about the answer. When a grep comes back empty, the next command is the one that proves
the term exists at all.

**The rule.** No assertion about upstream from a search that returned nothing. Prove the
symbol exists somewhere before describing what it does.

---

## Mechanism 2: using a control that does not control for the variable

**Instance (five rounds).** The GRPO merged arm was used four times as proof that a
parameter name exists in the rollout's `params_dict`. The merged arm does not enable LoRA,
so its `params_dict` is the unwrapped model. Every conclusion drawn from it about a
LoRA-enabled name was unfounded, and one of them (round 3) turned a bug in vLLM's mapping
into a "contract" the sender was changed to satisfy.

**The shape.** A control arm differs from the test arm in exactly one thing only if you
have checked WHAT it differs in. "It works there" is not transitive across a difference
you have not enumerated.

**The rule.** Before using arm B as an oracle for arm A, write down the list of things
that differ between them. If the property in question is on that list, B is not an oracle.

---

## Mechanism 3: chasing assertions instead of modelling the disagreement

**Instance.** The EP x TP failure was `S(1) -> P(sum)`. I keyed one conditional on
`enable_ep`, which moved the error one level up to the wrapper's own check. I then keyed
all three consistently, which produced an NCCL collective timeout -- a deadlock, strictly
worse than the error it replaced.

**Instance (the same shape, same day).** The adapter-only weight sync converged over five
rounds of "add the next name the KeyError asks for".

**The shape.** Each fix satisfies the assertion in front of it without asking what the
runtime is actually doing. Two or three rounds of this feel like progress because
something changes every time.

**The rule.** After the SECOND assertion in the same chain, stop and instrument. In the
vLLM case a ten-minute diagnostic that printed the real candidate names ended four rounds
of guessing. In the sharding case the deadlock was the instrument: it proved the
disagreement was physical placement, not a label.

---

## Mechanism 4: repeating a mistake that is already written down, including by me

**Instance.** `run_postmerge_gate.sh` exported `KIMI_VIT_DEP=1` for the text arm. That DEP
is INVALID on text flavors (the finding-50 guard fires: "this rank owns 1 vision stage but
0 were wired") had already cost a full text-arm launch, and the correction is recorded.
Worse, the doc containing the WRONG version -- `run_matrix_dep_dyncp.sh`'s header, "the
text arm carries the same knobs for symmetry, but they are inert there" -- is a file I
wrote, and I copied its framing instead of its correction.

**The shape.** A correction recorded in a document does not propagate to the scripts and
headers that stated the original claim. The stale statement is the one that gets copied,
because it reads as settled.

**The rule.** When a claim is corrected, grep the repo for the claim's WORDING and fix
every copy, not just the doc that records the correction. "inert" was searchable.

---

## Mechanism 5: reasoning from a constraint list nobody re-checked

**Instance, the largest one.** Three constraints justified 20 `use_local_output=True` and
the entire imperative TP plan:

| stated constraint | what is actually true |
|---|---|
| fla's triton kernels do not dispatch through DTensor | Already solved IN OUR CODE. `_to_local_if_dtensor`'s own docstring says KDA "strips DTensor at the kernel call site ... and re-DTensors at the end". KDA is DTensor-in / DTensor-out; the plan strips an EXTRA layer at the boundary. |
| PP's P2P can only send plain tensors | Upstream handles it in the framework: `pipeline_parallel.py` obtains each stage's mesh so it can "re-wrap incoming tensors as DTensors with the correct placements". Not a model concern. |
| AttnRes's `torch.stack` cannot mix plain and DTensor | The problem is MIXING. All-DTensor stacks fine. This constraint is self-inflicted by having stripped some boundaries and not others. |

So the plain-boundary design is not the forced consequence of three hard constraints. It
is a historical layer that was never cleaned up after the fla issue was fixed at the right
level. I asserted the three-constraint story twice in one session, once while explicitly
calling it "被三个硬约束逼出来的唯一解".

**The shape.** A constraint list acquires authority by being repeated. Each repetition
cites the previous one, and the original evidence is never revisited -- especially when the
list is load-bearing for a design decision already made.

**The rule.** Before citing a constraint to justify a design, open the code that the
constraint is about and confirm it still binds. Constraints have expiry dates; fixes
elsewhere retire them silently.

---

## Mechanism 6: letting a claim's SCOPE drift

**Instance.** "torchtitan now forces DTensor throughout." Technically false --
`sharding_config is None` makes the machinery a no-op, and our imperative path still runs
(11 of 13 cells pass). The accurate statement is narrower and worse: `kimi_k3` is the ONLY
file left in `torchtitan/models/` that calls `parallelize_module`, and `use_local_output`
appears once in all of upstream's models against 20 times in ours. Not "forbidden" --
"sole remaining user", which means nobody has any reason to preserve compatibility with it.

**Instance.** "`lora_skip_prefixes` is part of this fix." It is read only when loading an
adapter from a checkpoint, and has no effect on the IPC weight-sync path GRPO uses. I had
described it as causally related to five rounds of KeyErrors it cannot influence.

**The shape.** A true observation about one path gets restated as a general mechanism, and
the restatement is what gets used for the decision.

**The rule.** State the scope with the claim, in the same sentence. "On the checkpoint
load path" and "in one of four call sites" are part of the finding, not caveats to add if
challenged.

---

## Mechanism 7: proceeding on an unverified premise about my own runs

**Instance.** Rounds 3, 4 and 5 of the GRPO sync ran CONCURRENTLY on the same two GPUs,
because `timeout 3000` keeps a driver alive for 50 minutes and I launched each round right
after reading the previous round's log. Four jobs shared two GPUs. The KeyError conclusions
survive -- contention cannot invent a specific missing key -- but nothing resource-shaped
from those runs is usable, including a `DataLoader worker killed by signal: Killed` that I
had waved off as benign teardown and which has exactly the signature of memory pressure.

**The shape.** The premise "the previous run has finished" was never checked, because
reading its log felt like confirmation that it was over.

**The rule.** Before launching anything on the GPUs, `pgrep`. A log that has stopped
growing is not a process that has exited.

---

## Mechanism 5b: attributing a difference to the most interesting nearby change

**Instance.** The text arm drifted 1e-2 after the merge while both multimodal arms moved
1e-5. I attributed it to the text flavor being 20 of 21 layers MoE and therefore most
exposed to the merge's MoE sharding rework. A fact already in hand refutes it: the drifting
cells include `fsdp2`, with `ep=1` and `tp=1`, where no MoE sharding code runs at all.

The real cause was one grep away -- `3f71477c8 Mask loss at document boundaries (#4075)`
changes `hf_datasets/text_datasets.py`, and the text arm's dataset is `c4` while the
multimodal arms use `cc12m-test`. Upstream even regenerated its own golden loss files in
that commit, which is as explicit as a "this changes your losses" gets.

**The shape.** The recently-studied subsystem becomes the default explanation. I had spent
hours inside the MoE sharding code, so a MoE story was available and felt informed; the
data path was not in mind at all, and nothing prompted me to ask which commit touches the
input pipeline.

**The rule.** For an unexplained numerical difference, enumerate the commits that touch the
path FIRST -- data, loss, kernel, parallelism -- and only then reason about magnitude. And
check the cheapest discriminator: a cell with the suspected feature disabled. `fsdp2` was
sitting in the same table.

## What actually worked, so it gets repeated

* **The smoke gate.** `run_postmerge_gate.sh` runs one 2-step cell per arm and exits on
  failure. It caught a missing venv (`rc=127`) and then an OOM, each time before three
  hours of matrix. Two saves on its first day.
* **The matrix as a merge gate.** `git merge` reported no conflicts, compileall was clean,
  the model imported, the CPU suite passed 371 tests -- and 2 of 13 cells failed. No static
  check available would have found it.
* **Instrumenting instead of guessing.** One diagnostic that printed the candidate keys
  under the missing name's parent ended a four-round guessing loop in a single run.
* **Reverting a fix that is worse than the bug.** The three-conditional patch produced a
  deadlock; reverting it and recording why is a result, not a failure.
* **Checking a hypothesis I was about to write down.** "Pre-merge the label was already
  wrong, nobody validated it" was disproved in one command (`protocols/module.py` did not
  change in the merge and the validation predates it). That one would have been a
  load-bearing wrong conclusion in a document.
