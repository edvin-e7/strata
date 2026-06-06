# Strata — design & honest positioning

> Design doc + portfolio case-study seed. Posture: showable engine (WHAT/WHY),
> with the broader stack strategy kept as its own concern. — 2026-06-04

## What it is

A from-scratch, vectorized, columnar in-memory data engine in Go — built
**AI-native** and **local-first** from day one.

## What it is NOT (the honesty that is itself the senior signal)

It does **not** claim to beat Polars or DuckDB on raw speed. Those engines (Rust /
C++, hand-tuned SIMD, Arrow, a decade of query-optimizer work) own that war, and
they own it for hard physical reasons — Go has a GC and weaker SIMD ergonomics. A
data engineer who claims "my Go dataframe is faster than Polars" signals they don't
know the field. We claim the opposite kind of thing, and we benchmark honestly.

## The claim we *can* win

1. **Genuinely well-engineered columnar core** — real vectorized execution
   (contiguous columns, selection vectors, no row-by-row copies), honestly
   benchmarked. Competence is in the engineering, not a number we can't hit.
2. **AI-native, not AI-bolted-on** — embeddings/vectors are a first-class column
   type; querying via a **local** LLM (Ollama / gemma-e7) is native; semantic
   search over a column sits next to `GROUP BY`. The fast engines don't have this.
3. **Local-first / owned** — your data, your machine, your engine. Nothing leaves.
   No cloud dependency, no data exfiltration by design.

## Why Go (resolving the earlier "Go is wrong")

"Go is wrong" was true for *one* goal: beating everyone on raw speed. For the goal
that actually matters here — **demonstrate AI-engineering competence, ship a real
system, differentiate** — Go is a *strong* choice: it's the language of modern
data/AI infrastructure (Kubernetes, Docker, vector DBs, data tooling), it ships as
a single static binary, and it shows you build production systems, not benchmarks.
A well-engineered columnar engine in Go is a top-tier AI-eng portfolio piece.

## Honest benchmarks → full write-up in [BENCHMARKS.md](BENCHMARKS.md)

The first benchmark falsified the obvious assumption: the textbook two-pass columnar
path (`FilterGT`→`SumAt`, ~38 ms, 40 MB alloc) **lost** to a naive row-store scan
(~37 ms, 0 alloc) — it materializes a full-column selection vector and walks the data
twice. Fusing filter+sum into one pass (`SumWhereGT`) is **~6× faster (6.3 ms, 0
alloc)**, and the disassembler shows why: the compiler predicates the contiguous-slice
reduction into branchless `CSEL`, while the strided struct access keeps a
mispredicting branch. Measured, not assumed — including that it's *not* SIMD and *not*
bandwidth at this scale. Every number ships with the command that reproduces it.

## MVP scope (achievable in weeks, not a DuckDB-clone in years)

- **Columns:** int64, float64, dictionary-encoded string, bool, and the
  differentiator — **vector/embedding column**.
- **Vectorized operators:** filter, project, group-by + aggregate, sort, hash-join
  — all over selection vectors, no intermediate copies.
- **Honest bench harness:** vs a naive row store and vs pandas; positioned
  *honestly* next to Polars/DuckDB ("same ballpark on X, not on Y").
- **AI-native layer:** NL → query plan via local LLM; semantic search over the
  embedding column.
- **Test backbone:** spec-first + falsification matrix (correctness before any
  benchmark is trusted).

## Non-goals

- No cost-based query optimizer (that's DuckDB's decade).
- No "faster than Polars" claim, ever.
- No distributed mode. Single-machine, local-first, on purpose.

## Roadmap (blocks)

1. ✅ Vectorized columnar core seed: Int64 column, filter→aggregate, honest 10M bench.
2. ✅ Column types (float64, dict-string, bool) + a tiny typed table (`typed.go`):
   float64 (NaN-correct filter), bit-packed bool (popcount count + set-bit iteration),
   dictionary-encoded string (group/filter on int codes, decode only at the edges), and
   a `TypedTable` that stores them behind a typed schema and validates column type as
   well as existence before any op. Keystone cross-type op: `GroupSumFloat`
   (revenue-by-region) over the selection vector. The NL planner still drives the
   int64 `Table`; migrating it onto `TypedTable` is folded into block 6's write-up.
3. ✅ Group-by + aggregate, vectorized: hash-aggregation over the selection vector
   (filter→group-by→sum, no intermediate), deterministic key-sorted output, honest
   10M-row bench. Driven through the same validate-everything NL→query path.
4. ✅ The embedding/vector column + semantic search (cosine top-k, native column op).
5. ✅ Local-LLM NL→query layer (Ollama/gemma-e7, model proposes / engine disposes).
6. ✅ Honest benchmark suite + write-up ([BENCHMARKS.md](BENCHMARKS.md)) — the portfolio
   artifact. Columnar-vs-row-store study (the two-pass path lost; fused `SumWhereGT`
   wins ~6×), branch-vs-bandwidth experiment, disassembly evidence (`CSEL` vs `BLE`),
   and the full operator-cost table — reporting the rows where the number went the
   wrong way, which is the senior signal.
7. ✅ Vectorized hash-join (`join.go`): the last MVP operator blocks 1–6 left unbuilt.
   Inner equi-join on int64 keys, two-phase build/probe (O(L+R), not O(L·R)). The
   result is `JoinResult{Left, Right}` — two parallel selection vectors, never a
   materialized joined-row table, so it composes straight into the existing ops
   (`amount.SumAt(jr.Left)`). filter→join flows selections through both sides;
   output order is deterministic (probe-then-build index order — the map is never
   iterated for output, so map randomization can't leak in, no post-sort). Table-level
   `Join` inherits the validate-everything contract (unknown key → clean error, never
   a crash). Honest 10M-fact ⋈ 1000-dim bench (~76 ms; the cost is dominated by
   materializing the two ~10M output index vectors, not the probe). Build side is the
   right input by design — choosing it from cardinality is cost-based planning, a
   non-goal.
8. ✅ Vectorized order-by + top-N (`sort.go`): the final MVP operator. A sort returns a
   PERMUTATION — a `Selection` of row indices that, gathered, walks the column in order
   — so it never moves or copies the data and composes (filter→order-by→`SumAt`).
   `OrderBy` is a full `sort.Slice` with a total, deterministic comparator (value, then
   ascending row index, so ties are reproducible in both directions despite an unstable
   sort); `TopN` is the bounded O(m log k) form (the int64 analog of `vector.go`'s
   `insertTopK`) and is pinned by test to equal `OrderBy()[:k]` exactly. Honest bench:
   full `OrderBy` of 10M rows ~2.2 s / 40 MB (the permutation vector), but `TopN` k=10
   over the same 10M is ~21 ms / 48 B — ~100× faster and effectively zero-alloc, which
   is why TopN is the path for "top 10 by X". **With this the MVP vectorized-operator
   set (filter, project, group-by+aggregate, sort, hash-join) is complete.** Follow-on
   (not in this block, like join): wiring ORDER BY / LIMIT into the NL→query planner.
