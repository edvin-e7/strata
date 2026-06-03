# Strata — design & honest positioning

> Private design doc + portfolio case-study seed. Posture: showable engine
> (WHAT/WHY), footwork/stack-strategy stays dark (see Cairn's doctrine). Private,
> no remote, until release-ready. — 2026-06-04

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
   On-thesis with *Own Your Intelligence*.

## Why Go (resolving the earlier "Go is wrong")

"Go is wrong" was true for *one* goal: beating everyone on raw speed. For the goal
that actually matters here — **demonstrate AI-engineering competence, ship a real
system, differentiate** — Go is a *strong* choice: it's the language of modern
data/AI infrastructure (Kubernetes, Docker, vector DBs, data tooling), it ships as
a single static binary, and it shows you build production systems, not benchmarks.
A well-engineered columnar engine in Go is a top-tier AI-eng portfolio piece.

## First honest benchmark (reproduce: `go test -bench=. -benchmem`)

| Workload | Rows | Result |
|---|---|---|
| filter (`> 500`) → sum, single core, no SIMD | 10,000,000 | **~39 ms, 1 alloc** (≈258M rows/s) |

Naive baseline, deliberately. It only goes up from here — and every claim ships
with the command that reproduces it.

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
2. Column types (float64, dict-string, bool) + a tiny typed table.
3. Group-by + aggregate, vectorized.
4. The embedding/vector column + semantic search.
5. Local-LLM NL→query layer.
6. Honest benchmark suite + write-up (the portfolio artifact).
