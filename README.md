# strata

A from-scratch, vectorized, **columnar** in-memory data engine in Go — built
**AI-native** and **local-first**.

> Posture: private/portfolio, no remote, until release-ready. The engine + honest
> benchmarks are the showable part (WHAT/WHY); the broader stack strategy stays
> dark. See [DESIGN.md](DESIGN.md).

## The honest pitch

Not "faster than Polars" — that war is over and Rust/C++ won it. strata is the
engine the fast ones aren't: **specialized, local-first, AI-native.** Semantic
search lives *inside* the engine, next to filter and aggregate, over data you own,
at zero GPU cost. Small players win by going narrow and deep, not broad.

## What works today (reproducible)

```bash
go test ./...                      # correctness
go test -bench=. -benchmem         # honest numbers
```

- Vectorized columnar `Int64Column`: filter → aggregate, **10M rows in ~39 ms**,
  1 alloc, single core, no SIMD.
- **Group-by + aggregate**: hash-aggregation flowing the selection vector straight
  through (filter→group-by→sum, no intermediate), deterministic key-sorted output —
  **10M rows / 1000 groups in ~65 ms**.
- **Hash-join**: vectorized inner equi-join (build/probe, O(L+R)) whose result is a
  pair of selection vectors — index pairs that compose straight back into the other
  ops, so no joined-row table is ever materialized — deterministic order, filter→join
  flows through. **10M-row fact ⋈ 1000-row dimension in ~76 ms**.
- **Order-by + top-N**: a sort returns a *permutation* selection (never moves the data),
  deterministic ties; the bounded `TopN` does top-10 of **10M rows in ~21 ms / 48 B**,
  ~100× the full sort, and is pinned by test to equal `OrderBy()[:k]`.
- `VectorColumn`: cosine top-k **semantic search** as a native column operation.
- **Local NL→query**: a local LLM (Ollama/gemma-e7) proposes a validated query plan;
  the engine validates and runs it — a hallucinated column or op errors, never runs.

Honest, reproducible, narrow on purpose. See [DESIGN.md](DESIGN.md) for the roadmap.
