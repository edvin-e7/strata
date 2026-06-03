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
- `VectorColumn`: cosine top-k **semantic search** as a native column operation.

Honest, reproducible, narrow on purpose. See [DESIGN.md](DESIGN.md) for the roadmap.
