# Strata — honest benchmarks

> The portfolio artifact (roadmap block 6). Posture: this is a *showable* document —
> WHAT was measured and WHY, in full, including where the naive design lost. No claim
> here beats Polars or DuckDB; we never make that claim (see [DESIGN.md](DESIGN.md)).
> The value on display is the engineering judgment and the honesty, not a trophy
> number. — 2026-06-04

**Machine:** Apple M1 Max · `darwin/arm64` · Go 1.25.5 · single goroutine (the loops
are single-threaded; the `-10` suffix is `GOMAXPROCS`, not parallelism).

**Reproduce everything:**

```sh
go test -bench=. -benchmem -run='^$'        # the whole suite
go test -bench='FilterSum_10M|Selective' -benchmem -count=3 -run='^$'   # the headline study
```

Every number below came out of that command on this machine. Run it yourself; if your
numbers differ, yours are the real ones.

---

## The headline finding: the naive columnar path *lost*, and fixing it was the lesson

The first thing the benchmark did was falsify the obvious assumption. The composable
columnar path — `FilterGT` produces a selection vector, `SumAt` consumes it — is the
textbook "vectorized columnar" shape. It **lost to a naive row store**:

| Variant (10M int64 rows, filter `> 500` → sum, ~50% pass) | Time | Alloc |
|---|---:|---:|
| Two-pass columnar — `FilterGT`→`SumAt` | **38.2 ms** | 40 MB, 1 alloc |
| Naive row store, wide records (64 B) | 36.7 ms | 0 |
| Naive row store, narrow records (16 B) | 36.6 ms | 0 |
| **Fused columnar — `SumWhereGT`** | **6.3 ms** | **0** |

Two things jump out, and both are the point of doing this honestly:

1. **The two-pass columnar path is the slowest *and* allocates 40 MB.** It builds a
   full-column selection vector and walks the data twice. When the only output is the
   aggregate, that intermediate is pure overhead. The naive row store, fusing
   filter+sum in one allocation-free pass, beat it.

2. **The fix — fusing the filter and the sum into one pass (`SumWhereGT`) — is ~6×
   faster than everything else, with zero allocation.** Not because of a clever data
   structure; because it stopped doing unnecessary work and let the compiler do its
   job (next section).

The composable two-pass form still earns its place — a selection vector can feed
several downstream operators, which an aggregate-fused op cannot. But "vectorized
columnar" is not automatically fast, and pretending it is would be the junior move.

---

## Why fused is 6× faster — measured, not asserted

The speedup is too large for "one pass instead of two" (that buys ~2×). So I ran the
mechanism down with a falsifiable experiment and the disassembler.

### Experiment: vary selectivity, watch what moves

A `> threshold` filter at ~50% selectivity is the worst case for a branch predictor
(it's a coin flip per row). Making it predictable (threshold 990 → ~1% pass) isolates
the branch-misprediction cost:

| Variant | ~50% pass (unpredictable) | ~1% pass (predictable) | Δ |
|---|---:|---:|---:|
| Row store (wide) | 36.7 ms | **10.6 ms** | **3.5×** |
| Fused columnar (`SumWhereGT`) | 6.3 ms | 6.4 ms | **none** |

The row store gets **3.5× faster** just by making its branch predictable — so ~26 ms
of its 37 ms was branch misprediction. The fused columnar loop **doesn't move at all**
with selectivity → it has no data-dependent branch to mispredict.

Wide records (64 B) and narrow records (16 B) ran identically (36.7 vs 36.6 ms), which
rules out memory bandwidth as the bottleneck at this scale. It's the branch, not the
bytes.

### Disassembly: the compiler predicated one loop and not the other

`go tool objdump` confirms exactly why the fused columnar loop has no branch to
mispredict, and the row store does:

**Fused columnar** (`SumWhereGT` over a contiguous `[]int64`):

```asm
MOVD  (R2)(R1<<3), R4    ; load c.Data[i]  — contiguous, ×8-byte stride
ADD   R4, R3, R5         ; total + v  (computed unconditionally)
CSEL  GT, R5, R3, R3     ; total = (v > t) ? total+v : total  ← BRANCHLESS select
```

**Row store** (`rows[j].value` over a `[]struct`):

```asm
LSL   $6, R4, R6         ; j << 6  — ×64-byte stride (the record width)
MOVD  (R2)(R6), R6       ; load rows[j].value
CMP   $500, R6
BLE   -6(PC)             ; ← REAL data-dependent BRANCH
ADD   R6, R5, R5         ; total += value
```

The Go compiler turns the clean contiguous-slice reduction into a **branchless
conditional select (`CSEL`)** — so a 50/50 predicate costs the same as a 99/1 one. The
strided struct-field access defeats that optimization and keeps a real branch (`BLE`),
which mispredicts ~50% of the time. That asymmetry *is* the columnar win here.

**The honest framing of the win:** it is **not** SIMD (the `CSEL` loop is scalar — I
checked; no vector registers) and it is **not** cache locality at this scale (wide ≈
narrow). It is that a **contiguous, typed column is something the compiler can compile
to branchless code, and a row of mixed fields is not.** That is a real, specific,
defensible reason columnar layout helps — and it's the one the measurements actually
support, rather than the cache-line story I'd have *assumed*.

Even with its branch made predictable, the row store (10.6 ms) stays ~1.6× behind the
fused column (6.4 ms): the residual is the 64-byte strided load vs the packed 8-byte
one. So both effects are real; the branchless codegen is simply the larger one.

---

## The rest of the operator set

Each measured on its own terms — no comparison claim, just the honest cost of the op
as built (a naive scalar loop; **no SIMD, no multi-threading, no compression** — all
explicit non-goals).

| Operation | Shape | Time | Alloc |
|---|---|---:|---:|
| `SumWhereGT` (fused filter→sum) | 10M int64 | 6.3 ms (≈1.6B rows/s) | 0 |
| `GroupSum` (hash group-by + sum) | 10M rows, ~1000 keys | 64.8 ms | 90 KB, 24 |
| `HashJoinInt64` (inner equi-join) | 10M fact ⋈ 1000 dim | 75.8 ms | 80 MB, ~1k |
| `FilterGT`→`SumAt` (float64, two-pass) | 10M float64 | 50.8 ms | 40 MB, 1 |
| `CountTrue` (bit-packed popcount) | 10M bits (~1.25 MB) | 69 µs | 0 |
| `FilterEq` (dictionary-encoded string) | 10M rows, ~100 cats | 7.2 ms | 2 MB, 26 |
| `TopKCosine` (semantic search, k=10) | 100k × 128-dim | 11.4 ms | 80 B, 1 |

Notes, honestly:

- **`CountTrue` at 69 µs** is the bit-packing payoff: 10M booleans live in ~1.25 MB, so
  the whole column fits in cache and the count is a `popcount` sweep — three orders of
  magnitude under the row-wise scans, because the data is 64× smaller and there's no
  per-row work.
- **`GroupSum` (64.8 ms) is the slowest op**, dominated by Go map operations on ~1000
  keys (hashing + probing per row). A real engine radix-partitions or uses open
  addressing; strata uses the standard library map. Honest, and a clear next target.
- **`HashJoinInt64` (75.8 ms, 80 MB)** is a star-join: a 10M-row fact table joined to a
  1000-row dimension, each fact row matching one dimension row. The honest read of that
  80 MB is that it is **not** the hash table (the 1000-row build side is tiny) — it is
  the *output*: two ~10M-element `uint32` selection vectors (≈40 MB each), the matched
  index pairs. That is inherent to a join that returns 10M pairs, not engine waste; a
  consumer that only aggregates the result (`SumAt(jr.Left)`) could fuse probe+aggregate
  to avoid materializing it, the same fusion lesson as `SumWhereGT`. The build/probe
  itself is the cheap part. Same `map`-based caveat as `GroupSum` for the build phase.
- **`TopKCosine` (11.4 ms for 100k×128)** is the AI-native differentiator — semantic
  search as a native column op, no GPU, no external vector DB — and it's a single
  contiguous pass with a bounded top-k, 80 bytes allocated total.
- **The two-pass float path still allocates 40 MB** for the same reason the int64 one
  does. The same fused-op fix applies; it just isn't built yet (scope).

---

## What this does and does not claim

- ✅ A genuinely well-engineered, vectorized columnar core, **benchmarked honestly**,
  including the case where the first design lost and *why*, measured down to the
  instruction.
- ✅ A real, specific reason columnar layout helps on this workload (branchless
  codegen over contiguous typed memory), distinguished from the reasons it *didn't*
  (not bandwidth, not SIMD, at this scale).
- ✅ AI-native columns (embedding/vector search) and bit-packed booleans the row store
  doesn't have.
- ❌ **No** claim to beat Polars or DuckDB — they own raw-speed for hard reasons
  (Rust/C++, hand-tuned SIMD, Arrow, a decade of query optimization). This is a naive
  scalar engine by design.
- ❌ No cost-based optimizer, no distributed mode, no SIMD. Single-machine,
  local-first, on purpose.

The senior signal is the second column of every table above — including the rows where
the number went the wrong way.
