# strata — CLAUDE.md

A from-scratch, vectorized, **columnar** in-memory data engine in Go — built
**AI-native** and **local-first**. Portfolio/case-study piece, not a Polars/DuckDB
clone. Read `README.md`, `DESIGN.md`, and `BENCHMARKS.md` first — they carry the
honest positioning and the reproducible numbers.

## Posture (Cairn doctrine — read before any public mention)
Private/portfolio. The engine + **honest** benchmarks are the showable part
(WHAT/WHY); the broader stack strategy stays dark. **Sell the vision, not the
footwork.** Has an `origin` remote but treat as no-push by default — do NOT push
unless Edvin explicitly says so.

## Stack
- **Go 1.25.5**, module `strata`. Single flat package `strata` at repo root.
  Zero third-party deps (stdlib only — `go.mod` has no `require` block).
- **NL→query** via a LOCAL LLM: gemma-e7 over Ollama (`http://localhost:11434`,
  `/api/chat`, `temperature:0`). No API key, nothing leaves the machine, $0.

## Golden path (verified working on this Mac, 2026-06-06)
```bash
go build ./...                       # exit 0
go vet ./...                         # exit 0
go test ./...                        # exit 0 — full correctness suite
go test -bench=. -benchmem -run '^$' # honest numbers (see BENCHMARKS.md)
```
The two live NL→query tests (`TestOllamaPlannerLive`, `TestOllamaPlannerLiveGrouped`)
hit gemma-e7 for real when Ollama is up (~10s combined) and **skip cleanly** when
it's down (`http.Get(/api/tags)` probe) — so CI stays green either way. To exercise
them, ensure `curl -sf http://localhost:11434/api/tags` succeeds and
`ollama list` shows `gemma-e7`.

## Architecture / key files
- `column.go` int64 column (filter→aggregate, fused `SumWhereGT`); `vector.go`
  embedding column + cosine top-k semantic search; `typed.go` float64 / bit-packed
  bool / dict-string columns + `TypedTable`.
- `table.go` int64 `Table` + `Query`/`Execute` (validate-everything contract);
  `join.go` vectorized hash-join (selection-vector result, never materialized);
  `sort.go` order-by + bounded `TopN`.
- `planner.go` `OllamaPlanner` (model proposes a `Query`, engine disposes) +
  `parseQueryJSON`/`extractJSONObject` (brace-scanner that survives chatty model
  prose). `bench_test.go` the honest benchmark harness.

## Conventions / gotchas
- **Selection vectors everywhere.** Operators return index selections that compose
  (filter→join→`SumAt`); no intermediate row tables are materialized. Keep new ops
  in this style.
- **Model proposes, engine disposes.** The planner NEVER executes. Everything runs
  through `Table.Execute`, which validates column existence/type and op — a
  hallucinated column or bad op errors, never runs. Don't bypass this.
- The planner is liberal on input (`normalizeOp` maps `$gt`/`gt`/"greater than"→`>`);
  the executor stays strict. Small-model variance is expected — the live grouped test
  `Skipf`s rather than fails when the model doesn't emit a grouped plan.
- **Honest benchmarks only.** Never claim "faster than Polars/DuckDB" (explicit
  non-goal in `DESIGN.md`). Ship every number with the command that reproduces it,
  including the ones that went the wrong way.
- Non-goals (don't build): cost-based optimizer, distributed mode, SIMD.
- New code: spec-first, falsification matrix (correctness before any benchmark is
  trusted). Match the existing test style.

## Done-level / goal
"Klar" = builds, full `go test ./...` green, and the golden path works end-to-end on
this machine today. The MVP vectorized-operator set (filter, project,
group-by+aggregate, sort, hash-join) is **complete** (DESIGN.md roadmap blocks 1–8
all ✅). Follow-on per DESIGN.md: wire ORDER BY / LIMIT into the NL→query planner.

## $0 / local-model & data rules
- LLM is local-only (gemma-e7 via Ollama). No paid APIs — no Anthropic/Gemini/Google
  billing, no network beyond localhost. Keep it that way.
- Code identifiers/comments/commit messages in English; the only user-facing surface
  is the engine, so sv-SE doesn't apply here.
- No PII in this repo (synthetic/generated data only); don't introduce real data.
