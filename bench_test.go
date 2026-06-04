package strata

import (
	"math/rand"
	"testing"
)

// bench_test.go — Block 6: the honest benchmark suite.
//
// The point is NOT a big number to brag about. It is to measure the one thing strata
// can honestly claim, find where the naive design actually LOSES, and show the
// boundary of every claim — on THIS machine, reproducibly. We never benchmark against
// Polars/DuckDB and never claim to beat them. See BENCHMARKS.md for the written-up
// findings (including the one where the columnar two-pass lost to a row store).
//
// Reproduce:  go test -bench=. -benchmem -run='^$'
//
// The filter→sum comparison reuses the same data the canonical columnar bench
// (BenchmarkFilterSumGT_10M, column.go's two-pass FilterGT→SumAt) runs on, so the
// columnar / row-store / fused numbers are directly comparable. Sinks stop the
// compiler from optimizing the measured work away.
var (
	sinkInt64   int64
	sinkFloat64 float64
	sinkInt     int
	sinkSel     Selection
)

const benchRows = 10_000_000

// benchInt64Data matches tenMillion() in column_test.go (seed 42, Intn(1000)) so the
// row-store and fused benchmarks measure the SAME data as BenchmarkFilterSumGT_10M.
func benchInt64Data() []int64 {
	r := rand.New(rand.NewSource(42))
	data := make([]int64, benchRows)
	for i := range data {
		data[i] = int64(r.Intn(1000))
	}
	return data
}

// ---------------------------------------------------------------------------
// The honest centerpiece: columnar vs a naive row store, same filter→sum work.
//
// Read these alongside BenchmarkFilterSumGT_10M (the two-pass columnar path). The
// finding (BENCHMARKS.md): the two-pass columnar path does NOT beat a fused row-store
// scan — it allocates a 40 MB selection vector and walks the data twice. The fix is
// SumWhereGT, which fuses filter+sum in one allocation-free pass.
// ---------------------------------------------------------------------------

// wideRow is a realistic record: 8 int64 fields (64 bytes), of which this query reads
// one. Real tables are wide and a query touches a few columns — the case columnar is
// supposed to win. (It turns out, at this scale, the layout is not the bottleneck —
// the branch is. See the selectivity benchmarks below.)
type wideRow struct {
	value                      int64
	f1, f2, f3, f4, f5, f6, f7 int64 // columns this query never reads
}

// narrowRow is the honesty check: a 2-field record (16 bytes). With little unused
// data to skip, any columnar layout advantage should nearly vanish — and it does.
type narrowRow struct {
	value int64
	other int64
}

func makeWideRows(data []int64) []wideRow {
	rows := make([]wideRow, len(data))
	for i, v := range data {
		rows[i].value = v
	}
	return rows
}

func makeNarrowRows(data []int64) []narrowRow {
	rows := make([]narrowRow, len(data))
	for i, v := range data {
		rows[i].value = v
	}
	return rows
}

// BenchmarkColumnarFused_FilterSum_10M: the fused columnar path (SumWhereGT) — one
// pass, no selection vector, zero allocation. This is the fair columnar number to put
// next to the row store; the two-pass FilterGT→SumAt (BenchmarkFilterSumGT_10M) is the
// composable-but-heavier form.
func BenchmarkColumnarFused_FilterSum_10M(b *testing.B) {
	c := NewInt64Column(benchInt64Data())
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkInt64 = c.SumWhereGT(500)
	}
}

// BenchmarkRowStoreWide_FilterSum_10M: identical query, naive row store, 64-byte
// records. The loop reads only .value, but the CPU streams whole 64-byte records.
func BenchmarkRowStoreWide_FilterSum_10M(b *testing.B) {
	rows := makeWideRows(benchInt64Data())
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var total int64
		for j := range rows {
			if rows[j].value > 500 {
				total += rows[j].value
			}
		}
		sinkInt64 = total
	}
}

// BenchmarkRowStoreNarrow_FilterSum_10M: the boundary of the claim — 16-byte records,
// little unused data to skip. If wide ≈ narrow, the scan is not bandwidth-bound here.
func BenchmarkRowStoreNarrow_FilterSum_10M(b *testing.B) {
	rows := makeNarrowRows(benchInt64Data())
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var total int64
		for j := range rows {
			if rows[j].value > 500 {
				total += rows[j].value
			}
		}
		sinkInt64 = total
	}
}

// BenchmarkRowStoreWide_Selective_10M: the other half of the branch experiment. Same
// row store, threshold 990 (~1% pass → predictable branch). If this collapses toward
// the fused columnar time while the 50%-selectivity row store stayed at ~37 ms, the
// row store's cost was branch misprediction — and the columnar fused loop was winning
// because the compiler predicated ITS branch away (it is selectivity-insensitive).
func BenchmarkRowStoreWide_Selective_10M(b *testing.B) {
	rows := makeWideRows(benchInt64Data())
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var total int64
		for j := range rows {
			if rows[j].value > 990 {
				total += rows[j].value
			}
		}
		sinkInt64 = total
	}
}

// BenchmarkColumnarFused_Selective_10M tests the branch-misprediction hypothesis:
// threshold 990 lets only ~1% of rows through, so the `> threshold` branch is highly
// predictable. If this is dramatically faster than the 50%-selectivity SumWhereGT(500)
// above, the scan is branch-bound, not memory-bound — which is why real engines reach
// for SIMD / branchless filters (a non-goal here; strata's scan is a naive scalar loop).
func BenchmarkColumnarFused_Selective_10M(b *testing.B) {
	c := NewInt64Column(benchInt64Data())
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkInt64 = c.SumWhereGT(990) // ~1% pass -> predictable branch
	}
}

// ---------------------------------------------------------------------------
// The new typed-column ops (block 2), each on its own terms — no comparison claim,
// just the honest cost of the op as built. (GroupSum / TopKCosine already have
// benchmarks in group_test.go / vector_test.go.)
// ---------------------------------------------------------------------------

// BenchmarkFloat64_FilterSum_10M: the float column two-pass path (its own summation
// cost, distinct from int64).
func BenchmarkFloat64_FilterSum_10M(b *testing.B) {
	r := rand.New(rand.NewSource(42))
	data := make([]float64, benchRows)
	for i := range data {
		data[i] = float64(r.Intn(1000))
	}
	c := NewFloat64Column(data)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sel := c.FilterGT(500)
		sinkFloat64 = c.SumAt(sel)
	}
}

// BenchmarkBoolCountTrue_10M: the bit-packed bool popcount sweep — 10M bits is ~1.25
// MB, so the whole column fits in cache and the count is near bandwidth-bound.
func BenchmarkBoolCountTrue_10M(b *testing.B) {
	r := rand.New(rand.NewSource(9))
	data := make([]bool, benchRows)
	for i := range data {
		data[i] = r.Intn(2) == 0
	}
	c := NewBoolColumn(data)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkInt = c.CountTrue()
	}
}

// BenchmarkDictFilterEq_10M: dictionary-encoded equality — the scan compares uint32
// codes, not strings (the encoding payoff), over ~100 categories.
func BenchmarkDictFilterEq_10M(b *testing.B) {
	r := rand.New(rand.NewSource(11))
	cats := make([]string, 100)
	for i := range cats {
		cats[i] = "category-" + string(rune('A'+i%26)) + string(rune('0'+i/26))
	}
	data := make([]string, benchRows)
	for i := range data {
		data[i] = cats[r.Intn(len(cats))]
	}
	c := NewDictColumn(data)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkSel = c.FilterEq(cats[0])
	}
}
