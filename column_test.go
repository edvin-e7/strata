package strata

import (
	"math/rand"
	"testing"
)

// Correctness first — falsification discipline: these fail loudly if the
// vectorized loops are wrong, before we ever trust a benchmark number.

func TestFilterGT(t *testing.T) {
	c := NewInt64Column([]int64{5, 1, 9, 3, 7})
	got := c.FilterGT(4) // values > 4 are at rows 0 (5), 2 (9), 4 (7)
	want := Selection{0, 2, 4}
	if len(got) != len(want) {
		t.Fatalf("FilterGT len = %d (%v), want %d", len(got), got, len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("FilterGT[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestSumAndSumAt(t *testing.T) {
	c := NewInt64Column([]int64{5, 1, 9, 3, 7})
	if got := c.Sum(); got != 25 {
		t.Fatalf("Sum = %d, want 25", got)
	}
	sel := c.FilterGT(4) // rows 0,2,4 -> 5+9+7 = 21
	if got := c.SumAt(sel); got != 21 {
		t.Fatalf("SumAt = %d, want 21", got)
	}
}

// SumWhereGT is the fused one-pass form: it must equal FilterGT→SumAt exactly, never
// approximate it. The equivalence is pinned across a random column so a mutation that
// drops the predicate (summing everything) or sums the rejected rows goes RED.
func TestSumWhereGT(t *testing.T) {
	c := NewInt64Column([]int64{5, 1, 9, 3, 7})
	if got := c.SumWhereGT(4); got != 21 { // 5 + 9 + 7
		t.Fatalf("SumWhereGT(4) = %d, want 21", got)
	}
	if got := c.SumWhereGT(100); got != 0 { // nothing passes
		t.Fatalf("SumWhereGT(100) = %d, want 0", got)
	}
	// Equivalence to the composable two-pass form over a larger random column.
	big := tenMillion()
	if fused, twoPass := big.SumWhereGT(500), big.SumAt(big.FilterGT(500)); fused != twoPass {
		t.Fatalf("SumWhereGT(%d) = %d but FilterGT+SumAt = %d — fused path diverged", 500, fused, twoPass)
	}
}

func tenMillion() *Int64Column {
	r := rand.New(rand.NewSource(42))
	data := make([]int64, 10_000_000)
	for i := range data {
		data[i] = int64(r.Intn(1000))
	}
	return NewInt64Column(data)
}

// Honest benchmark: a full filter→aggregate over 10M rows. Run with
//
//	go test -bench=. -benchmem
//
// Report the number as-is. Never dress it up as "faster than X".
func BenchmarkFilterSumGT_10M(b *testing.B) {
	c := tenMillion()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sel := c.FilterGT(500)
		_ = c.SumAt(sel)
	}
}
