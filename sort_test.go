package strata

import (
	"math/rand"
	"reflect"
	"testing"
)

// sort_test.go — Block 8 falsification matrix. Correctness first: these fail loudly if
// order-by/top-N is wrong (wrong direction, non-deterministic ties, mutated input,
// TopN diverging from OrderBy) before any sort benchmark is trusted.

func TestOrderByAscending(t *testing.T) {
	c := NewInt64Column([]int64{3, 1, 2}) // values 1@1, 2@2, 3@0
	got := c.OrderBy(nil, false)
	if want := (Selection{1, 2, 0}); !reflect.DeepEqual(got, want) {
		t.Fatalf("OrderBy asc = %v, want %v", got, want)
	}
}

func TestOrderByDescending(t *testing.T) {
	c := NewInt64Column([]int64{3, 1, 2}) // 3@0, 2@2, 1@1
	got := c.OrderBy(nil, true)
	if want := (Selection{0, 2, 1}); !reflect.DeepEqual(got, want) {
		t.Fatalf("OrderBy desc = %v, want %v", got, want)
	}
}

// Ties (equal values) break by ascending original row index — in BOTH directions.
// This is the property that makes the order total and reproducible.
func TestOrderByTieBreakByIndex(t *testing.T) {
	c := NewInt64Column([]int64{5, 1, 5, 1}) // value 5 at rows 0,2 ; value 1 at rows 1,3
	asc := c.OrderBy(nil, false)
	if want := (Selection{1, 3, 0, 2}); !reflect.DeepEqual(asc, want) {
		t.Fatalf("OrderBy asc with ties = %v, want %v", asc, want)
	}
	desc := c.OrderBy(nil, true)
	if want := (Selection{0, 2, 1, 3}); !reflect.DeepEqual(desc, want) {
		t.Fatalf("OrderBy desc with ties = %v, want %v", desc, want)
	}
}

// filter → order-by flows the selection through: only the selected rows are ordered,
// indices still referring to the original column.
func TestOrderByWithSelection(t *testing.T) {
	c := NewInt64Column([]int64{30, 10, 40, 20})
	sel := c.FilterGT(15) // rows 0(30), 2(40), 3(20)
	got := c.OrderBy(sel, false)
	if want := (Selection{3, 0, 2}); !reflect.DeepEqual(got, want) { // 20,30,40
		t.Fatalf("OrderBy(sel) asc = %v, want %v", got, want)
	}
}

// OrderBy must not scramble the caller's selection — it sorts a copy, so a selection
// feeding another op downstream is left intact.
func TestOrderByDoesNotMutateInput(t *testing.T) {
	c := NewInt64Column([]int64{3, 1, 2})
	sel := Selection{0, 1, 2}
	before := append(Selection(nil), sel...)
	_ = c.OrderBy(sel, true)
	if !reflect.DeepEqual(sel, before) {
		t.Fatalf("OrderBy mutated caller's selection: %v, was %v", sel, before)
	}
}

func TestOrderByEmpty(t *testing.T) {
	if got := NewInt64Column(nil).OrderBy(nil, false); len(got) != 0 {
		t.Fatalf("OrderBy of empty column = %v, want empty", got)
	}
	c := NewInt64Column([]int64{1, 2, 3})
	if got := c.OrderBy(c.FilterGT(9999), true); len(got) != 0 {
		t.Fatalf("OrderBy of empty selection = %v, want empty", got)
	}
}

// The selection-vector composition claim, executable: order-by then gather the values
// in order and assert monotonicity — the sort produced a real ordering, not a shuffle.
func TestOrderByGatherIsMonotonic(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	d := make([]int64, 2000)
	for i := range d {
		d[i] = int64(r.Intn(100)) // lots of ties
	}
	c := NewInt64Column(d)
	desc := c.OrderBy(nil, true)
	for i := 1; i < len(desc); i++ {
		if c.Data[desc[i-1]] < c.Data[desc[i]] {
			t.Fatalf("desc order not non-increasing at %d: %d then %d", i, c.Data[desc[i-1]], c.Data[desc[i]])
		}
	}
	asc := c.OrderBy(nil, false)
	for i := 1; i < len(asc); i++ {
		if c.Data[asc[i-1]] > c.Data[asc[i]] {
			t.Fatalf("asc order not non-decreasing at %d: %d then %d", i, c.Data[asc[i-1]], c.Data[asc[i]])
		}
	}
}

// The keystone cross-check: TopN(k) must be byte-identical to OrderBy()[:k] for every
// k — same rows, same deterministic order — across directions, with and without a
// pre-filter, including k=0, k=1, and k > row count. If the bounded insertion and the
// full sort ever disagree, one of them is wrong; this pins them together.
func TestTopNMatchesOrderByPrefix(t *testing.T) {
	r := rand.New(rand.NewSource(5))
	d := make([]int64, 1500)
	for i := range d {
		d[i] = int64(r.Intn(60)) // heavy ties to stress the tiebreak boundary
	}
	c := NewInt64Column(d)
	sels := map[string]Selection{"all": nil, "filtered": c.FilterGT(30)}
	for _, desc := range []bool{false, true} {
		for name, sel := range sels {
			full := c.OrderBy(sel, desc)
			for _, k := range []int{0, 1, 7, 100, len(full), len(full) + 50} {
				got := c.TopN(sel, k, desc)
				want := full
				if k < len(full) {
					want = full[:k]
				}
				if k <= 0 {
					want = Selection{}
				}
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("TopN(%s, k=%d, desc=%v) = %v, want OrderBy prefix %v", name, k, desc, got, want)
				}
			}
		}
	}
}

// TopN composes the same way OrderBy does: gather the top-n values straight out of the
// column. The sum of the n largest is an honest end-to-end check of the bounded path.
func TestTopNComposesGather(t *testing.T) {
	c := NewInt64Column([]int64{10, 50, 20, 40, 30})
	top3 := c.TopN(nil, 3, true) // 50,40,30
	if got := c.SumAt(top3); got != 120 {
		t.Fatalf("sum of top-3 = %d, want 120", got)
	}
}

func TestTopNZeroOrNegative(t *testing.T) {
	c := NewInt64Column([]int64{1, 2, 3})
	for _, k := range []int{0, -1, -100} {
		if got := c.TopN(nil, k, true); len(got) != 0 {
			t.Fatalf("TopN(k=%d) = %v, want empty", k, got)
		}
	}
}

// Honest order-by benchmark: full sort of 10M random int64 by value, single core.
// Reproduce:  go test -bench=OrderBy -benchmem -run='^$'
func BenchmarkOrderBy_10M(b *testing.B) {
	c := NewInt64Column(benchSortData())
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkSel = c.OrderBy(nil, true)
	}
}

// The bounded payoff: top-10 of the same 10M rows — O(m log k), no full sort.
// Reproduce:  go test -bench=TopN -benchmem -run='^$'
func BenchmarkTopN_10M_k10(b *testing.B) {
	c := NewInt64Column(benchSortData())
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkSel = c.TopN(nil, 10, true)
	}
}

func benchSortData() []int64 {
	r := rand.New(rand.NewSource(23))
	d := make([]int64, 10_000_000)
	for i := range d {
		d[i] = int64(r.Intn(1_000_000))
	}
	return d
}
