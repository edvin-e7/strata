package strata

import (
	"math/rand"
	"reflect"
	"testing"
)

// join_test.go — Block 7 falsification matrix. Correctness first: these fail loudly
// if the hash-join is wrong (missed match, wrong pair, non-deterministic order, a
// validation hole an LLM could drive through) before any join benchmark is trusted.

// One-to-one: each left key matches exactly one right key. Result pairs come out in
// probe order (left ascending), each carrying the single matching right index.
func TestHashJoinOneToOne(t *testing.T) {
	left := NewInt64Column([]int64{1, 2, 3})
	right := NewInt64Column([]int64{3, 2, 1}) // build: {3:[0],2:[1],1:[2]}
	got := HashJoinInt64(left, nil, right, nil)
	want := JoinResult{Left: Selection{0, 1, 2}, Right: Selection{2, 1, 0}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("HashJoin one-to-one = %+v, want %+v", got, want)
	}
}

// One-to-many: a left row whose key hits several build rows emits one pair per build
// row, in build (ascending-index) order.
func TestHashJoinOneToMany(t *testing.T) {
	left := NewInt64Column([]int64{1, 2})
	right := NewInt64Column([]int64{1, 1, 2}) // build: {1:[0,1],2:[2]}
	got := HashJoinInt64(left, nil, right, nil)
	want := JoinResult{Left: Selection{0, 0, 1}, Right: Selection{0, 1, 2}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("HashJoin one-to-many = %+v, want %+v", got, want)
	}
}

// Many-to-many on a shared key is the per-key cartesian product — inherent to inner
// join semantics. Two left rows × two right rows on key 1 = four pairs. The honest
// caveat in HashJoinInt64 is exactly this case; the test pins that we produce it
// correctly rather than dropping or duplicating pairs.
func TestHashJoinManyToMany(t *testing.T) {
	left := NewInt64Column([]int64{1, 1})
	right := NewInt64Column([]int64{1, 1})
	got := HashJoinInt64(left, nil, right, nil)
	want := JoinResult{Left: Selection{0, 0, 1, 1}, Right: Selection{0, 1, 0, 1}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("HashJoin many-to-many = %+v, want %+v", got, want)
	}
}

// No shared key → zero pairs, never a silent cross join.
func TestHashJoinNoMatch(t *testing.T) {
	got := HashJoinInt64(NewInt64Column([]int64{1, 2}), nil, NewInt64Column([]int64{3, 4}), nil)
	if got.Len() != 0 {
		t.Fatalf("HashJoin disjoint keys = %+v, want no pairs", got)
	}
}

// Empty input on either side → zero pairs, no panic.
func TestHashJoinEmptyInput(t *testing.T) {
	if g := HashJoinInt64(NewInt64Column(nil), nil, NewInt64Column([]int64{1}), nil); g.Len() != 0 {
		t.Fatalf("empty left = %+v, want no pairs", g)
	}
	if g := HashJoinInt64(NewInt64Column([]int64{1}), nil, NewInt64Column(nil), nil); g.Len() != 0 {
		t.Fatalf("empty right = %+v, want no pairs", g)
	}
}

// Selection vectors must flow straight into the join — a filter→join chain restricted
// to the selected rows, no intermediate materialization, indices still referring to
// the original columns.
func TestHashJoinWithSelection(t *testing.T) {
	leftKey := NewInt64Column([]int64{1, 2, 1, 2})
	leftVal := NewInt64Column([]int64{5, 50, 5, 50})
	sel := leftVal.FilterGT(10)               // rows 1,3 (both key 2)
	right := NewInt64Column([]int64{2, 9, 2}) // build: {2:[0,2],9:[1]}
	got := HashJoinInt64(leftKey, sel, right, nil)
	// li=1 key2 → (1,0),(1,2) ; li=3 key2 → (3,0),(3,2)
	want := JoinResult{Left: Selection{1, 1, 3, 3}, Right: Selection{0, 2, 0, 2}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("HashJoin(leftSel) = %+v, want %+v", got, want)
	}

	// And a pre-filter on the build (right) side restricts which build rows exist.
	rsel := right.FilterGT(5) // right row 1 only (key 9)
	got = HashJoinInt64(leftKey, nil, right, rsel)
	if got.Len() != 0 { // no left key equals 9
		t.Fatalf("HashJoin(rightSel keeping only key 9) = %+v, want no pairs", got)
	}
}

// The join result is selection-vector currency, so it composes with every existing op
// with zero glue: aggregate the matched rows straight out of the original columns via
// SumAt. This is the claim "no joined-row table is ever built" made executable.
func TestHashJoinComposesWithSumAt(t *testing.T) {
	orderCustomer := NewInt64Column([]int64{1, 2, 1})
	orderAmount := NewInt64Column([]int64{100, 200, 300})
	custID := NewInt64Column([]int64{1, 2})
	custRegion := NewInt64Column([]int64{10, 20})

	jr := HashJoinInt64(orderCustomer, nil, custID, nil)
	// pairs: (0,0)(1,1)(2,0) → all three orders match a customer
	if got := orderAmount.SumAt(jr.Left); got != 600 {
		t.Fatalf("matched order amount = %d, want 600", got)
	}
	if got := custRegion.SumAt(jr.Right); got != 40 { // 10 + 20 + 10
		t.Fatalf("matched customer region sum = %d, want 40", got)
	}
}

// Determinism pin: the join must be byte-identical across runs (Go randomizes map
// iteration; we never iterate the map for output, so order is fixed by probe-then-build
// index order). Equal-across-runs and the order invariant — Left non-decreasing, and
// strictly increasing Right within an equal Left run — are correctness properties.
func TestHashJoinDeterministicOrder(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	ld := make([]int64, 3000)
	rd := make([]int64, 3000)
	for i := range ld {
		ld[i] = int64(r.Intn(50)) // low cardinality → many-to-many, exposes map randomization
		rd[i] = int64(r.Intn(50))
	}
	left, right := NewInt64Column(ld), NewInt64Column(rd)
	a := HashJoinInt64(left, nil, right, nil)
	b := HashJoinInt64(left, nil, right, nil)
	if !reflect.DeepEqual(a, b) {
		t.Fatal("HashJoin order not stable across runs — output is leaking map randomization")
	}
	for i := 1; i < a.Len(); i++ {
		if a.Left[i-1] > a.Left[i] {
			t.Fatalf("Left not non-decreasing at %d: %d then %d", i, a.Left[i-1], a.Left[i])
		}
		if a.Left[i-1] == a.Left[i] && a.Right[i-1] >= a.Right[i] {
			t.Fatalf("Right not increasing within equal Left at %d: %d then %d", i, a.Right[i-1], a.Right[i])
		}
	}
}

// The table-level Join inherits the validate-everything contract: an unknown key on
// either side is a clean error, never a crash and never a silently-wrong join. This is
// the safety that lets an LLM name the join keys.
func TestJoinTableValidatesColumns(t *testing.T) {
	orders := NewTable().AddInt64("customer", NewInt64Column([]int64{1, 2, 1}))
	customers := NewTable().AddInt64("id", NewInt64Column([]int64{1, 2}))

	if _, err := Join(orders, customers, "nope", "id", nil, nil); err == nil {
		t.Fatal("Join with unknown left key: expected error, got nil")
	}
	if _, err := Join(orders, customers, "customer", "nope", nil, nil); err == nil {
		t.Fatal("Join with unknown right key: expected error, got nil")
	}

	jr, err := Join(orders, customers, "customer", "id", nil, nil)
	if err != nil {
		t.Fatalf("valid Join: %v", err)
	}
	want := JoinResult{Left: Selection{0, 1, 2}, Right: Selection{0, 1, 0}}
	if !reflect.DeepEqual(jr, want) {
		t.Fatalf("table Join = %+v, want %+v", jr, want)
	}
	// Gather matched amounts through the table's Column accessor.
	orders.AddInt64("amount", NewInt64Column([]int64{100, 200, 300}))
	if got := orders.Column("amount").SumAt(jr.Left); got != 600 {
		t.Fatalf("joined order amount = %d, want 600", got)
	}
}

// Honest hash-join benchmark: a star-join — a 10M-row fact table joined to a
// 1000-row dimension on an int64 key, single core, no SIMD. Each fact row matches one
// dimension row, so the result is ~10M pairs (two selection vectors); the bench
// therefore measures the build + probe + the unavoidable output materialization.
// Reproduce:  go test -bench=HashJoin -benchmem -run='^$'
func BenchmarkHashJoin_10M_fact_1000_dim(b *testing.B) {
	r := rand.New(rand.NewSource(17))
	fact := make([]int64, 10_000_000)
	for i := range fact {
		fact[i] = int64(r.Intn(1000))
	}
	dim := make([]int64, 1000)
	for i := range dim {
		dim[i] = int64(i) // unique keys 0..999
	}
	left, right := NewInt64Column(fact), NewInt64Column(dim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkInt = HashJoinInt64(left, nil, right, nil).Len()
	}
}
