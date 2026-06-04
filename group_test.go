package strata

import (
	"context"
	"math/rand"
	"net/http"
	"reflect"
	"testing"
	"time"
)

// Correctness first — falsification discipline: these fail loudly if the grouping
// loops are wrong, before any group-by benchmark number is trusted.

func TestGroupSum(t *testing.T) {
	keys := NewInt64Column([]int64{1, 2, 1, 2, 1})
	vals := NewInt64Column([]int64{10, 20, 30, 40, 50})
	// group 1: rows 0,2,4 -> 10+30+50=90 ; group 2: rows 1,3 -> 20+40=60
	got := GroupSum(keys, vals, nil)
	want := []GroupResult{{1, 90}, {2, 60}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("GroupSum = %+v, want %+v", got, want)
	}
}

func TestGroupCount(t *testing.T) {
	keys := NewInt64Column([]int64{1, 2, 1, 2, 1})
	got := GroupCount(keys, nil)
	want := []GroupResult{{1, 3}, {2, 2}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("GroupCount = %+v, want %+v", got, want)
	}
}

// The selection vector must flow straight into the group-by — a filter→group-by→sum
// chain restricted to the selected rows, no intermediate materialization.
func TestGroupSumWithSelection(t *testing.T) {
	keys := NewInt64Column([]int64{1, 2, 1, 2, 1})
	vals := NewInt64Column([]int64{10, 20, 30, 40, 50})
	sel := vals.FilterGT(25) // rows 2(30),3(40),4(50)
	// keys at those rows: 1,2,1 -> group 1: 30+50=80 ; group 2: 40
	got := GroupSum(keys, vals, sel)
	want := []GroupResult{{1, 80}, {2, 40}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("GroupSum(sel) = %+v, want %+v", got, want)
	}
}

// A filter that matches nothing yields a non-nil empty selection (not "all rows"):
// the result must be zero groups, never a silent group-over-everything.
func TestGroupSumEmptySelection(t *testing.T) {
	keys := NewInt64Column([]int64{1, 2, 1})
	vals := NewInt64Column([]int64{10, 20, 30})
	got := GroupSum(keys, vals, vals.FilterGT(9999))
	if len(got) != 0 {
		t.Fatalf("GroupSum(empty sel) = %+v, want no groups", got)
	}
}

// Locks the determinism pin: Go randomizes map iteration order, so without the sort
// two runs over a many-key table would (almost surely) differ. Equal-across-runs and
// strictly key-ascending is a correctness property, not cosmetics.
func TestGroupDeterministicOrder(t *testing.T) {
	r := rand.New(rand.NewSource(11))
	n := 5000
	kd := make([]int64, n)
	vd := make([]int64, n)
	for i := range kd {
		kd[i] = int64(r.Intn(200)) // 200 distinct keys, plenty to expose map randomization
		vd[i] = int64(r.Intn(1000))
	}
	keys, vals := NewInt64Column(kd), NewInt64Column(vd)
	a := GroupSum(keys, vals, nil)
	b := GroupSum(keys, vals, nil)
	if !reflect.DeepEqual(a, b) {
		t.Fatal("GroupSum order not stable across runs — the sort pin is missing")
	}
	for i := 1; i < len(a); i++ {
		if a[i-1].Key >= a[i].Key {
			t.Fatalf("groups not strictly key-ascending at %d: %d then %d", i, a[i-1].Key, a[i].Key)
		}
	}
}

// Mismatched key/value lengths is a caller error and must panic at the op, mirroring
// the table's ragged-column guard — never an out-of-range crash mid-loop.
func TestGroupSumPanicsOnRaggedColumns(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on mismatched key/value column lengths")
		}
	}()
	GroupSum(NewInt64Column([]int64{1, 2, 3}), NewInt64Column([]int64{1, 2}), nil)
}

func TestExecuteGroup(t *testing.T) {
	tb := sampleTable() // amount=[100,200,300,400,500] flag=[1,0,1,0,1]
	// sum amount by flag: flag1 rows 0,2,4 -> 900 ; flag0 rows 1,3 -> 600
	got, err := tb.ExecuteGroup(Query{Agg: "sum", Column: "amount", GroupBy: "flag"})
	if err != nil {
		t.Fatalf("ExecuteGroup sum: %v", err)
	}
	if want := []GroupResult{{0, 600}, {1, 900}}; !reflect.DeepEqual(got, want) {
		t.Fatalf("sum amount by flag = %+v, want %+v", got, want)
	}
	// count by flag: flag0 -> 2, flag1 -> 3
	got, err = tb.ExecuteGroup(Query{Agg: "count", GroupBy: "flag"})
	if err != nil {
		t.Fatalf("ExecuteGroup count: %v", err)
	}
	if want := []GroupResult{{0, 2}, {1, 3}}; !reflect.DeepEqual(got, want) {
		t.Fatalf("count by flag = %+v, want %+v", got, want)
	}
}

func TestExecuteGroupWithFilter(t *testing.T) {
	tb := sampleTable()
	// sum amount by flag where amount>250: rows 2(300),3(400),4(500); flags 1,0,1
	// flag0 -> 400 ; flag1 -> 800
	got, err := tb.ExecuteGroup(Query{Agg: "sum", Column: "amount", GroupBy: "flag", Filter: &Predicate{"amount", ">", 250}})
	if err != nil {
		t.Fatalf("ExecuteGroup filtered: %v", err)
	}
	if want := []GroupResult{{0, 400}, {1, 800}}; !reflect.DeepEqual(got, want) {
		t.Fatalf("sum amount by flag where amount>250 = %+v, want %+v", got, want)
	}
}

// The grouped path inherits the same validate-everything safety as Execute, and the
// two entry points reject each other's shape instead of silently doing the wrong thing.
func TestExecuteGroupRejectsHallucination(t *testing.T) {
	tb := sampleTable()
	badGroup := []Query{
		{Agg: "sum", Column: "amount", GroupBy: "nope"},                     // unknown group_by column
		{Agg: "sum", Column: "ssn", GroupBy: "flag"},                        // unknown sum column
		{Agg: "median", GroupBy: "flag"},                                    // unsupported agg
		{Agg: "count", GroupBy: "flag", Filter: &Predicate{"flag", "<", 1}}, // unsupported op
		{Agg: "count"}, // no group_by -> wrong entry point
	}
	for i, q := range badGroup {
		if _, err := tb.ExecuteGroup(q); err == nil {
			t.Fatalf("ExecuteGroup case %d %+v: expected error, got nil", i, q)
		}
	}
	// And a grouped query must not silently collapse through the scalar Execute.
	if _, err := tb.Execute(Query{Agg: "count", GroupBy: "flag"}); err == nil {
		t.Fatal("Execute with group_by set: expected error directing to ExecuteGroup, got nil")
	}
}

// Live, guarded: ask gemma-e7 for a grouped breakdown and assert it matches the
// deterministic ground truth. Skips cleanly when Ollama is down, so CI stays stable.
func TestOllamaPlannerLiveGrouped(t *testing.T) {
	if _, err := http.Get("http://localhost:11434/api/tags"); err != nil {
		t.Skip("Ollama not reachable — skipping live grouped NL→query test")
	}
	tb := sampleTable()
	p := NewOllamaPlanner()
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	q, err := p.Plan(ctx, "what is the total amount for each flag?", tb.Schema())
	if err != nil {
		t.Fatalf("planner: %v", err)
	}
	if q.GroupBy == "" {
		t.Skipf("model did not produce a grouped plan (%+v) — small-model variance, not an engine bug", q)
	}
	got, err := tb.ExecuteGroup(q)
	if err != nil {
		t.Fatalf("execute planned grouped query %+v: %v", q, err)
	}
	if want := []GroupResult{{0, 600}, {1, 900}}; !reflect.DeepEqual(got, want) {
		t.Fatalf("NL→grouped query = %+v, want %+v (planned: %+v)", got, want, q)
	}
	t.Logf("live NL→grouped OK: %+v → %+v", q, got)
}

// Honest group-by benchmark: sum a value column grouped by a ~1000-key column over
// 10M rows, single core, no SIMD. Reproduce: go test -bench=GroupSum -benchmem
func BenchmarkGroupSum_10M_1000keys(b *testing.B) {
	r := rand.New(rand.NewSource(13))
	n := 10_000_000
	kd := make([]int64, n)
	vd := make([]int64, n)
	for i := range kd {
		kd[i] = int64(r.Intn(1000))
		vd[i] = int64(r.Intn(1000))
	}
	keys, vals := NewInt64Column(kd), NewInt64Column(vd)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = GroupSum(keys, vals, nil)
	}
}
