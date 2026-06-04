package strata

import (
	"math"
	"testing"
)

// Falsification-first: every test here fails loudly on a wrong loop before any
// benchmark or claim is trusted. Each one is meant to go RED under a plausible
// mutation of the code it covers (off-by-one tail mask, missing NaN skip, code/string
// confusion, non-deterministic group order).

// ---- Float64Column ----

func TestFloat64FilterSumAt(t *testing.T) {
	c := NewFloat64Column([]float64{5.5, 1.0, 9.25, 3.0, 7.5})
	if got := c.Sum(); got != 26.25 {
		t.Fatalf("Sum = %v, want 26.25", got)
	}
	sel := c.FilterGT(4) // rows 0 (5.5), 2 (9.25), 4 (7.5)
	want := Selection{0, 2, 4}
	if len(sel) != len(want) {
		t.Fatalf("FilterGT = %v, want %v", sel, want)
	}
	for i := range want {
		if sel[i] != want[i] {
			t.Fatalf("FilterGT[%d] = %d, want %d", i, sel[i], want[i])
		}
	}
	if got := c.SumAt(sel); got != 22.25 { // 5.5 + 9.25 + 7.5
		t.Fatalf("SumAt = %v, want 22.25", got)
	}
}

// NaN must never pass a > filter: every comparison against NaN is false, so a NaN row
// is "not greater than" anything. A mutation that special-cased NaN the wrong way (or
// used >= on a sentinel) would surface here.
func TestFloat64FilterExcludesNaN(t *testing.T) {
	c := NewFloat64Column([]float64{1, math.NaN(), 2, math.Inf(-1), 3})
	sel := c.FilterGT(0)
	want := Selection{0, 2, 4} // 1, 2, 3 — NaN and -Inf excluded
	if len(sel) != len(want) {
		t.Fatalf("FilterGT over NaN/Inf = %v, want %v", sel, want)
	}
	for i := range want {
		if sel[i] != want[i] {
			t.Fatalf("FilterGT[%d] = %d, want %d", i, sel[i], want[i])
		}
	}
}

// ---- BoolColumn (bit-packed) ----

// The classic bitset bug is the ragged tail: a length that is not a multiple of 64.
// 65 rows spans two words with one live bit in the second; 130 spans three. If
// counting or set-bit iteration mishandled the partial word this goes RED.
func TestBoolColumnRaggedTail(t *testing.T) {
	data := make([]bool, 65)
	data[0] = true
	data[64] = true // first bit of the SECOND word
	c := NewBoolColumn(data)

	if c.Len() != 65 {
		t.Fatalf("Len = %d, want 65", c.Len())
	}
	if !c.At(0) || !c.At(64) {
		t.Fatalf("At(0)=%v At(64)=%v, want both true", c.At(0), c.At(64))
	}
	if c.At(1) || c.At(63) {
		t.Fatalf("At(1)=%v At(63)=%v, want both false", c.At(1), c.At(63))
	}
	if got := c.CountTrue(); got != 2 {
		t.Fatalf("CountTrue = %d, want 2", got)
	}
	sel := c.FilterTrue()
	if len(sel) != 2 || sel[0] != 0 || sel[1] != 64 {
		t.Fatalf("FilterTrue = %v, want [0 64]", sel)
	}
}

func TestBoolColumnDense(t *testing.T) {
	data := make([]bool, 130)
	for i := range data {
		data[i] = i%2 == 0 // 65 trues at even indices 0..128
	}
	c := NewBoolColumn(data)
	if got := c.CountTrue(); got != 65 {
		t.Fatalf("CountTrue = %d, want 65", got)
	}
	sel := c.FilterTrue()
	if len(sel) != 65 {
		t.Fatalf("FilterTrue len = %d, want 65", len(sel))
	}
	for k, row := range sel {
		if int(row) != k*2 {
			t.Fatalf("FilterTrue[%d] = %d, want %d (must be ascending evens)", k, row, k*2)
		}
	}
}

func TestBoolColumnEmptyAndAllFalse(t *testing.T) {
	if c := NewBoolColumn(nil); c.CountTrue() != 0 || len(c.FilterTrue()) != 0 || c.Len() != 0 {
		t.Fatalf("empty bool column not empty: count=%d sel=%v len=%d", c.CountTrue(), c.FilterTrue(), c.Len())
	}
	c := NewBoolColumn(make([]bool, 100)) // all false
	if c.CountTrue() != 0 || len(c.FilterTrue()) != 0 {
		t.Fatalf("all-false column: count=%d sel=%v, want 0 and empty", c.CountTrue(), c.FilterTrue())
	}
}

// ---- DictColumn ----

func TestDictColumnEncoding(t *testing.T) {
	vals := []string{"north", "south", "north", "east", "south", "north"}
	c := NewDictColumn(vals)
	if c.Len() != 6 {
		t.Fatalf("Len = %d, want 6", c.Len())
	}
	if c.Cardinality() != 3 { // north, south, east
		t.Fatalf("Cardinality = %d, want 3", c.Cardinality())
	}
	for i, want := range vals {
		if got := c.At(i); got != want {
			t.Fatalf("At(%d) = %q, want %q", i, got, want)
		}
	}
}

// FilterEq must distinguish "matches nothing" from "matches everything": an absent
// value returns an empty (non-nil) selection, never nil (which means all rows
// elsewhere). A mutation returning nil for the absent case would be a silent
// whole-table scan — caught here.
func TestDictFilterEqPresentAndAbsent(t *testing.T) {
	c := NewDictColumn([]string{"north", "south", "north", "east"})
	sel := c.FilterEq("north")
	if len(sel) != 2 || sel[0] != 0 || sel[1] != 2 {
		t.Fatalf("FilterEq(north) = %v, want [0 2]", sel)
	}
	absent := c.FilterEq("west")
	if absent == nil {
		t.Fatalf("FilterEq(absent) returned nil — must be empty non-nil to mean 'no rows', not 'all rows'")
	}
	if len(absent) != 0 {
		t.Fatalf("FilterEq(absent) = %v, want empty", absent)
	}
}

// ---- GroupSumFloat (the keystone cross-type op) ----

func TestGroupSumFloat(t *testing.T) {
	region := NewDictColumn([]string{"north", "south", "north", "east", "south"})
	revenue := NewFloat64Column([]float64{100, 200, 50, 25, 75})

	got := GroupSumFloat(region, revenue, nil)
	// Deterministic key-ascending order: east, north, south.
	want := []StringGroupResult{
		{Key: "east", Value: 25},
		{Key: "north", Value: 150}, // 100 + 50
		{Key: "south", Value: 275}, // 200 + 75
	}
	assertStringGroups(t, "all rows", got, want)

	// With a selection (e.g. the result of a prior filter): only rows 1,2,4.
	gotSel := GroupSumFloat(region, revenue, Selection{1, 2, 4})
	wantSel := []StringGroupResult{
		{Key: "north", Value: 50}, // row 2
		{Key: "south", Value: 275}, // rows 1,4
	}
	assertStringGroups(t, "selected", gotSel, wantSel)
}

func TestGroupSumFloatLengthMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("GroupSumFloat with mismatched column lengths did not panic")
		}
	}()
	keys := NewDictColumn([]string{"a", "b"})
	vals := NewFloat64Column([]float64{1}) // shorter on purpose
	GroupSumFloat(keys, vals, nil)
}

// ---- TypedTable ----

func TestTypedTableSchema(t *testing.T) {
	tt := NewTypedTable().
		AddInt64("id", NewInt64Column([]int64{1, 2, 3})).
		AddFloat64("revenue", NewFloat64Column([]float64{10, 20, 30})).
		AddString("region", NewDictColumn([]string{"n", "s", "n"})).
		AddBool("active", NewBoolColumn([]bool{true, false, true}))

	schema := tt.Schema()
	want := []ColumnSchema{
		{"id", TypeInt64}, {"revenue", TypeFloat64}, {"region", TypeString}, {"active", TypeBool},
	}
	if len(schema) != len(want) {
		t.Fatalf("Schema = %v, want %v", schema, want)
	}
	for i := range want {
		if schema[i] != want[i] {
			t.Fatalf("Schema[%d] = %+v, want %+v", i, schema[i], want[i])
		}
	}
	if TypeFloat64.String() != "float64" || TypeString.String() != "string" {
		t.Fatalf("ColumnType.String mismatch: %q %q", TypeFloat64, TypeString)
	}
}

func TestTypedTableRaggedPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("adding a shorter column did not panic at construction")
		}
	}()
	NewTypedTable().
		AddInt64("id", NewInt64Column([]int64{1, 2, 3})).
		AddFloat64("revenue", NewFloat64Column([]float64{10, 20})) // ragged
}

// A query asked of the wrong physical type must be a clean error, never a crash and
// never a silently wrong answer — the whole point of carrying types in the schema.
func TestTypedTableWrongTypeErrors(t *testing.T) {
	tt := NewTypedTable().
		AddString("region", NewDictColumn([]string{"n", "s"})).
		AddFloat64("revenue", NewFloat64Column([]float64{10, 20}))

	if _, err := tt.FilterGTFloat("region", 0); err == nil {
		t.Fatal("FilterGTFloat on a string column should error")
	}
	if _, err := tt.FilterEq("revenue", "n"); err == nil {
		t.Fatal("FilterEq on a float column should error")
	}
	if _, err := tt.FilterGTFloat("nope", 0); err == nil {
		t.Fatal("FilterGTFloat on an unknown column should error")
	}
	if _, err := tt.GroupSumFloat("revenue", "revenue", nil); err == nil {
		t.Fatal("GroupSumFloat with a non-string group column should error")
	}
}

// End-to-end through the typed table: "total revenue by region for rows over 60" —
// filter float → group string → sum float, all over the selection vector.
func TestTypedTableComposedFilterGroupSum(t *testing.T) {
	tt := NewTypedTable().
		AddString("region", NewDictColumn([]string{"north", "south", "north", "east", "south"})).
		AddFloat64("revenue", NewFloat64Column([]float64{100, 200, 50, 25, 75}))

	sel, err := tt.FilterGTFloat("revenue", 60) // rows 0 (100), 1 (200), 4 (75)
	if err != nil {
		t.Fatalf("FilterGTFloat: %v", err)
	}
	got, err := tt.GroupSumFloat("region", "revenue", sel)
	if err != nil {
		t.Fatalf("GroupSumFloat: %v", err)
	}
	want := []StringGroupResult{
		{Key: "north", Value: 100}, // only row 0 survived (row 2's 50 filtered out)
		{Key: "south", Value: 275}, // rows 1,4
	}
	assertStringGroups(t, "composed", got, want)
}

func assertStringGroups(t *testing.T, label string, got, want []StringGroupResult) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %d groups %v, want %d %v", label, len(got), got, len(want), want)
	}
	for i := range want {
		if got[i].Key != want[i].Key || got[i].Value != want[i].Value {
			t.Fatalf("%s: group[%d] = %+v, want %+v", label, i, got[i], want[i])
		}
	}
}
