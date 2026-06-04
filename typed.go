package strata

import (
	"fmt"
	"math/bits"
	"sort"
)

// typed.go — Block 2 of the roadmap: the typed column core. The seed engine was
// int64-only (column.go, table.go); a real columnar engine carries several physical
// types, each with its own contiguous layout and its own honest cost story. This
// file adds three: float64, dictionary-encoded string, and a bit-packed bool — plus
// a tiny TypedTable that stores them together behind a typed schema and runs
// validate-everything operations across them (the same "never a crash, never a
// silently wrong answer" contract Table.Execute holds for the int64 path).
//
// The NL→query planner still drives the int64 Table; migrating it onto TypedTable
// is a later block. Keeping that surface frozen here is deliberate — block 2 grows
// the typed core without destabilising the validated LLM path.

// ---------------------------------------------------------------------------
// Float64Column
// ---------------------------------------------------------------------------

// Float64Column is a contiguous column of float64s — the same cache-friendly
// single-slice layout as Int64Column, for the same reason: an operator sweeps it in
// one tight loop. Floats are their own type, not int64s in disguise: comparisons
// must respect NaN, and summation is order-dependent and lossy in ways integers are
// not. Those differences are documented at each op rather than papered over.
type Float64Column struct {
	Data []float64
}

// NewFloat64Column wraps a backing slice as a column (no copy).
func NewFloat64Column(data []float64) *Float64Column { return &Float64Column{Data: data} }

// Len reports the row count.
func (c *Float64Column) Len() int { return len(c.Data) }

// FilterGT returns the indices where value > threshold. NaN is excluded for free:
// every comparison against NaN is false in IEEE-754, so a NaN row never enters the
// selection — the correct behaviour (a NaN is not "greater than" anything), achieved
// by the language semantics rather than a special-case branch.
func (c *Float64Column) FilterGT(threshold float64) Selection {
	sel := make(Selection, 0, len(c.Data))
	for i, v := range c.Data {
		if v > threshold {
			sel = append(sel, uint32(i))
		}
	}
	return sel
}

// Sum aggregates the whole column. Caveat (honest): float summation is
// order-dependent and accumulates rounding error, and a single NaN/Inf in the data
// propagates to the result — both are IEEE-754 reality, not bugs. The order here is
// the column's contiguous order, so the (lossy) result is at least deterministic. A
// compensated (Kahan) accumulator is a future option; the seed stays naive on
// purpose, like the int64 path.
func (c *Float64Column) Sum() float64 {
	var total float64
	for _, v := range c.Data {
		total += v
	}
	return total
}

// SumAt aggregates only the selected rows — the float half of a filter→aggregate
// pipeline, with no intermediate materialization. Same determinism/precision caveat
// as Sum.
func (c *Float64Column) SumAt(sel Selection) float64 {
	var total float64
	for _, i := range sel {
		total += c.Data[i]
	}
	return total
}

// ---------------------------------------------------------------------------
// BoolColumn — bit-packed
// ---------------------------------------------------------------------------

// BoolColumn stores one bit per row, packed 64 to a uint64 word. A bool needs a
// single bit, so a []bool would waste 8× the memory and the cache lines that go with
// it; bit-packing is the honest columnar representation, and it turns "how many are
// true" into a popcount sweep and "which rows are true" into set-bit iteration —
// both genuinely vectorized, no per-row branch on the value.
type BoolColumn struct {
	words []uint64
	n     int
}

// NewBoolColumn packs a []bool into the bitset. Only true bits are ever set, and
// only for indices < n, so the bits past the last row stay zero — which is what lets
// CountTrue popcount every word without a tail mask (see there).
func NewBoolColumn(data []bool) *BoolColumn {
	c := &BoolColumn{words: make([]uint64, (len(data)+63)/64), n: len(data)}
	for i, v := range data {
		if v {
			c.words[i>>6] |= 1 << uint(i&63)
		}
	}
	return c
}

// Len reports the row count.
func (c *BoolColumn) Len() int { return c.n }

// At reports whether row i is true.
func (c *BoolColumn) At(i int) bool { return c.words[i>>6]&(1<<uint(i&63)) != 0 }

// CountTrue returns the number of true rows: a popcount over every word. No tail
// mask is needed because construction never sets a bit at index >= n, so the unused
// high bits of the final word are guaranteed zero — masking them would be dead code,
// and pretending otherwise would be cargo-cult defensiveness. (The "ragged tail"
// test pins this: a length that is not a multiple of 64 must still count correctly.)
func (c *BoolColumn) CountTrue() int {
	var total int
	for _, w := range c.words {
		total += bits.OnesCount64(w)
	}
	return total
}

// FilterTrue returns the indices of the true rows, in ascending order. It iterates
// set bits directly (clear-lowest-set-bit), skipping whole zero words — the columnar
// idiom for a sparse boolean, far cheaper than testing every row when trues are few.
// Because no bit past n is ever set, no out-of-range row can be emitted.
func (c *BoolColumn) FilterTrue() Selection {
	sel := make(Selection, 0, c.n)
	for w, word := range c.words {
		for word != 0 {
			row := w*64 + bits.TrailingZeros64(word)
			sel = append(sel, uint32(row))
			word &= word - 1 // clear the lowest set bit
		}
	}
	return sel
}

// ---------------------------------------------------------------------------
// DictColumn — dictionary-encoded string
// ---------------------------------------------------------------------------

// DictColumn dictionary-encodes a string column: every distinct string is stored
// once in dict, and each row holds a small integer code into it. This is the
// standard columnar string representation, and the payoff is that string operations
// become integer operations — an equality filter resolves the target to a code once,
// then compares ints; a group-by groups on codes (cheap) and only decodes to strings
// at the end. Low-cardinality columns (region, status, category) compress hard.
type DictColumn struct {
	dict  []string          // code -> string; index is the code
	codes []uint32          // per-row code into dict
	index map[string]uint32 // string -> code; for building and equality lookup
}

// NewDictColumn builds the dictionary and the per-row codes in one pass.
// Caveat (honest): codes are uint32, so cardinality is capped at ~4.3B distinct
// values — ample for the real use (categoricals), and the cap is documented rather
// than silently wrapped.
func NewDictColumn(values []string) *DictColumn {
	c := &DictColumn{index: make(map[string]uint32, len(values)), codes: make([]uint32, len(values))}
	for i, v := range values {
		code, ok := c.index[v]
		if !ok {
			code = uint32(len(c.dict))
			c.dict = append(c.dict, v)
			c.index[v] = code
		}
		c.codes[i] = code
	}
	return c
}

// Len reports the row count.
func (c *DictColumn) Len() int { return len(c.codes) }

// At returns the string at row i (decoded through the dictionary).
func (c *DictColumn) At(i int) string { return c.dict[c.codes[i]] }

// Cardinality reports the number of distinct values.
func (c *DictColumn) Cardinality() int { return len(c.dict) }

// FilterEq returns the indices whose value equals value. The string is resolved to
// its code once; the scan then compares uint32s, not strings. A value absent from
// the dictionary yields an empty (non-nil, zero-length) selection — distinct from a
// nil selection, which elsewhere means "all rows": "matches nothing" and "matches
// everything" must not collapse to the same thing.
func (c *DictColumn) FilterEq(value string) Selection {
	sel := make(Selection, 0)
	code, ok := c.index[value]
	if !ok {
		return sel // value never appears -> matches no rows
	}
	for i, code2 := range c.codes {
		if code2 == code {
			sel = append(sel, uint32(i))
		}
	}
	return sel
}

// StringGroupResult is one group keyed by a (decoded) string with a float aggregate —
// the shape of "revenue by region", the canonical low-cardinality columnar workload.
type StringGroupResult struct {
	Key   string  `json:"key"`
	Value float64 `json:"value"`
}

// GroupSumFloat groups rows by a dict-encoded string column and sums a float64 column
// within each group — the keystone cross-type operation, exercising string keys,
// float values, and the selection vector together. Grouping runs on the integer
// codes (the dictionary-encoding payoff: int-keyed map, no string hashing per row),
// and codes are decoded to strings only when building the result. When sel is nil all
// rows participate; otherwise the selection flows straight through, so a
// filter→group-by→sum chain never materializes an intermediate.
//
// Results are sorted by key string ascending: Go randomizes map iteration order, and
// a non-deterministic engine result is a correctness bug, not a cosmetic one (the
// same pin GroupSum applies to int keys). The float Value carries Sum's precision
// caveat. A key/value length mismatch is a caller error and panics at the call, like
// GroupSum, rather than reading out of range mid-aggregate.
func GroupSumFloat(keys *DictColumn, values *Float64Column, sel Selection) []StringGroupResult {
	if len(keys.codes) != len(values.Data) {
		panic(fmt.Sprintf("strata: GroupSumFloat key column has %d rows but value column has %d — must match", len(keys.codes), len(values.Data)))
	}
	acc := make(map[uint32]float64)
	if sel == nil {
		for i, code := range keys.codes {
			acc[code] += values.Data[i]
		}
	} else {
		for _, i := range sel {
			acc[keys.codes[i]] += values.Data[i]
		}
	}
	out := make([]StringGroupResult, 0, len(acc))
	for code, v := range acc {
		out = append(out, StringGroupResult{Key: keys.dict[code], Value: v})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Key < out[j].Key })
	return out
}

// ---------------------------------------------------------------------------
// TypedTable — the tiny typed table
// ---------------------------------------------------------------------------

// ColumnType names a column's physical type, so a schema can describe what each
// column actually is — the planner (and a human) needs that to know which ops are
// legal on which column.
type ColumnType int

const (
	TypeInt64 ColumnType = iota
	TypeFloat64
	TypeBool
	TypeString // dictionary-encoded
)

// String renders a ColumnType for schemas and error messages.
func (t ColumnType) String() string {
	switch t {
	case TypeInt64:
		return "int64"
	case TypeFloat64:
		return "float64"
	case TypeBool:
		return "bool"
	case TypeString:
		return "string"
	default:
		return fmt.Sprintf("ColumnType(%d)", int(t))
	}
}

// ColumnSchema is one named, typed column in a TypedTable's schema.
type ColumnSchema struct {
	Name string     `json:"name"`
	Type ColumnType `json:"type"`
}

// column is the minimal contract every physical column satisfies — enough for the
// table to enforce equal length without caring about the concrete type.
type column interface{ Len() int }

type typedCol struct {
	col column
	typ ColumnType
}

// TypedTable holds named columns of mixed physical type behind a typed schema. It is
// the int64-only Table grown up: same insertion-ordered schema and same
// equal-length-or-panic construction contract, but heterogeneous, and its query
// methods validate the column type as well as its existence before running — so an
// op asked of the wrong type is a clean error, never a crash or a wrong answer.
type TypedTable struct {
	cols  map[string]typedCol
	order []string
}

// NewTypedTable returns an empty table.
func NewTypedTable() *TypedTable { return &TypedTable{cols: map[string]typedCol{}} }

// AddInt64 / AddFloat64 / AddBool / AddString attach a named, typed column. The type
// is fixed by which method you call, so a column can never be mislabelled in the
// schema — a stronger guarantee than a generic Add(col, declaredType) that trusts
// the caller to pass a matching pair.
func (t *TypedTable) AddInt64(name string, c *Int64Column) *TypedTable {
	return t.addCol(name, c, TypeInt64)
}
func (t *TypedTable) AddFloat64(name string, c *Float64Column) *TypedTable {
	return t.addCol(name, c, TypeFloat64)
}
func (t *TypedTable) AddBool(name string, c *BoolColumn) *TypedTable {
	return t.addCol(name, c, TypeBool)
}
func (t *TypedTable) AddString(name string, c *DictColumn) *TypedTable {
	return t.addCol(name, c, TypeString)
}

// addCol enforces the equal-length invariant at construction — a ragged table is a
// caller error and panics HERE, not as an out-of-range crash mid-query — exactly as
// Table.AddInt64 does, so TypedTable's validate-everything query contract stays honest.
func (t *TypedTable) addCol(name string, c column, typ ColumnType) *TypedTable {
	if len(t.order) > 0 {
		if rows := t.rows(); c.Len() != rows {
			panic(fmt.Sprintf("strata: column %q has %d rows but the table has %d — all columns must be equal length", name, c.Len(), rows))
		}
	}
	if _, ok := t.cols[name]; !ok {
		t.order = append(t.order, name)
	}
	t.cols[name] = typedCol{col: c, typ: typ}
	return t
}

// Schema returns the columns as (name, type) in insertion order — a defensive copy,
// so a caller can't mutate the table's schema through it.
func (t *TypedTable) Schema() []ColumnSchema {
	out := make([]ColumnSchema, len(t.order))
	for i, name := range t.order {
		out[i] = ColumnSchema{Name: name, Type: t.cols[name].typ}
	}
	return out
}

func (t *TypedTable) rows() int {
	for _, name := range t.order {
		return t.cols[name].col.Len()
	}
	return 0
}

// FilterGTFloat resolves "column > threshold" on a float64 column to a selection.
// An unknown column or a non-float column is a clean error, never a crash.
func (t *TypedTable) FilterGTFloat(name string, threshold float64) (Selection, error) {
	tc, ok := t.cols[name]
	if !ok {
		return nil, fmt.Errorf("unknown column %q", name)
	}
	c, ok := tc.col.(*Float64Column)
	if !ok {
		return nil, fmt.Errorf("column %q is %s, not float64 — > on it is undefined", name, tc.typ)
	}
	return c.FilterGT(threshold), nil
}

// FilterEq resolves "column == value" on a dict-encoded string column to a selection.
func (t *TypedTable) FilterEq(name, value string) (Selection, error) {
	tc, ok := t.cols[name]
	if !ok {
		return nil, fmt.Errorf("unknown column %q", name)
	}
	c, ok := tc.col.(*DictColumn)
	if !ok {
		return nil, fmt.Errorf("column %q is %s, not string — == on it is undefined", name, tc.typ)
	}
	return c.FilterEq(value), nil
}

// FilterTrue resolves "where column is true" on a bool column to a selection.
func (t *TypedTable) FilterTrue(name string) (Selection, error) {
	tc, ok := t.cols[name]
	if !ok {
		return nil, fmt.Errorf("unknown column %q", name)
	}
	c, ok := tc.col.(*BoolColumn)
	if !ok {
		return nil, fmt.Errorf("column %q is %s, not bool", name, tc.typ)
	}
	return c.FilterTrue(), nil
}

// GroupSumFloat groups by a string column and sums a float64 column per group,
// optionally pre-filtered by sel. Both columns are validated for existence AND type
// before anything runs — the cross-type op that proves the typed core does real work
// ("revenue by region"), with the same clean-error-never-crash contract as the rest.
func (t *TypedTable) GroupSumFloat(groupCol, valueCol string, sel Selection) ([]StringGroupResult, error) {
	gc, ok := t.cols[groupCol]
	if !ok {
		return nil, fmt.Errorf("unknown group_by column %q", groupCol)
	}
	keys, ok := gc.col.(*DictColumn)
	if !ok {
		return nil, fmt.Errorf("group_by column %q is %s, not string", groupCol, gc.typ)
	}
	vc, ok := t.cols[valueCol]
	if !ok {
		return nil, fmt.Errorf("unknown value column %q", valueCol)
	}
	vals, ok := vc.col.(*Float64Column)
	if !ok {
		return nil, fmt.Errorf("value column %q is %s, not float64", valueCol, vc.typ)
	}
	return GroupSumFloat(keys, vals, sel), nil
}
