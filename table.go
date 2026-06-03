package strata

import "fmt"

// Table is a minimal set of named Int64 columns — enough to give the natural-
// language query layer something real to run against. Scope is honest: int64-only
// for now (float/string columns are on the roadmap).
type Table struct {
	cols  map[string]*Int64Column
	order []string
}

// NewTable returns an empty table.
func NewTable() *Table { return &Table{cols: map[string]*Int64Column{}} }

// AddInt64 attaches a named column (insertion order preserved for the schema).
func (t *Table) AddInt64(name string, c *Int64Column) *Table {
	if _, ok := t.cols[name]; !ok {
		t.order = append(t.order, name)
	}
	t.cols[name] = c
	return t
}

// Schema returns column names in insertion order — handed to the planner so the
// model can only reference columns that actually exist.
func (t *Table) Schema() []string { return t.order }

func (t *Table) rows() int {
	for _, name := range t.order {
		return t.cols[name].Len()
	}
	return 0
}

// Predicate is a single comparison. The operator set is intentionally tiny for now.
type Predicate struct {
	Column string `json:"column"`
	Op     string `json:"op"`
	Value  int64  `json:"value"`
}

// Query is a structured, validated request against a Table. The LLM produces THIS —
// never SQL it executes, never code — so a hallucinated query fails validation
// instead of running. The model proposes; the engine disposes.
type Query struct {
	Agg    string     `json:"agg"`    // "sum" | "count"
	Column string     `json:"column"` // column to sum (ignored for count)
	Filter *Predicate `json:"filter"` // optional
}

// Execute validates q against the table's real schema, then runs it on the
// vectorized columnar ops. Every unknown column or unsupported op is a clean error,
// never a crash and never a wrong silent result — that safety is the whole point of
// letting an LLM drive.
func (t *Table) Execute(q Query) (int64, error) {
	var sel Selection
	filtered := false
	if q.Filter != nil {
		c, ok := t.cols[q.Filter.Column]
		if !ok {
			return 0, fmt.Errorf("unknown filter column %q", q.Filter.Column)
		}
		if q.Filter.Op != ">" {
			return 0, fmt.Errorf("unsupported operator %q (only %q)", q.Filter.Op, ">")
		}
		sel = c.FilterGT(q.Filter.Value)
		filtered = true
	}
	switch q.Agg {
	case "count":
		if filtered {
			return int64(len(sel)), nil
		}
		return int64(t.rows()), nil
	case "sum":
		c, ok := t.cols[q.Column]
		if !ok {
			return 0, fmt.Errorf("unknown column %q to sum", q.Column)
		}
		if filtered {
			return c.SumAt(sel), nil
		}
		return c.Sum(), nil
	default:
		return 0, fmt.Errorf("unsupported aggregate %q (want sum|count)", q.Agg)
	}
}
