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
// All columns must be equal length: a ragged table is a caller error and panics
// HERE, at construction, rather than risking an out-of-range panic at query time on
// a filter→sum across mismatched columns. That keeps Execute's "never a crash on
// validated input" contract honest. (Found by adversarial review 2026-06-04.)
func (t *Table) AddInt64(name string, c *Int64Column) *Table {
	if len(t.order) > 0 {
		if rows := t.rows(); c.Len() != rows {
			panic(fmt.Sprintf("strata: column %q has %d rows but the table has %d — all columns must be equal length", name, c.Len(), rows))
		}
	}
	if _, ok := t.cols[name]; !ok {
		t.order = append(t.order, name)
	}
	t.cols[name] = c
	return t
}

// Schema returns column names in insertion order — handed to the planner so the
// model can only reference columns that actually exist. A defensive copy, so a
// caller can't reorder the table's internal schema through the returned slice.
func (t *Table) Schema() []string { return append([]string(nil), t.order...) }

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
//
// Routing: a query with GroupBy set returns one row per group → call ExecuteGroup; an
// empty GroupBy is a scalar aggregate → call Execute. (Execute and ExecuteGroup each
// reject the other's shape rather than silently collapsing it to the wrong answer.)
type Query struct {
	Agg     string     `json:"agg"`      // "sum" | "count"
	Column  string     `json:"column"`   // column to sum (ignored for count)
	GroupBy string     `json:"group_by"` // optional: group rows by this column before aggregating
	Filter  *Predicate `json:"filter"`   // optional
}

// applyFilter resolves an optional predicate to a selection vector. It returns
// (nil, nil) when there is no filter — callers read a nil selection as "all rows",
// which lets Sum/count take their whole-column fast path. An unknown column or
// unsupported op is a clean validation error, never a crash.
func (t *Table) applyFilter(f *Predicate) (Selection, error) {
	if f == nil {
		return nil, nil
	}
	c, ok := t.cols[f.Column]
	if !ok {
		return nil, fmt.Errorf("unknown filter column %q", f.Column)
	}
	if f.Op != ">" {
		return nil, fmt.Errorf("unsupported operator %q (only %q)", f.Op, ">")
	}
	return c.FilterGT(f.Value), nil
}

// Execute validates a scalar q against the table's real schema, then runs it on the
// vectorized columnar ops. Every unknown column or unsupported op is a clean error,
// never a crash and never a wrong silent result — that safety is the whole point of
// letting an LLM drive. A nil selection (no filter) takes the whole-column path.
func (t *Table) Execute(q Query) (int64, error) {
	if q.GroupBy != "" {
		return 0, fmt.Errorf("query groups by %q — use ExecuteGroup for grouped results", q.GroupBy)
	}
	sel, err := t.applyFilter(q.Filter)
	if err != nil {
		return 0, err
	}
	switch q.Agg {
	case "count":
		if sel != nil {
			return int64(len(sel)), nil
		}
		return int64(t.rows()), nil
	case "sum":
		c, ok := t.cols[q.Column]
		if !ok {
			return 0, fmt.Errorf("unknown column %q to sum", q.Column)
		}
		if sel != nil {
			return c.SumAt(sel), nil
		}
		return c.Sum(), nil
	default:
		return 0, fmt.Errorf("unsupported aggregate %q (want sum|count)", q.Agg)
	}
}

// ExecuteGroup runs a grouped q: it groups rows by q.GroupBy and aggregates per group,
// optionally pre-filtered. It shares Execute's validate-everything contract — an
// unknown group/sum/filter column or unsupported op is a clean error, never a crash
// and never a silently wrong group. Results are sorted by key (see GroupSum).
func (t *Table) ExecuteGroup(q Query) ([]GroupResult, error) {
	if q.GroupBy == "" {
		return nil, fmt.Errorf("ExecuteGroup needs group_by — use Execute for a scalar aggregate")
	}
	keys, ok := t.cols[q.GroupBy]
	if !ok {
		return nil, fmt.Errorf("unknown group_by column %q", q.GroupBy)
	}
	sel, err := t.applyFilter(q.Filter)
	if err != nil {
		return nil, err
	}
	switch q.Agg {
	case "count":
		return GroupCount(keys, sel), nil
	case "sum":
		c, ok := t.cols[q.Column]
		if !ok {
			return nil, fmt.Errorf("unknown column %q to sum", q.Column)
		}
		return GroupSum(keys, c, sel), nil
	default:
		return nil, fmt.Errorf("unsupported aggregate %q (want sum|count)", q.Agg)
	}
}
