// Package strata is a from-scratch, vectorized, columnar in-memory data engine
// written in Go. It is built AI-native and local-first (see DESIGN.md).
//
// Honest positioning: strata does NOT claim to beat Polars or DuckDB on raw speed
// — those engines (Rust/C++, hand-tuned SIMD) own that, for hard reasons. strata's
// claim is different: a genuinely well-engineered columnar core, honestly
// benchmarked, differentiated by semantic/embedding columns and local-LLM querying
// that the fast engines do not have. The competence is in the engineering and the
// honesty, not in a benchmark we can't win.
package strata

// Int64Column is a typed, contiguous column. Contiguous storage is the whole point
// of a columnar engine: an operator sweeps the entire slice in one tight,
// branch-predictable, cache-friendly loop instead of chasing row-by-row pointers.
type Int64Column struct {
	Data []int64
}

// NewInt64Column wraps a backing slice as a column (no copy).
func NewInt64Column(data []int64) *Int64Column { return &Int64Column{Data: data} }

// Len reports the row count.
func (c *Int64Column) Len() int { return len(c.Data) }

// Selection is a vector of row indices that passed a predicate — the columnar
// engine's currency. Operators take and return selections instead of materializing
// filtered copies, so a filter→aggregate chain never copies the underlying data.
type Selection []uint32

// FilterGT returns the indices where value > threshold. A single tight loop over
// contiguous memory: this is the "vectorized" in vectorized execution.
func (c *Int64Column) FilterGT(threshold int64) Selection {
	sel := make(Selection, 0, len(c.Data))
	for i, v := range c.Data {
		if v > threshold {
			sel = append(sel, uint32(i))
		}
	}
	return sel
}

// Sum aggregates the whole column. Caveat (honest): like any naive int64 aggregate
// it can overflow and wrap silently past 2^63 — documented, not yet guarded. A
// widening/saturating accumulator is on the roadmap.
func (c *Int64Column) Sum() int64 {
	var total int64
	for _, v := range c.Data {
		total += v
	}
	return total
}

// SumAt aggregates only the selected rows — the second half of a filter→aggregate
// pipeline, with no intermediate materialization.
func (c *Int64Column) SumAt(sel Selection) int64 {
	var total int64
	for _, i := range sel {
		total += c.Data[i]
	}
	return total
}
