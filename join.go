package strata

import "fmt"

// join.go — Block 7: the vectorized hash-join, the last operator the MVP scope
// (DESIGN.md) named but the first six blocks left unbuilt.
//
// Like every strata operator the join speaks the engine's one currency — selection
// vectors of row indices, never materialized rows. An inner equi-join produces a set
// of matched (left-row, right-row) pairs; we return them as two parallel selections
// so a downstream project or aggregate gathers straight from the original columns
// (e.g. valueCol.SumAt(jr.Left)) with no intermediate copy. That is the whole point
// of doing the join columnar instead of building joined rows.

// JoinResult is the output of an equi-join: two parallel selection vectors of equal
// length. Row Left[k] of the left input matches row Right[k] of the right input.
// The indices are positions in the ORIGINAL columns (not positions within any input
// selection), so the result composes directly with every other op: the matched
// left values are leftValueCol.SumAt(jr.Left), the matched right values
// rightValueCol.SumAt(jr.Right), and so on — no joined-row table is ever built.
type JoinResult struct {
	Left  Selection
	Right Selection
}

// Len reports the number of matched row pairs.
func (j JoinResult) Len() int { return len(j.Left) }

// HashJoinInt64 is an inner equi-join on two int64 key columns: it returns every
// (left-row, right-row) pair whose keys are equal. Classic two-phase hash-join —
// build a hash table on one side, then probe it with the other in a single pass —
// which is O(L+R) instead of the O(L·R) nested-loop join.
//
// leftSel / rightSel are optional pre-filters (nil = all rows), exactly like
// GroupSum's selection: a filter→join chain flows the selection vectors straight in,
// so only the surviving rows ever participate and nothing is materialized between the
// filter and the join. A non-nil selection is trusted to index its own column — the
// same contract as SumAt/GroupSum, because selections come from the engine's own
// FilterGT, never from the LLM. An out-of-range index panics rather than erroring, by
// design: the validate-everything surface is the LLM-supplied query/keys, not raw
// index vectors the engine produced itself.
//
// Build side: strata builds the hash table on the RIGHT input and probes with the
// left. For least memory, pass the SMALLER relation as right. strata deliberately
// does NOT pick the build side for you — choosing it from cardinality estimates is
// cost-based query planning, an explicit non-goal (DESIGN.md). Honesty over a knob we
// can't tune well.
//
// Output order is deterministic — a non-deterministic engine result is a correctness
// bug, not a cosmetic one (the same doctrine that pins GroupSum's key order). The
// pairs come out ordered by probe iteration (left rows in leftSel order, or ascending
// index when leftSel is nil) and, within each left row, by build iteration (right
// rows in rightSel order, or ascending index when rightSel is nil). We never iterate
// the Go map to produce output, so map-iteration randomization can't leak in and no
// post-sort is needed.
//
// Caveat (honest): output size is inherent to inner-join semantics — duplicate keys on
// both sides produce the per-key cartesian product, so a join on a low-cardinality key
// can blow up to O(L·R) rows. That is the join, not a strata flaw; we just don't hide
// it. Memory is O(rows on the build/right side) for the hash table.
func HashJoinInt64(left *Int64Column, leftSel Selection, right *Int64Column, rightSel Selection) JoinResult {
	// Build phase: hash the right side, key → right row indices in build order.
	build := make(map[int64][]uint32)
	if rightSel == nil {
		for i, k := range right.Data {
			build[k] = append(build[k], uint32(i))
		}
	} else {
		for _, i := range rightSel {
			build[right.Data[i]] = append(build[right.Data[i]], i)
		}
	}

	// Probe phase: scan the left side in order, emit a pair per matching build row.
	// Pre-size to the probe count: a reasonable floor (one-to-one is the common case);
	// it grows for one-to-many without changing the result.
	probeLen := len(left.Data)
	if leftSel != nil {
		probeLen = len(leftSel)
	}
	res := JoinResult{Left: make(Selection, 0, probeLen), Right: make(Selection, 0, probeLen)}
	probe := func(li uint32) {
		for _, ri := range build[left.Data[li]] {
			res.Left = append(res.Left, li)
			res.Right = append(res.Right, ri)
		}
	}
	if leftSel == nil {
		for i := range left.Data {
			probe(uint32(i))
		}
	} else {
		for _, i := range leftSel {
			probe(i)
		}
	}
	return res
}

// Join is the table-level equi-join: it joins two tables on named key columns and
// inherits Execute's validate-everything contract. An unknown key column on either
// side is a clean error, never a crash and never a silently-wrong join — the same
// safety that lets an LLM drive the scalar/grouped paths extends to joins. leftSel /
// rightSel are optional pre-filters resolved against each table.
//
// Returns row-index pairs (JoinResult); the caller projects or aggregates the matched
// rows from whichever columns it wants, e.g. orders.Column("amount").SumAt(jr.Left).
func Join(left, right *Table, leftKey, rightKey string, leftSel, rightSel Selection) (JoinResult, error) {
	lc, ok := left.cols[leftKey]
	if !ok {
		return JoinResult{}, fmt.Errorf("unknown left join column %q", leftKey)
	}
	rc, ok := right.cols[rightKey]
	if !ok {
		return JoinResult{}, fmt.Errorf("unknown right join column %q", rightKey)
	}
	return HashJoinInt64(lc, leftSel, rc, rightSel), nil
}

// Column exposes a named int64 column so a caller can gather join/filter results out
// of a table (jr.Left / jr.Right index straight into it). Returns nil for an unknown
// name: this is a trusted programmer-facing accessor, NOT the LLM-facing validated
// surface. Execute/Join validate the query and the join keys; choosing the right
// column to gather afterwards is the caller's own responsibility, exactly like
// indexing a slice — a typo'd name yields nil, not a validated error.
func (t *Table) Column(name string) *Int64Column { return t.cols[name] }
