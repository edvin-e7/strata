package strata

import "sort"

// sort.go — Block 8: vectorized order-by / top-N, the last operator the MVP scope
// (DESIGN.md) named. Like every strata operator it speaks selection vectors: a sort
// does not move or rewrite a column, it returns a PERMUTATION — a Selection of original
// row indices that, gathered, walks the rows in order. So order-by composes with
// everything (filter → order-by → SumAt) and never copies the underlying data.

// OrderBy returns the rows ordered by this column's value: a Selection of original row
// indices such that gathering them walks the column in sorted order. desc=false is
// ascending, desc=true descending. sel restricts the sort to those rows (nil = all);
// the caller's selection is never mutated (OrderBy sorts a copy).
//
// Total, deterministic order: equal values always break by ASCENDING original row
// index, in both directions. That tiebreak makes the comparator a total order, so the
// result is identical across runs even though sort.Slice is not stable — and it makes
// TopN(k) exactly equal to OrderBy()[:k].
//
// Caveat (honest): this is a full comparison sort, O(m log m) over the m selected rows
// via the standard library's introsort — not a radix or parallel sort (a deliberate
// non-goal, like the stdlib map in GroupSum). When only the first few rows are wanted,
// TopN is the bounded O(m log k) path; prefer it over OrderBy()[:k].
func (c *Int64Column) OrderBy(sel Selection, desc bool) Selection {
	out := materializeSelection(sel, len(c.Data))
	sort.Slice(out, func(i, j int) bool {
		return rankBefore(c.Data[out[i]], out[i], c.Data[out[j]], out[j], desc)
	})
	return out
}

// TopN returns up to n rows ordered by this column's value — the bounded form of
// OrderBy and the common shape of a real query ("top 10 by revenue"). It keeps a
// running, sorted, length-n selection in a single pass, so it is O(m log n) — cheaper
// than OrderBy()[:n] when n ≪ m — while returning the identical rows in the identical
// deterministic order (value, then ascending row index). sel restricts the input
// (nil = all rows); n <= 0 yields an empty selection.
func (c *Int64Column) TopN(sel Selection, n int, desc bool) Selection {
	if n <= 0 {
		return Selection{}
	}
	top := make(Selection, 0, n)
	consider := func(idx uint32) { top = insertTopNInt64(top, c, idx, n, desc) }
	if sel == nil {
		for i := range c.Data {
			consider(uint32(i))
		}
	} else {
		for _, i := range sel {
			consider(i)
		}
	}
	return top
}

// rankBefore reports whether row (aVal,aIdx) sorts before (bVal,bIdx): the primary key
// is the value (descending when desc, else ascending), and equal values break by
// ascending row index so the order is total and reproducible.
func rankBefore(aVal int64, aIdx uint32, bVal int64, bIdx uint32, desc bool) bool {
	if aVal != bVal {
		if desc {
			return aVal > bVal
		}
		return aVal < bVal
	}
	return aIdx < bIdx
}

// materializeSelection returns a writable identity selection 0..n-1 when sel is nil,
// or a copy of sel otherwise. A copy because OrderBy sorts in place and must not
// scramble a selection the caller still holds (e.g. one feeding another op).
func materializeSelection(sel Selection, n int) Selection {
	if sel == nil {
		out := make(Selection, n)
		for i := range out {
			out[i] = uint32(i)
		}
		return out
	}
	out := make(Selection, len(sel))
	copy(out, sel)
	return out
}

// insertTopNInt64 keeps top as the best-n rows of column c, sorted best-first under
// rankBefore. It is the int64 analog of vector.go's insertTopK for cosine search —
// the same bounded-insertion idiom (unifying the two behind a generic is a deliberate
// later cleanup, not worth a type-parameter refactor of working code now). The
// strict-better / not-better boundary is exactly what resolves ties by ascending row
// index: an equal-valued row arriving later (larger index) is "not better" than the
// seated one under rankBefore, so it never evicts it — matching OrderBy precisely.
func insertTopNInt64(top Selection, c *Int64Column, idx uint32, n int, desc bool) Selection {
	if len(top) == n && !rankBefore(c.Data[idx], idx, c.Data[top[len(top)-1]], top[len(top)-1], desc) {
		return top // not better than the current worst, and no room
	}
	if len(top) < n {
		top = append(top, idx)
	} else {
		top[len(top)-1] = idx // replace the worst
	}
	// bubble the new row up until the slice is best-first again.
	for i := len(top) - 1; i > 0 && rankBefore(c.Data[top[i]], top[i], c.Data[top[i-1]], top[i-1], desc); i-- {
		top[i], top[i-1] = top[i-1], top[i]
	}
	return top
}
