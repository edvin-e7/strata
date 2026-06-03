package strata

import "math"

// VectorColumn stores fixed-width embeddings contiguously, row-major: NumRows × Dim
// float32s in one flat slice. Contiguous layout keeps a similarity scan
// cache-friendly — the same reason the scalar columns are contiguous.
//
// Why this is the differentiator: semantic search lives *inside* the engine, next
// to filter and aggregate, over data you own locally. Polars and DuckDB make you
// leave for a separate vector store; strata treats an embedding as just another
// column. That's the AI-native, local-first bet — and it costs zero GPU.
type VectorColumn struct {
	Dim  int
	Data []float32
}

// NewVectorColumn wraps a flat row-major slice (len must be a positive multiple of dim).
func NewVectorColumn(dim int, data []float32) *VectorColumn {
	if dim <= 0 || len(data)%dim != 0 {
		panic("strata: VectorColumn data length must be a positive multiple of dim")
	}
	return &VectorColumn{Dim: dim, Data: data}
}

// NumRows reports the embedding count.
func (v *VectorColumn) NumRows() int { return len(v.Data) / v.Dim }

// Row returns the i-th embedding as a sub-slice (no copy).
func (v *VectorColumn) Row(i int) []float32 {
	off := i * v.Dim
	return v.Data[off : off+v.Dim]
}

// Neighbor is one (row, cosine-score) search result.
type Neighbor struct {
	Row   uint32
	Score float32
}

// TopKCosine returns the k rows most cosine-similar to query, sorted best-first.
// One tight pass over the contiguous data computes each row's dot product and norm;
// a bounded insertion keeps only the running top-k. No GPU, no external vector DB.
func (v *VectorColumn) TopKCosine(query []float32, k int) []Neighbor {
	if k <= 0 || v.NumRows() == 0 || len(query) != v.Dim {
		return nil
	}
	var qNorm float64
	for _, q := range query {
		qNorm += float64(q) * float64(q)
	}
	qNorm = math.Sqrt(qNorm)
	if qNorm == 0 {
		return nil
	}

	top := make([]Neighbor, 0, k)
	for i := 0; i < v.NumRows(); i++ {
		row := v.Row(i)
		var dot, rNorm float64
		for j, q := range query {
			rv := float64(row[j])
			dot += float64(q) * rv
			rNorm += rv * rv
		}
		rNorm = math.Sqrt(rNorm)
		if rNorm == 0 {
			continue
		}
		top = insertTopK(top, Neighbor{Row: uint32(i), Score: float32(dot / (qNorm * rNorm))}, k)
	}
	return top
}

// insertTopK keeps top sorted descending by Score with length <= k.
func insertTopK(top []Neighbor, n Neighbor, k int) []Neighbor {
	if len(top) == k && n.Score <= top[len(top)-1].Score {
		return top // worse than the current worst, and no room
	}
	if len(top) < k {
		top = append(top, n)
	} else {
		top[len(top)-1] = n // replace the worst
	}
	// bubble n up until the slice is sorted descending again
	for i := len(top) - 1; i > 0 && top[i].Score > top[i-1].Score; i-- {
		top[i], top[i-1] = top[i-1], top[i]
	}
	return top
}
