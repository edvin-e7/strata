package strata

import (
	"math/rand"
	"testing"
)

func abs32(f float32) float32 {
	if f < 0 {
		return -f
	}
	return f
}

// Correctness on a hand-checkable case before any benchmark is trusted.
// query [1,0] vs rows: [1,0]→cos 1.0, [0,1]→0, [1,1]→~0.707, [-1,0]→-1.
// top-2 must be row 0 then row 2.
func TestTopKCosine(t *testing.T) {
	v := NewVectorColumn(2, []float32{1, 0, 0, 1, 1, 1, -1, 0})
	got := v.TopKCosine([]float32{1, 0}, 2)
	if len(got) != 2 {
		t.Fatalf("len = %d, want 2 (%+v)", len(got), got)
	}
	if got[0].Row != 0 || abs32(got[0].Score-1.0) > 1e-6 {
		t.Fatalf("top[0] = %+v, want row 0 score 1.0", got[0])
	}
	if got[1].Row != 2 || abs32(got[1].Score-0.70710677) > 1e-5 {
		t.Fatalf("top[1] = %+v, want row 2 score ~0.707", got[1])
	}
}

func benchEmbeddings(n, dim int) *VectorColumn {
	r := rand.New(rand.NewSource(7))
	data := make([]float32, n*dim)
	for i := range data {
		data[i] = r.Float32()
	}
	return NewVectorColumn(dim, data)
}

// Honest semantic-search benchmark: top-10 over 100k embeddings × 128 dims,
// single core, no SIMD, no GPU. Reproduce: go test -bench=Cosine -benchmem
func BenchmarkTopKCosine_100k_128d(b *testing.B) {
	v := benchEmbeddings(100_000, 128)
	r := rand.New(rand.NewSource(8))
	q := make([]float32, 128)
	for i := range q {
		q[i] = r.Float32()
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = v.TopKCosine(q, 10)
	}
}
