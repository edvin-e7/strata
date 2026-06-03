package strata

import (
	"context"
	"net/http"
	"testing"
	"time"
)

func sampleTable() *Table {
	return NewTable().
		AddInt64("amount", NewInt64Column([]int64{100, 200, 300, 400, 500})).
		AddInt64("flag", NewInt64Column([]int64{1, 0, 1, 0, 1}))
}

func TestExecute(t *testing.T) {
	tb := sampleTable()
	if got, _ := tb.Execute(Query{Agg: "count"}); got != 5 {
		t.Fatalf("count all = %d, want 5", got)
	}
	got, err := tb.Execute(Query{Agg: "count", Filter: &Predicate{"amount", ">", 250}})
	if err != nil || got != 3 { // 300,400,500
		t.Fatalf("count(amount>250) = %d err=%v, want 3", got, err)
	}
	got, err = tb.Execute(Query{Agg: "sum", Column: "amount", Filter: &Predicate{"amount", ">", 250}})
	if err != nil || got != 1200 { // 300+400+500
		t.Fatalf("sum(amount where amount>250) = %d err=%v, want 1200", got, err)
	}
}

// The safety property that makes LLM-driving sound: a query naming a column or op
// that doesn't exist MUST error, never execute, never silently return wrong data.
func TestExecuteRejectsHallucination(t *testing.T) {
	tb := sampleTable()
	cases := []Query{
		{Agg: "sum", Column: "ssn"},                            // unknown sum column
		{Agg: "count", Filter: &Predicate{"nope", ">", 1}},     // unknown filter column
		{Agg: "median"},                                        // unsupported agg
		{Agg: "count", Filter: &Predicate{"amount", "<", 250}}, // unsupported op
	}
	for i, q := range cases {
		if _, err := tb.Execute(q); err == nil {
			t.Fatalf("case %d %+v: expected validation error, got nil", i, q)
		}
	}
}

// Live, guarded: actually talk to gemma-e7 if Ollama is up, and assert the
// English answer matches the deterministic ground truth. Skips cleanly when Ollama
// isn't reachable, so this never makes CI flaky.
func TestOllamaPlannerLive(t *testing.T) {
	if _, err := http.Get("http://localhost:11434/api/tags"); err != nil {
		t.Skip("Ollama not reachable — skipping live NL→query test")
	}
	tb := sampleTable()
	p := NewOllamaPlanner()
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	q, err := p.Plan(ctx, "how many rows have amount greater than 250?", tb.Schema())
	if err != nil {
		t.Fatalf("planner: %v", err)
	}
	got, err := tb.Execute(q)
	if err != nil {
		t.Fatalf("execute planned query %+v: %v", q, err)
	}
	if got != 3 {
		t.Fatalf("NL→query answered %d, want 3 (planned: %+v)", got, q)
	}
	t.Logf("live NL→query OK: %q → %+v → %d", "how many rows have amount greater than 250?", q, got)
}
