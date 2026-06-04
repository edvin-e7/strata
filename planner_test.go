package strata

import "testing"

// Locks the brace-scanner fix (the old first-'{'..last-'}' heuristic broke on these).
func TestParseQueryJSON(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want Query
	}{
		{"trailing prose with braces", `{"agg":"count","filter":null} (note: {} = no filter)`, Query{Agg: "count"}},
		{"leading fence", "```json\n{\"agg\":\"sum\",\"column\":\"amount\"}\n```", Query{Agg: "sum", Column: "amount"}},
		{"brace inside string value", `{"agg":"sum","column":"am}ount"}`, Query{Agg: "sum", Column: "am}ount"}},
		{"op dialect normalized", `{"agg":"count","filter":{"column":"x","op":"$gt","value":5}}`, Query{Agg: "count", Filter: &Predicate{"x", ">", 5}}},
		// Chatty model emits a stray brace in prose BEFORE the real JSON. The first
		// balanced object ("{each flag}") isn't a Query — recover the real one.
		{"stray prose brace before object", `Sure! Here's the query for {each flag}: {"agg":"sum","column":"amount"}`, Query{Agg: "sum", Column: "amount"}},
		// Leading "{}" unmarshals cleanly into a ZERO Query; must NOT swallow the real one.
		{"leading empty object before real one", `Result {}: {"agg":"count","column":""}`, Query{Agg: "count"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := parseQueryJSON(c.in)
			if err != nil {
				t.Fatalf("parseQueryJSON(%q) error: %v", c.in, err)
			}
			if got.Agg != c.want.Agg || got.Column != c.want.Column {
				t.Fatalf("got {agg:%q col:%q}, want {agg:%q col:%q}", got.Agg, got.Column, c.want.Agg, c.want.Column)
			}
			if (got.Filter == nil) != (c.want.Filter == nil) {
				t.Fatalf("filter presence mismatch: got %+v want %+v", got.Filter, c.want.Filter)
			}
			if got.Filter != nil && (got.Filter.Column != c.want.Filter.Column || got.Filter.Op != c.want.Filter.Op || got.Filter.Value != c.want.Filter.Value) {
				t.Fatalf("filter got %+v want %+v", got.Filter, c.want.Filter)
			}
		})
	}
}

func TestParseQueryJSON_NoObject(t *testing.T) {
	if _, err := parseQueryJSON("sorry, no idea"); err == nil {
		t.Fatal("expected error when there is no JSON object")
	}
}
