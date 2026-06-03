package strata

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// Planner turns a natural-language question into a structured Query. Whatever a
// Planner returns is always run through Table.Execute, which validates it — so the
// planner can never do anything the engine wouldn't allow.
type Planner interface {
	Plan(ctx context.Context, question string, schema []string) (Query, error)
}

// OllamaPlanner asks a LOCAL model (gemma-e7 via Ollama by default) to emit a Query
// as JSON. Nothing leaves the machine, no API key, $0. The model only proposes a
// plan; it never touches execution.
type OllamaPlanner struct {
	Host  string
	Model string
}

// NewOllamaPlanner targets the local Ollama default.
func NewOllamaPlanner() *OllamaPlanner {
	return &OllamaPlanner{Host: "http://localhost:11434", Model: "gemma-e7"}
}

const plannerSystem = `You translate a question about a data table into a JSON query.
Output ONLY a JSON object — no prose, no markdown fences.
Schema: {"agg":"sum"|"count","column":"<the column to sum, or empty string for count>","filter":{"column":"<name>","op":">","value":<integer>} }
Set "filter" to null when the question has no condition. Use ONLY column names from the provided schema. The operator MUST be the literal character ">" (greater-than) — never "$gt", "gt", or "<".`

type ollamaChatReq struct {
	Model    string              `json:"model"`
	Stream   bool                `json:"stream"`
	Options  map[string]any      `json:"options"`
	Messages []map[string]string `json:"messages"`
}

type ollamaChatResp struct {
	Message struct {
		Content string `json:"content"`
	} `json:"message"`
}

// Plan calls the local model and parses its JSON into a Query. Validation happens
// later, in Table.Execute.
func (p *OllamaPlanner) Plan(ctx context.Context, question string, schema []string) (Query, error) {
	user := fmt.Sprintf("Table columns: %s\nQuestion: %s", strings.Join(schema, ", "), question)
	reqBody, _ := json.Marshal(ollamaChatReq{
		Model:   p.Model,
		Stream:  false,
		Options: map[string]any{"temperature": 0},
		Messages: []map[string]string{
			{"role": "system", "content": plannerSystem},
			{"role": "user", "content": user},
		},
	})
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.Host+"/api/chat", bytes.NewReader(reqBody))
	if err != nil {
		return Query{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return Query{}, fmt.Errorf("ollama unreachable: %w", err)
	}
	defer resp.Body.Close()
	// A non-200 (e.g. 404 = model not pulled) decodes to an empty Content otherwise,
	// masking the real cause behind a confusing "unparseable query". Surface it.
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return Query{}, fmt.Errorf("ollama returned HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	var cr ollamaChatResp
	// Bound the body so a runaway/hostile local endpoint can't make us allocate without limit.
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&cr); err != nil {
		return Query{}, err
	}
	q, err := parseQueryJSON(cr.Message.Content)
	if err != nil {
		return Query{}, fmt.Errorf("model returned unparseable query %q: %w", cr.Message.Content, err)
	}
	return q, nil
}

// parseQueryJSON extracts the first complete JSON object (models sometimes add
// prose or fences despite instructions) and unmarshals it. Uses a brace-depth scan
// that respects string literals — robust by construction. The old first-'{'..last-'}'
// heuristic broke when prose adjacent to the JSON contained a stray brace (found by
// adversarial review 2026-06-04).
func parseQueryJSON(s string) (Query, error) {
	obj, ok := extractJSONObject(s)
	if !ok {
		return Query{}, fmt.Errorf("no JSON object in %q", s)
	}
	var q Query
	if err := json.Unmarshal([]byte(obj), &q); err != nil {
		return Query{}, err
	}
	// Be liberal in what we accept: small local models reach for familiar operator
	// dialects ($gt, gt, "greater than"). Normalize to the engine's canonical ">".
	// The executor stays strict — this just maps model-ese to the one true form.
	if q.Filter != nil {
		q.Filter.Op = normalizeOp(q.Filter.Op)
	}
	return q, nil
}

// extractJSONObject returns the first balanced {...} object in s, tracking string
// literals (and escapes) so a brace inside a string value doesn't end it early, and
// prose after the object is ignored.
func extractJSONObject(s string) (string, bool) {
	start := strings.IndexByte(s, '{')
	if start < 0 {
		return "", false
	}
	depth, inStr, esc := 0, false, false
	for i := start; i < len(s); i++ {
		switch c := s[i]; {
		case inStr && esc:
			esc = false
		case inStr && c == '\\':
			esc = true
		case inStr && c == '"':
			inStr = false
		case inStr:
			// any other byte inside a string literal
		case c == '"':
			inStr = true
		case c == '{':
			depth++
		case c == '}':
			depth--
			if depth == 0 {
				return s[start : i+1], true
			}
		}
	}
	return "", false
}

func normalizeOp(op string) string {
	switch strings.ToLower(strings.TrimSpace(op)) {
	case ">", "$gt", "gt", "greater", "greater than", "greaterthan", "gt.":
		return ">"
	}
	return op
}
