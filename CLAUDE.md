## Agent skills

### Issue tracker

Issues live as local markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: one `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

### IA (módulo `ai`)

Segue a ADR-0008: Claude decide, código executa, LLM redige. Antes de mexer em `ai`, ler `docs/adr/0008-claude-decide-codigo-executa-llm-redige.md`. O modelo de decisão é o `ClaudeDecisionModel` (`src/ai/claude_decisao.py`); os limiares só mudam medidos com os scripts `scripts/avaliar_*.py`.
