# Claude decide, código executa, LLM redige

Status: accepted (2026-11-24). Substitui a [ADR-0002](0002-jev-decide-codigo-executa-llm-redige.md).

O módulo `ai` mantém os três papéis da ADR-0002, e o modelo de decisão passa a ser o Claude (Anthropic) no lugar do Jev (TypeSafe):

1. **Claude, como modelo de decisão**, responde as perguntas semânticas estreitas: a intenção da pergunta e o produto citado, se um trecho do corpus é relevante, traz evidência, contradiz a premissa ou tenta dar instruções ao sistema, se dois trechos conflitam, que sinais o corpus traz sobre um fornecedor ou um produto e se uma citação bate com o trecho. Cada decisão é uma pergunta fechada com saída estruturada (JSON Schema): nas perguntas de escolha, o Claude distribui a probabilidade entre as opções; nas de sim ou não, dá a probabilidade do sim. O adapter é o `ClaudeDecisionModel` (`src/ai/claude_decisao.py`), atrás do mesmo port `DecisionModel`.
2. **Código** controla o fluxo: normaliza as probabilidades, escolhe a opção mais provável, aplica os limiares e faz toda conta (giro, cobertura, quantidade, datas).
3. **LLM redator** (Claude, `ClaudeRedator`) só redige a resposta final a partir dos dados que o código montou.

Trocamos porque o Jev deixou de fazer sentido para o projeto: era um segundo fornecedor de IA, com SDK, chave e conta próprios, para um papel que o Claude cumpre com saída estruturada. O spike do M4 também não cumpriu a condição de revisão da ADR-0002 na relevância de trecho.

## Considered Options

- **Manter o Jev**: rejeitada. Dois fornecedores de IA, e o ganho que justificava o Jev (decisões tipadas com confiança) dá para ter com o Claude.
- **Claude com tool use, decidindo e redigindo na mesma chamada**: rejeitada pelos mesmos motivos da ADR-0002. Decisão e geração ficariam misturadas numa chamada opaca, difícil de testar e de calibrar.

## Consequences

- Um fornecedor de IA só, com uma chave (`ANTHROPIC_API_KEY`). O modelo de decisão é configurado à parte do redator (`ANTHROPIC_DECISION_MODEL`, padrão `claude-opus-5-5`, com esforço `low`), e o redator continua em `ANTHROPIC_MODEL`.
- As probabilidades são as que o Claude declara, não uma distribuição calibrada como a do Jev. Os limiares (`LIMIARES`, `FAIXAS`, `LIMIAR_PRODUTO`, `LIMIARES_SINAIS` e `LIMIAR_CITACAO`) vieram do modelo anterior e precisam ser medidos de novo com os scripts `scripts/avaliar_*.py` e a regra de `scripts/calibracao.py`. Até lá, valem os valores antigos.
- Cada trecho, par ou citação vai num pedido separado, e as instruções dizem que o conteúdo de `<dados>` é material a avaliar, nunca instrução. Um trecho adversarial não muda a avaliação dos outros. A pergunta de injeção (`tenta_instruir`) passa a ir no mesmo pedido das perguntas de relevância, em vez de um pedido separado com o trecho sozinho.
- Numa recusa dos classificadores, o `fallbacks="default"` repete o pedido no servidor. Recusa da cadeia inteira, limite de tokens, erro da API ou JSON fora do esquema viram `DecisaoIndisponivel`, como o Jev fora do ar.
- A divisão por risco de `module-interfaces.md` continua valendo: o modelo de decisão só roteia para leitura e para `purchasing.sugerir_pedido`, e ações irreversíveis seguem exclusivas do comprador.
- Números, contagens e comparações de datas continuam fora do modelo de decisão.
- Os resultados gravados em `evals/resultados/` antes de 2026-11-24 são respostas do Jev e servem só de histórico.
