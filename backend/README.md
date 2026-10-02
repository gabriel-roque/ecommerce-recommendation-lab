# Backend

Backend FastAPI, SQLAlchemy e TensorFlow do E-commerce Recommendation Lab.

## Desenvolvimento

Da raiz do projeto:

```bash
PYTHONPATH=.:backend python -m pytest backend/tests -q
```

Dentro de `backend/`:

```bash
python -m mypy app
```

## Dados e bootstrap

`app.dataset.synthetic` gera e valida `data/ecommerce_sintetico_br`. Não há download externo nem schema legado. `python -m app.cli bootstrap`:

1. valida checksums, contagens, UTF-8, FKs, datas, preços BRL e canais;
2. compara `source_fingerprint` com `dataset_state`;
3. importa tabelas canônicas via PostgreSQL COPY somente quando necessário;
4. enfileira uma única run automática `full` para fingerprint ainda não treinado ou enfileirado.

## Contrato de recomendação

Rota personalizada aceita `model_weight`, `category_weight`, `popularity_weight` e `trend_weight`. Soma deve ser 100. Defaults são 45, 30, 15 e 10.

Cada item retorna:

- `score`: 0 a 100;
- `grade`: `baixo`, `medio`, `alto` ou `muito_alto`;
- `breakdown`: `raw_score`, `weight`, `contribution`, `label` e explicação factual;
- `evidence`: evidências observadas mantidas para consumidores atuais;
- `article`: campos genéricos de produto.

Com modelo ativo, API pontua todo catálogo não visto. Sem modelo, resposta usa popularidade e informa pesos efetivamente aplicados. Scores não são probabilidades nem inferências causais.

## Artefatos

Formato aceito: `ecommerce-hybrid`, versão 2, dataset `ecommerce_sintetico_br`. Carregamento exige `source_fingerprint` igual ao manifesto atual.
