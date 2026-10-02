# E-commerce Recommendation Lab

Laboratório didático de recomendação para e-commerce brasileiro. O projeto usa apenas dados sintéticos em PT-BR, versionados no repositório, com preços em BRL.

## Dataset

O dataset padrão é `ecommerce_sintetico_br` e fica em `data/ecommerce_sintetico_br/`:

- `customers.csv`: 2.000 clientes brasileiros;
- `products.csv`: 500 produtos genéricos;
- `purchases.csv`: 47.500 compras;
- `dataset_manifest.json`: checksums, fingerprint, contagens e contrato da fonte;
- `dataset_stats.json`: estatísticas observadas.

Os três CSVs somam exatamente 50.000 registros. O gerador usa seed fixa e produz os mesmos bytes em toda execução:

```bash
PYTHONPATH=backend python -m app.cli generate-data
PYTHONPATH=backend python -m app.cli verify-data
```

O sinal sintético inclui afinidades fortes, cauda longa de clientes e produtos, sazonalidade, tendência recente e ruído. As personas fixas são Lucas, Mariana, Rafael, Camila, Juliana e Bruno.

## Arquitetura

```text
CSV sintético -> bootstrap idempotente -> PostgreSQL
                                         |
                                         v
                                 trainer TensorFlow
                                         |
                                         v
                              artefato ecommerce-hybrid
                                         |
                                         v
React <- FastAPI <- ranking explicável / fallback de popularidade
```

Tabelas principais:

- `customers`: ID interno, ID público, nome, idade, faixa etária, estado e região;
- `articles`: produto, departamento, categoria, subcategoria, marca e faixa de preço;
- `purchases`: data, cliente, produto, preço BRL e canal;
- `dataset_state`: fingerprint importado;
- `training_runs`, `training_metrics`, `training_network_snapshots`: execução e telemetria;
- `model_versions`: artefatos treinados e ativação.

## Docker

Requisitos: Docker Compose v1.29 ou Compose v2.

```bash
cp .env.example .env
docker compose up --build
```

Também funciona com `docker-compose up --build`.

O serviço one-shot `bootstrap` aguarda PostgreSQL, valida os CSVs, importa apenas quando o fingerprint muda e enfileira exatamente um treino automático para esse fingerprint. `api` e `trainer` iniciam somente após sucesso do bootstrap. Segundo boot preserva banco, modelo e volume, sem nova importação ou novo treino.

Serviços:

- frontend: <http://localhost:5173>
- API: <http://localhost:8000>
- OpenAPI: <http://localhost:8000/docs>

## Treinamento

Perfis determinísticos:

- `fast`: 10.000 compras;
- `balanced`: 25.000 compras;
- `extensive`: 40.000 compras;
- `full`: todas as 47.500 compras.

Primeiro boot enfileira `full`, 4 epochs, batch 2.048. Enquanto treino não termina, API usa fallback de popularidade. Artefatos têm formato `ecommerce-hybrid` versão 2 e registram `source_fingerprint`.

Treino manual continua disponível:

```bash
curl -X POST http://localhost:8000/training/runs \
  -H 'Content-Type: application/json' \
  -d '{"data_profile":"fast","epochs":3,"batch_size":2048}'
```

## Ranking

`GET /customers/{profile_id}/recommendations` pontua todo produto elegível do catálogo. Com modelo ativo, score final usa escala 0 a 100:

```text
45 * model_raw + 30 * category_raw + 15 * popularity_raw + 10 * trend_raw
```

Pesos podem ser alterados por `model_weight`, `category_weight`, `popularity_weight` e `trend_weight`. Todos devem ser não negativos e somar 100. Resposta inclui `applied_weights`, grade (`baixo`, `medio`, `alto`, `muito_alto`) e breakdown factual dos quatro componentes.

Score serve somente para ordenação. Não representa probabilidade de compra nem efeito causal. Sem modelo, API informa motivo e usa popularidade observada com normalização logarítmica.

## API

- `GET /health`
- `GET /dataset/stats`
- `GET /training/profiles`
- `POST /training/runs`
- `GET /training/runs`
- `GET /training/runs/{id}`
- `GET /training/runs/{id}/events`
- `GET /customers/random`
- `GET /customers/{profile_id}`
- `GET /customers/{profile_id}/history`
- `GET /customers/{profile_id}/recommendations`
- `GET /recommendations/popular`

## Verificação

Execute da raiz:

```bash
PYTHONPATH=.:backend python -m pytest backend/tests -q
cd backend && python -m mypy app
docker compose config
```

## Estrutura

```text
backend/app/api/             contratos HTTP, ranking e SSE
backend/app/database/        schema e sessões SQLAlchemy
backend/app/dataset/         gerador, validação, manifesto e importação
backend/app/models/          features, mappings, modelo e artefatos
backend/app/recommendation/  inferência e popularidade
backend/app/training/        sampling, pipeline, métricas e worker
data/ecommerce_sintetico_br/ dataset versionado
data/processed/              caches e resultados derivados ignorados
artifacts/                   modelos locais ignorados
```
