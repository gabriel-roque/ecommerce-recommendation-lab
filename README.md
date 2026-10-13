# E-commerce Recommendation Lab

Hands-on recommendation lab for Brazilian e-commerce. The project uses only synthetic PT-BR data committed to the repository, with prices in BRL.

## Dataset

The default dataset is `ecommerce_sintetico_br`, stored in `data/ecommerce_sintetico_br/`:

- `customers.csv`: 2,000 synthetic Brazilian customers;
- `products.csv`: 500 generic products;
- `purchases.csv`: 47,500 purchases;
- `dataset_manifest.json`: checksums, fingerprint, counts, and source contract;
- `dataset_stats.json`: observed statistics.

The three CSV files contain exactly 50,000 records. The generator uses a fixed seed and produces the same bytes on every run:

```bash
PYTHONPATH=backend python -m app.cli generate-data
PYTHONPATH=backend python -m app.cli verify-data
```

The synthetic signal includes strong preferences, long-tail customer and product activity, seasonality, recent trends, and noise. Fixed personas include Lucas, Mariana, Rafael, Camila, Juliana, and Bruno.

## Architecture

```text
Synthetic CSV -> idempotent bootstrap -> PostgreSQL
                                             |
                                             v
                                    TensorFlow trainer
                                             |
                                             v
                                  ecommerce-hybrid artifact
                                             |
                                             v
React <- FastAPI <- explainable ranking / popularity fallback
```

Main tables:

- `customers`: internal ID, public ID, name, age, age band, state, and region;
- `articles`: product, department, category, subcategory, brand, and price band;
- `purchases`: date, customer, product, BRL price, and channel;
- `dataset_state`: imported source fingerprint;
- `training_runs`, `training_metrics`, `training_network_snapshots`: execution state and telemetry;
- `model_versions`: trained artifacts and activation state.

## Docker

Requirements: Docker Compose v1.29 or Compose v2.

```bash
cp .env.example .env
docker compose up --build
```

`docker-compose up --build` is also supported.

The one-shot `bootstrap` service waits for PostgreSQL, validates the CSV files, imports them only when the fingerprint changes, and queues exactly one automatic training run for that fingerprint. `api` and `trainer` start only after bootstrap succeeds. A second startup preserves the database, model, and volumes without importing or training again.

Services:

- frontend: <http://localhost:5173>
- API: <http://localhost:8000>
- OpenAPI: <http://localhost:8000/docs>

## Training

Deterministic data profiles:

- `fast`: 10,000 purchases;
- `balanced`: 25,000 purchases;
- `extensive`: 40,000 purchases;
- `full`: all 47,500 purchases.

The first startup queues the `full` profile with 4 epochs and a batch size of 2,048. Until training finishes, the API uses the popularity fallback. Artifacts use `ecommerce-hybrid` format version 2 and record the `source_fingerprint`.

Manual training remains available:

```bash
curl -X POST http://localhost:8000/training/runs \
  -H 'Content-Type: application/json' \
  -d '{"data_profile":"fast","epochs":3,"batch_size":2048}'
```

## Ranking

`GET /customers/{profile_id}/recommendations` scores every eligible product in the catalog. With an active model, the final score uses a 0-100 scale:

```text
45 * model_raw + 30 * category_raw + 15 * popularity_raw + 10 * trend_raw
```

Weights can be changed through `model_weight`, `category_weight`, `popularity_weight`, and `trend_weight`. All weights must be non-negative and total 100. The response includes `applied_weights`, a grade (`baixo`, `medio`, `alto`, or `muito_alto`), and a factual breakdown of all four components.

Scores are used only for ranking. They do not represent purchase probability or causal effects. Without an active model, the API reports the fallback reason and uses logarithmically normalized observed popularity.

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

## Verification

Run from the repository root:

```bash
PYTHONPATH=.:backend python -m pytest backend/tests -q
cd backend && python -m mypy app
docker compose config
```

## Project Structure

```text
backend/app/api/             HTTP contracts, ranking, and SSE
backend/app/database/        SQLAlchemy schema and sessions
backend/app/dataset/         generator, validation, manifest, and import
backend/app/models/          features, mappings, model, and artifacts
backend/app/recommendation/  inference and popularity
backend/app/training/        sampling, pipeline, metrics, and worker
data/ecommerce_sintetico_br/ versioned synthetic dataset
data/processed/              ignored caches and derived results
artifacts/                   ignored local models
```
