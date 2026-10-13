# Backend

FastAPI, SQLAlchemy, and TensorFlow backend for E-commerce Recommendation Lab.

## Development

From the repository root:

```bash
PYTHONPATH=.:backend python -m pytest backend/tests -q
```

From `backend/`:

```bash
python -m mypy app
```

## Data and Bootstrap

`app.dataset.synthetic` generates and validates `data/ecommerce_sintetico_br`. No external download or legacy schema is required. `python -m app.cli bootstrap`:

1. validates checksums, counts, UTF-8 encoding, foreign keys, dates, BRL prices, and channels;
2. compares `source_fingerprint` with `dataset_state`;
3. imports canonical tables through PostgreSQL COPY only when required;
4. queues one automatic `full` run when the fingerprint has not been trained or queued.

## Recommendation Contract

The personalized route accepts `model_weight`, `category_weight`, `popularity_weight`, and `trend_weight`. Their sum must equal 100. Defaults are 45, 30, 15, and 10.

Each item returns:

- `score`: 0 to 100;
- `grade`: `baixo`, `medio`, `alto`, or `muito_alto`;
- `breakdown`: `raw_score`, `weight`, `contribution`, `label`, and a factual explanation;
- `evidence`: observed evidence preserved for current consumers;
- `article`: generic product fields.

With an active model, the API scores every unseen catalog product. Without a model, the response uses popularity and reports the weights effectively applied. Scores are neither probabilities nor causal inferences.

## Artifacts

Accepted format: `ecommerce-hybrid`, version 2, dataset `ecommerce_sintetico_br`. Loading requires a `source_fingerprint` that matches the current manifest.
