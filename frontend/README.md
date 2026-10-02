# E-commerce Recommendation Lab Frontend

Interface React em PT-BR para explorar conjunto de dados sintético, treino e recomendações de produtos. Valores monetários usam BRL. Pontuações representam força relativa de ranking, não probabilidade de compra nem causalidade.

## Configuração

```bash
cp .env.example .env
npm install
npm run dev
```

`VITE_API_URL` define origem da API. Valor padrão: `http://localhost:8000`.

## Scripts

```bash
npm run dev
npm test
npm run build
```

## Contratos usados

- `GET /dataset/stats`
- `GET /training/profiles`
- `POST /training/runs`
- `GET /training/runs`
- `GET /training/runs/{id}/events` via SSE
- `GET /customers/random`
- `GET /customers/{profile_id}`
- `GET /customers/{profile_id}/history?limit=8&offset=0`
- `GET /customers/{profile_id}/recommendations?k=10&exclude_seen_articles=true&model_weight=45&category_weight=30&popularity_weight=15&trend_weight=10`
- `GET /recommendations/popular?k=10`

Tipos e adaptadores ficam em `src/types/api.ts` e `src/services/api.ts`.
