# H&M Recommendation Lab Frontend

Interface React para explorar dataset, treino e recomendações H&M. Perfis são pseudônimos. Scores representam relevância relativa, não probabilidade.

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
- `GET /profiles/random`
- `GET /profiles/{profile_id}`
- `GET /profiles/{profile_id}/purchases?limit=8&offset=0`
- `GET /profiles/{profile_id}/recommendations?k=10&exclude_purchased_articles=true`
- `GET /recommendations/popular?k=10`

Tipos de payload ficam em `src/types/api.ts`.
