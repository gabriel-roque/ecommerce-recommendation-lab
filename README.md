# H&M Hybrid Recommendation Lab

Laboratório didático de recomendação de moda baseado exclusivamente no dataset **H&M Personalized Fashion Recommendations**. O projeto cobre validação e importação dos dados, treinamento de um modelo híbrido em TensorFlow, inferência com fallback de popularidade, API FastAPI e interface React.

## O que o projeto demonstra

- compras como único sinal de interação observado;
- embeddings colaborativos de perfis e artigos;
- faixa etária como contexto do perfil;
- metadados do catálogo como contexto do artigo;
- split cronológico para treino, validação e teste;
- métricas de ranking `Recall@10`, `HitRate@10` e `NDCG@10`;
- acompanhamento de treino por Server-Sent Events (SSE);
- recomendações com evidências factuais e fallback explícito;
- IDs públicos pseudonimizados e ausência de idade exata na API.

Scores representam relevância relativa para ordenação. Eles não são probabilidades calibradas nem estimativas causais.

## Dataset

Baixe os arquivos na página da competição [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data) e coloque estes três CSVs em `data/raw/`:

```text
data/raw/
├── articles.csv
├── customers.csv
└── transactions_train.csv
```

O pipeline preserva todos os campos oficiais desses arquivos. A camada de treino usa:

- `transactions_train.csv`: data, perfil, artigo, preço de origem e canal de venda;
- `customers.csv`: idade convertida para faixa etária e atributos categóricos do perfil;
- `articles.csv`: tipo, grupo, cor, departamento, seção, grupo de vestuário e demais metadados.

O dataset não declara moeda para `price`, não contém avaliações, estoque ou eventos de navegação. A interface identifica o valor somente como **preço de origem**.

## Arquitetura

```text
CSV H&M -> validação e manifesto -> PostgreSQL
                                      |
                                      v
                              trainer TensorFlow
                                      |
                                      v
                             artefato hm-hybrid
                                      |
                                      v
React <- FastAPI <- ranking híbrido / popularidade
```

Tabelas canônicas:

- `customers`: catálogo de perfis, ID público pseudônimo e faixa etária;
- `articles`: catálogo completo de artigos;
- `purchases`: compras observadas;
- `training_runs`, `training_metrics`, `training_network_snapshots`: telemetria de treino;
- `model_versions`: artefatos treinados e ativação.

O modelo combina embeddings de perfil e artigo com embeddings das features categóricas. Amostras negativas são geradas durante o treino. A inferência exclui artigos já comprados por padrão. Perfis fora do artefato ou falhas de carregamento usam popularidade de compras com motivo explícito.

## Execução com Docker

Requisitos:

- Docker com Compose;
- aproximadamente 10 GB livres para dados, banco, imagens e artefatos;
- os três CSVs H&M em `data/raw/`.

Copie as variáveis opcionais e inicie o banco:

```bash
cp .env.example .env
docker compose up -d postgres
```

Valide e importe o dataset completo:

```bash
docker compose run --rm api python -m app.cli import-data --full
docker compose run --rm api python -m app.cli optimize-db
docker compose run --rm api python -m app.cli validate-full --external-tests-passed
```

Inicie aplicação:

```bash
docker compose up --build -d api trainer frontend
```

Acesse:

- frontend: <http://localhost:5173>
- API: <http://localhost:8000>
- documentação OpenAPI: <http://localhost:8000/docs>

A importação completa processa `31.788.324` compras e pode levar vários minutos. O perfil de treino `full` pode levar horas, conforme CPU, memória e armazenamento disponíveis.

## Treinamento

A tela **Training Lab** oferece quatro escopos determinísticos:

- `fast`: 25.000 compras;
- `balanced`: 100.000 compras;
- `extensive`: 250.000 compras;
- `full`: todas as compras.

Também é possível iniciar um treino pela API:

```bash
curl -X POST http://localhost:8000/training/runs \
  -H 'Content-Type: application/json' \
  -d '{"data_profile":"fast","epochs":3,"embedding_dim":32,"dense_1":64,"dense_2":32,"batch_size":4096,"negative_ratio":2,"learning_rate":0.001}'
```

O trainer salva um artefato versionado em `artifacts/`, valida pesos e uma inferência real após recarregar o modelo e ativa a nova versão somente depois dessa validação.

## API principal

- `GET /health`: saúde da API e do banco;
- `GET /dataset/stats`: cobertura do catálogo, compras e faixas etárias;
- `GET /training/profiles`: escopos de treino;
- `POST /training/runs`: cria treino;
- `GET /training/runs`: lista treinos;
- `GET /training/runs/{id}`: estado, métricas e avaliação;
- `GET /training/runs/{id}/events`: stream SSE;
- `GET /customers/random?personalizable=true`: seleciona perfil público ativo;
- `GET /customers/{profile_id}`: resumo público do perfil;
- `GET /customers/{profile_id}/history`: compras paginadas;
- `GET /customers/{profile_id}/recommendations`: ranking híbrido ou fallback;
- `GET /recommendations/popular`: baseline global por compras.

## Privacidade e explicabilidade

- `customer_id` original nunca é exposto pela API;
- `profile_id` é derivado com hash e usado como identificador público;
- idade exata e código postal não são expostos;
- faixa etária é mostrada apenas como contexto agregado;
- evidências descrevem contagens observadas, afinidade de departamento e coorte etária;
- respostas distinguem ranking do modelo e fallback de popularidade.

## Testes

Backend:

```bash
python -m pytest backend/tests -q
python -m mypy backend/app
```

Frontend:

```bash
npm --prefix frontend test
npm --prefix frontend run build
```

Validação integral de arquivos, banco e importação:

```bash
docker compose run --rm api python -m app.cli validate-full
```

## Estrutura

```text
backend/app/api/             contratos HTTP e SSE
backend/app/database/        modelos e sessões SQLAlchemy
backend/app/dataset/         validação, manifesto e importação H&M
backend/app/models/          features, mappings, modelo e artefatos
backend/app/recommendation/  ranking e popularidade
backend/app/training/        sampling, pipeline, métricas e worker
frontend/src/                aplicação React
data/raw/                    CSVs H&M
data/processed/              manifestos, estatísticas e caches
artifacts/                   modelos H&M versionados
```
