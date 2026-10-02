export const glossary = {
  Epochs: {
    explanation: 'Número de passagens completas pelos dados usados no treino.',
    example: '3 épocas processam o conjunto de treino três vezes.',
  },
  'Batch size': {
    explanation: 'Quantidade de exemplos processados antes de cada atualização do modelo.',
    example: 'Batch size 4096 atualiza o modelo a cada 4.096 exemplos.',
  },
  Embedding: {
    explanation: 'Vetor numérico aprendido que representa cliente ou produto.',
  },
  'Learning rate': {
    explanation: 'Tamanho do ajuste aplicado aos parâmetros em cada atualização.',
  },
  'Negative sampling': {
    explanation: 'Criação de exemplos sem compra observada para ensinar o modelo a ordenar produtos.',
  },
  Loss: {
    explanation: 'Erro numérico que o treino tenta reduzir.',
  },
  'Validation loss': {
    explanation: 'Erro medido em dados separados, não usados para ajustar o modelo.',
  },
  Overfitting: {
    explanation: 'Quando o modelo aprende detalhes do treino e perde qualidade em dados novos.',
    example: 'Perda de treino cai enquanto perda de validação sobe.',
  },
  'HitRate@10': {
    explanation: 'Fração de clientes com ao menos uma compra futura entre os 10 primeiros resultados.',
  },
  'Recall@10': {
    explanation: 'Fração das compras futuras relevantes recuperada entre os 10 primeiros resultados.',
  },
  'NDCG@10': {
    explanation: 'Qualidade da ordem dos 10 primeiros resultados, com maior valor para acertos no topo.',
  },
  Score: {
    explanation: 'Força relativa usada para ordenar resultados. Não é probabilidade de compra.',
    example: '78,9 / 100 indica posição relativa no ranking calculado.',
  },
  Candidate: {
    explanation: 'Produto elegível para avaliação e possível entrada no ranking.',
  },
  Fallback: {
    explanation: 'Estratégia alternativa usada quando personalização não está disponível.',
    example: 'Ranking por popularidade quando não existe modelo ativo.',
  },
} as const

export type GlossaryTerm = keyof typeof glossary
