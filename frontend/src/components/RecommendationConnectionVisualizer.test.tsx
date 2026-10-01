import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { ArticleMetadata } from '../types/api'
import { RecommendationConnectionVisualizer } from './RecommendationConnectionVisualizer'

const article: ArticleMetadata = { article_id: '0706016001', product_name: 'Jersey top', product_type_name: 'Top', product_group_name: null, colour_group_name: 'Black', department_name: 'Jersey', section_name: null, garment_group_name: null, detail_desc: null }
const purchases = [{ purchased_at: '2026-01-01T00:00:00Z', article_id: article.article_id, source_price: .02, sales_channel_id: 2, article }]
const recommendations = [{ rank: 1, article_id: '0999999001', relevance_score: .92, article: { ...article, article_id: '0999999001' }, evidence: [] }]

describe('RecommendationConnectionVisualizer', () => {
  it('describes hybrid profile and purchase context', () => {
    render(<RecommendationConnectionVisualizer profileId="abc" purchases={purchases} recommendations={recommendations} strategy="hybrid" />)
    expect(screen.getByRole('img', { name: /conexões entre compras/i })).toHaveAccessibleDescription(/perfil pseudônimo abc/i)
    expect(screen.getByText(/representa profile_id pseudônimo/i)).toBeInTheDocument()
  })
  it('shows no profile purchase edge for popular fallback', () => {
    render(<RecommendationConnectionVisualizer profileId="abc" purchases={purchases} recommendations={recommendations} strategy="popular" />)
    expect(screen.getByRole('img', { name: /conexões entre compras/i })).toHaveAccessibleDescription(/Nenhuma conexão personalizada/i)
    expect(screen.queryByText(/Compra observada do artigo/i)).not.toBeInTheDocument()
  })
})
