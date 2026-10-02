import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { Product, Purchase, Recommendation } from '../types/api'
import { RecommendationConnectionVisualizer } from './RecommendationConnectionVisualizer'

const article: Product = { article_id: 'PROD-001', product_name: 'Cafeteira compacta', department_name: 'Casa', category_name: 'Cozinha', subcategory_name: 'Cafeteiras', brand: 'Aurora', price_band: 'R$ 100 a R$ 200', detail_desc: 'Cafeteira para uso doméstico.' }
const purchases: Purchase[] = [{ purchased_at: '2026-01-01T00:00:00Z', article_id: article.article_id, price: 149.9, sales_channel_id: 2, article }]
const recommendations: Recommendation[] = [{ rank: 1, article_id: 'PROD-002', score: 78.9, grade: 'alto', article: { ...article, article_id: 'PROD-002' }, evidence: [], breakdown: {} }]

describe('RecommendationConnectionVisualizer', () => {
  it('states that hybrid map is factual and non-causal', () => {
    render(<RecommendationConnectionVisualizer profileId="abc" customerName="Lucas Almeida" purchases={purchases} recommendations={recommendations} strategy="model" />)
    expect(screen.getByRole('img', { name: /relações visuais entre compras/i })).toHaveAccessibleDescription(/não indicam causalidade/i)
    expect(screen.getByText(/não explica recomendação individual/i)).toBeInTheDocument()
  })
  it('shows no customer purchase relation for popularity fallback', () => {
    render(<RecommendationConnectionVisualizer profileId="abc" customerName="Lucas Almeida" purchases={purchases} recommendations={recommendations} strategy="popularity" />)
    expect(screen.getByRole('img', { name: /relações visuais entre compras/i })).toHaveAccessibleDescription(/Nenhuma relação personalizada/i)
    expect(screen.queryByText(/Compra observada do produto/i)).not.toBeInTheDocument()
  })
})
