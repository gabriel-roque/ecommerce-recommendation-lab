import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../services/api'
import type { Product, Recommendation, RecommendationResult } from '../types/api'
import { PlaygroundPage } from './PlaygroundPage'

vi.mock('../services/api', () => ({
  api: {
    getPopularRecommendations: vi.fn(), getCustomer: vi.fn(), getCustomerPurchases: vi.fn(),
    getRandomCustomer: vi.fn(), getCustomerRecommendations: vi.fn(),
  },
}))

const weights = { model: 45, category: 30, popularity: 15, trend: 10 }
const product: Product = { article_id: 'PROD-001', product_name: 'Cafeteira compacta', department_name: 'Casa', category_name: 'Cozinha', subcategory_name: 'Cafeteiras', brand: 'Aurora', price_band: 'R$ 100 a R$ 200', detail_desc: 'Cafeteira para uso doméstico.' }
const popular: Recommendation = { rank: 1, article_id: product.article_id, score: 80, grade: 'muito_alto', article: product, evidence: [{ kind: 'global_popularity', description: '120 compras observadas.', related_article_ids: [] }], breakdown: { popularity: { raw_score: .8, weight: 100, contribution: 80, label: 'Popularidade', explanation: '120 compras observadas no catálogo.' } } }
const hybrid: Recommendation = { rank: 1, article_id: 'PROD-002', score: 78.9, grade: 'alto', article: { ...product, article_id: 'PROD-002', product_name: 'Liquidificador portátil' }, evidence: [{ kind: 'purchase_history_department', description: 'Histórico contém compras no departamento Casa.', related_article_ids: ['PROD-001'] }], breakdown: {
  model: { raw_score: .86, weight: 45, contribution: 38.7, label: 'Afinidade do modelo', explanation: 'Saída neural usada somente para ordenar produtos; não é probabilidade.' },
  category: { raw_score: .7, weight: 30, contribution: 21, label: 'Afinidade de categoria', explanation: '7 de 10 compras ocorreram em Casa ou Cozinha.' },
  popularity: { raw_score: .8, weight: 15, contribution: 12, label: 'Popularidade', explanation: '120 compras observadas no catálogo.' },
  trend: { raw_score: .72, weight: 10, contribution: 7.2, label: 'Tendência recente', explanation: '24 compras nos últimos 30 dias.' },
} }
const popularResult: RecommendationResult = { profile_id: '', strategy: 'popularity', fallback_reason: null, model_version: null, score_semantics: { name: 'normalized_purchase_popularity', description: 'Popularidade normalizada.', minimum: 0, maximum: 100 }, applied_weights: { model: 0, category: 0, popularity: 100, trend: 0 }, exclude_seen_articles: false, inference_duration_ms: 1, recommendations: [popular] }
const hybridResult: RecommendationResult = { profile_id: 'abc', strategy: 'model', fallback_reason: null, model_version: 'modelo-1', score_semantics: { name: 'hybrid_ranking_score', description: 'Pontuação híbrida relativa.', minimum: 0, maximum: 100 }, applied_weights: weights, exclude_seen_articles: true, inference_duration_ms: 12, recommendations: [hybrid] }
const customer = { profile_id: 'abc', name: 'Lucas', state: 'SP', region: 'Sudeste', age_band: '25-34', purchase_count: 10, first_purchase_at: '2020-01-01', last_purchase_at: '2020-02-01', personalizable: true }

describe('PlaygroundPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getPopularRecommendations).mockResolvedValue(popularResult)
    vi.mocked(api.getCustomer).mockResolvedValue(customer)
    vi.mocked(api.getCustomerPurchases).mockResolvedValue({ profile_id: 'abc', total: 10, limit: 8, offset: 0, purchases: [{ purchased_at: '2020-02-01T00:00:00Z', article_id: product.article_id, price: 149.9, sales_channel_id: 2, article: product }] })
    vi.mocked(api.getCustomerRecommendations).mockResolvedValue(hybridResult)
  })

  async function loadCustomer() {
    const actor = userEvent.setup()
    render(<PlaygroundPage />)
    await actor.type(screen.getByLabelText(/ID técnico/i), 'abc')
    await actor.click(screen.getByRole('button', { name: /^buscar$/i }))
    expect(await screen.findByText('Lucas')).toBeInTheDocument()
    return actor
  }

  it('shows Lucas, BRL price, score breakdown, and non-causal warning', async () => {
    const actor = await loadCustomer()
    expect(await screen.findByText(/R\$\s*149,90/)).toBeInTheDocument()
    expect(screen.getByText('Sudeste')).toBeInTheDocument()
    await actor.click(screen.getByRole('button', { name: /gerar ranking híbrido/i }))
    expect(await screen.findByText('Liquidificador portátil')).toBeInTheDocument()
    expect(screen.getByText('78,9 / 100')).toBeInTheDocument()
    expect(screen.getByText(/não representa probabilidade de compra nem causalidade/i)).toBeInTheDocument()
    await actor.click(screen.getAllByText(/Por que foi recomendado/i).at(-1)!)
    expect(screen.getByText('38,7 / 45')).toBeInTheDocument()
    expect(screen.getByText(/Histórico contém compras no departamento Casa/i)).toBeInTheDocument()
    expect(api.getCustomerRecommendations).toHaveBeenCalledWith('abc', 10, weights, expect.any(AbortSignal))
  })

  it('blocks invalid weight sum, restores defaults, and applies only on button click', async () => {
    const actor = await loadCustomer()
    await actor.click(screen.getByText(/Configuração avançada de pesos/i))
    fireEvent.change(screen.getByLabelText('Peso de Modelo'), { target: { value: '44' } })
    expect(screen.getByText('Soma: 99 / 100')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /aplicar pesos/i })).toBeDisabled()
    expect(api.getCustomerRecommendations).not.toHaveBeenCalled()
    await actor.click(screen.getByRole('button', { name: /restaurar padrões/i }))
    expect(screen.getByText('Soma: 100 / 100')).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('Peso de Modelo'), { target: { value: '44' } })
    fireEvent.change(screen.getByLabelText('Peso de Categoria'), { target: { value: '31' } })
    expect(api.getCustomerRecommendations).not.toHaveBeenCalled()
    await actor.click(screen.getByRole('button', { name: /aplicar pesos/i }))
    expect(api.getCustomerRecommendations).toHaveBeenCalledWith('abc', 10, { model: 44, category: 31, popularity: 15, trend: 10 }, expect.any(AbortSignal))
  })

  it('translates backend fallback code', async () => {
    vi.mocked(api.getCustomerRecommendations).mockResolvedValue({ ...popularResult, profile_id: 'abc', fallback_reason: 'no_active_model', exclude_seen_articles: true })
    const actor = await loadCustomer()
    await actor.click(screen.getByRole('button', { name: /gerar ranking híbrido/i }))
    expect(await screen.findByText(/Não há modelo ativo/i)).toBeInTheDocument()
  })

  it('loads a random customer covered by active model', async () => {
    vi.mocked(api.getRandomCustomer).mockResolvedValue({ ...customer, profile_id: 'cliente-modelo' })
    const actor = userEvent.setup()
    render(<PlaygroundPage />)
    await actor.click(screen.getByRole('button', { name: /cliente do modelo/i }))
    expect(await screen.findByText(/ID técnico cliente-modelo/i)).toBeInTheDocument()
    expect(api.getRandomCustomer).toHaveBeenCalledWith(expect.any(AbortSignal))
  })
})
