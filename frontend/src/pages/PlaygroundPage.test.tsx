import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../services/api'
import { PlaygroundPage } from './PlaygroundPage'

vi.mock('../services/api', () => ({
  api: {
    getPopularRecommendations: vi.fn(), getProfile: vi.fn(), getProfilePurchases: vi.fn(),
    getRandomProfile: vi.fn(), getProfileRecommendations: vi.fn(),
  },
}))

const article = { article_id: '0706016001', product_name: 'Jersey top', product_type_name: 'Top', product_group_name: 'Garment Upper body', colour_group_name: 'Black', department_name: 'Jersey', section_name: 'Womens Everyday Basics', garment_group_name: 'Jersey Basic', detail_desc: 'Soft jersey top.' }
const popular = { rank: 1, article_id: article.article_id, relevance_score: .8, article, evidence: [] }

describe('PlaygroundPage', () => {
  beforeEach(() => {
    vi.mocked(api.getPopularRecommendations).mockResolvedValue({ profile_id: 'global', strategy: 'popular', fallback_reason: null, model_version: null, exclude_purchased_articles: false, inference_duration_ms: 1, recommendations: [popular] })
    vi.mocked(api.getProfile).mockResolvedValue({ profile_id: 'abc', age_band: '25-34', purchase_count: 10, first_purchase_at: '2020-01-01T00:00:00Z', last_purchase_at: '2020-02-01T00:00:00Z' })
    vi.mocked(api.getProfilePurchases).mockResolvedValue({ profile_id: 'abc', total: 10, limit: 8, offset: 0, purchases: [{ purchased_at: '2020-02-01T00:00:00Z', article_id: article.article_id, source_price: .0245, sales_channel_id: 2, article }] })
    vi.mocked(api.getProfileRecommendations).mockResolvedValue({ profile_id: 'abc', strategy: 'hybrid', fallback_reason: null, model_version: 'hm-1', exclude_purchased_articles: true, inference_duration_ms: 12, recommendations: [{ ...popular, article_id: '0812001001', article: { ...article, article_id: '0812001001', product_name: 'Cotton shirt' }, relevance_score: .9234, evidence: [{ reason: 'Mesmo departamento de compras anteriores', cohort_buyers: 24, related_history_articles: 2, same_department: true }] }] })
  })

  it('loads profile, paginated real purchases, and hybrid evidence', async () => {
    const actor = userEvent.setup()
    render(<PlaygroundPage />)
    await actor.type(screen.getByLabelText(/profile_id/i), 'abc')
    await actor.click(screen.getByRole('button', { name: /^buscar$/i }))
    expect(await screen.findByText(/preço de origem 0,0245/i)).toBeInTheDocument()
    expect(screen.queryByText(/[€$£]/)).not.toBeInTheDocument()
    expect(screen.getByText('25-34')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /próxima/i })).toBeInTheDocument()
    await actor.click(screen.getByRole('button', { name: /gerar ranking híbrido/i }))
    expect(await screen.findByText('Cotton shirt')).toBeInTheDocument()
    expect(screen.getAllByText(/relevância 0,9234/i).length).toBeGreaterThan(0)
    expect(screen.getByText(/24 compradores na coorte/i)).toBeInTheDocument()
    expect(api.getProfileRecommendations).toHaveBeenCalledWith('abc', 10, expect.any(AbortSignal))
  })

  it('shows backend fallback reason', async () => {
    vi.mocked(api.getProfileRecommendations).mockResolvedValue({ profile_id: 'abc', strategy: 'popular', fallback_reason: 'Modelo híbrido ainda não cobre este perfil.', model_version: null, exclude_purchased_articles: true, inference_duration_ms: 2, recommendations: [popular] })
    const actor = userEvent.setup()
    render(<PlaygroundPage />)
    await actor.type(screen.getByLabelText(/profile_id/i), 'abc')
    await actor.click(screen.getByRole('button', { name: /^buscar$/i }))
    await actor.click(await screen.findByRole('button', { name: /gerar ranking híbrido/i }))
    expect(await screen.findByText('Modelo híbrido ainda não cobre este perfil.')).toBeInTheDocument()
  })

  it('shows not-found error without creating a phantom profile', async () => {
    vi.mocked(api.getProfile).mockRejectedValue(new Error('Profile not found'))
    const actor = userEvent.setup()
    render(<PlaygroundPage />)
    await actor.type(screen.getByLabelText(/profile_id/i), 'missing')
    await actor.click(screen.getByRole('button', { name: /^buscar$/i }))
    expect(await screen.findByText('Profile not found')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /gerar ranking híbrido/i })).not.toBeInTheDocument()
    expect(api.getProfilePurchases).not.toHaveBeenCalledWith('missing', expect.anything(), expect.anything(), expect.anything())
  })
})
