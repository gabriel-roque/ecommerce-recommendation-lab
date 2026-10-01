import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../services/api'
import type { DatasetStats } from '../types/api'
import { DatasetPage } from './DatasetPage'

vi.mock('../services/api', () => ({ api: { getDatasetStats: vi.fn() } }))

const stats: DatasetStats = {
  dataset_name: 'H&M Personalized Fashion Recommendations', purchase_count: 31_788_324, profile_count: 1_371_980, article_count: 105_542,
  average_purchases_per_profile: 23.2, median_purchases_per_profile: 9, average_purchases_per_article: 301.2,
  date_range: { start: '2018-09-20T00:00:00Z', end: '2020-09-22T00:00:00Z' }, sampled: false,
  purchases_per_profile: [{ bucket: '2-5', profiles: 12 }], top_articles: [{ article_id: '0706016001', purchases: 1200 }],
  age_coverage: { known: 1_356_119, missing: 15_861, known_ratio: .9884, groups: { '25-34': 300_000 } },
}

describe('DatasetPage', () => {
  beforeEach(() => vi.mocked(api.getDatasetStats).mockReset())
  it('shows H&M purchase, profile, article, and age statistics', async () => {
    vi.mocked(api.getDatasetStats).mockResolvedValue(stats)
    render(<DatasetPage />)
    expect(await screen.findByText('31.788.324')).toBeInTheDocument()
    expect(screen.getByText('Artigo 0706016001')).toBeInTheDocument()
    expect(screen.getByText('98,8%')).toBeInTheDocument()
    expect(screen.getByRole('img', { name: /perfis por faixa de idade/i })).toBeInTheDocument()
  })
  it('renders empty state when API has no statistics', async () => {
    vi.mocked(api.getDatasetStats).mockResolvedValue(null as unknown as DatasetStats)
    render(<DatasetPage />)
    expect(await screen.findByText('Estatísticas indisponíveis')).toBeInTheDocument()
  })
})
