import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../services/api'
import type { DatasetStats } from '../types/api'
import { DatasetPage } from './DatasetPage'

vi.mock('../services/api', () => ({ api: { getDatasetStats: vi.fn() } }))

const stats: DatasetStats = {
  dataset_name: 'E-commerce Recommendation Lab', currency: 'BRL', purchase_count: 31_788_324, customer_count: 1_371_980, product_count: 105_542,
  average_purchases_per_customer: 23.2, median_purchases_per_customer: 9, average_purchases_per_product: 301.2,
  date_range: { start: '2018-09-20', end: '2020-09-22' }, sampled: false,
  purchases_per_customer: [{ bucket: '2-5', customers: 12 }], top_products: [{ article_id: 'PROD-001', product_name: 'Notebook Aurora', purchases: 1200 }],
  age_coverage: { known: 1_356_119, missing: 15_861, known_ratio: .9884, groups: { '25-34': 300_000 } },
}

describe('DatasetPage', () => {
  beforeEach(() => vi.mocked(api.getDatasetStats).mockReset())
  it('shows synthetic commerce statistics with product terminology', async () => {
    vi.mocked(api.getDatasetStats).mockResolvedValue(stats)
    render(<DatasetPage />)
    expect(await screen.findByText('31.788.324')).toBeInTheDocument()
    expect(screen.getByText('Notebook Aurora')).toBeInTheDocument()
    expect(screen.getByText('PROD-001')).toBeInTheDocument()
    expect(screen.getByText('98,8%')).toBeInTheDocument()
    expect(screen.getByText(/valores monetários usam BRL/i)).toBeInTheDocument()
    expect(screen.getByRole('img', { name: /clientes por faixa de idade/i })).toBeInTheDocument()
  })
  it('renders empty state when API has no statistics', async () => {
    vi.mocked(api.getDatasetStats).mockResolvedValue(null as unknown as DatasetStats)
    render(<DatasetPage />)
    expect(await screen.findByText('Estatísticas indisponíveis')).toBeInTheDocument()
  })
})
