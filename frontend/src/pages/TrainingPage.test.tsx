import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../services/api'
import type { DatasetStats } from '../types/api'
import { TrainingPage } from './TrainingPage'

vi.mock('../services/api', () => ({
  api: { getDatasetStats: vi.fn(), getTrainingProfiles: vi.fn(), getTrainingRuns: vi.fn(), createTrainingRun: vi.fn() },
  subscribeToTraining: vi.fn(() => vi.fn()),
}))

const dataset: DatasetStats = { dataset_name: 'E-commerce Recommendation Lab', currency: 'BRL', purchase_count: 100, customer_count: 10, product_count: 10, average_purchases_per_customer: 10, median_purchases_per_customer: 5, average_purchases_per_product: 10, date_range: { start: null, end: null }, sampled: false, purchases_per_customer: [], top_products: [], age_coverage: { known: 8, missing: 2, known_ratio: .8, groups: {} } }

describe('TrainingPage', () => {
  beforeEach(() => {
    vi.mocked(api.getDatasetStats).mockResolvedValue(dataset)
    vi.mocked(api.getTrainingRuns).mockResolvedValue({ total: 0, items: [] })
    vi.mocked(api.getTrainingProfiles).mockRejectedValue(new Error('endpoint opcional indisponível'))
    vi.mocked(api.createTrainingRun).mockImplementation(() => new Promise(() => undefined))
  })
  it('sends purchase-only config and limits epochs to API contract', async () => {
    const actor = userEvent.setup()
    render(<TrainingPage />)
    expect(screen.getByText(/Compras \+ idade \+ catálogo/i)).toBeInTheDocument()
    expect(screen.getByRole('spinbutton', { name: 'Epochs' })).toHaveAttribute('max', '100')
    await waitFor(() => expect(screen.getByRole('button', { name: /treinar modelo/i })).toBeEnabled())
    await actor.click(screen.getByRole('button', { name: /treinar modelo/i }))
    expect(api.createTrainingRun).toHaveBeenCalledWith({
      data_profile: 'balanced', embedding_dim: 32, dense_1: 64, dense_2: 32,
      batch_size: 4096, epochs: 3, negative_ratio: 2, learning_rate: 0.001,
    })
  })
})
