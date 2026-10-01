import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../services/api'
import { TrainingPage } from './TrainingPage'

vi.mock('../services/api', () => ({
  api: { getDatasetStats: vi.fn(), getTrainingProfiles: vi.fn(), getTrainingRuns: vi.fn(), createTrainingRun: vi.fn() },
  subscribeToTraining: vi.fn(() => vi.fn()),
}))

const dataset = { dataset_name: 'H&M', purchase_count: 100, profile_count: 10, article_count: 10, average_purchases_per_profile: 10, median_purchases_per_profile: 5, average_purchases_per_article: 10, date_range: { start: null, end: null }, sampled: false, purchases_per_profile: [], top_articles: [], age_coverage: { known: 8, missing: 2, known_ratio: .8, groups: {} } }

describe('TrainingPage', () => {
  beforeEach(() => {
    vi.mocked(api.getDatasetStats).mockResolvedValue(dataset)
    vi.mocked(api.getTrainingRuns).mockResolvedValue({ total: 0, items: [] })
    vi.mocked(api.getTrainingProfiles).mockRejectedValue(new Error('optional endpoint unavailable'))
    vi.mocked(api.createTrainingRun).mockImplementation(() => new Promise(() => undefined))
  })
  it('sends purchase-only config and explains hybrid age context', async () => {
    const actor = userEvent.setup()
    render(<TrainingPage />)
    expect(screen.getByText(/Compras \+ idade \+ catálogo/i)).toBeInTheDocument()
    expect(screen.queryByText(/pesos dos sinais/i)).not.toBeInTheDocument()
    await waitFor(() => expect(screen.getByRole('button', { name: /train model/i })).toBeEnabled())
    await actor.click(screen.getByRole('button', { name: /train model/i }))
    expect(api.createTrainingRun).toHaveBeenCalledWith({
      data_profile: 'balanced', embedding_dim: 32, dense_1: 64, dense_2: 32,
      batch_size: 4096, epochs: 3, negative_ratio: 2, learning_rate: 0.001,
    })
  })
})
