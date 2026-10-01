import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { NeuralNetworkVisualizer } from './NeuralNetworkVisualizer'
import type { NetworkSnapshot } from '../types/api'

const snapshot: NetworkSnapshot = {
  epoch: 1,
  timestamp: '2026-01-01T00:00:00Z',
  probe_profile_id: '1234',
  probe_article_id: '5678',
  profile_embedding: [0.1, -0.2, 0.3, 0.4, 0, 0.8],
  article_embedding: [0.5, 0.1, -0.1, 0.7, 0.2, 0.4],
  dense_64: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
  dense_32: [0.2, 0.4, 0.6, 0.8],
  output_score: 0.712,
  connections: [{ source_layer: 'dense_32', source_index: 0, target_layer: 'output', target_index: 0, weight: -0.42 }],
}

describe('NeuralNetworkVisualizer', () => {
  it('does not invent activations before receiving a snapshot', () => {
    render(<NeuralNetworkVisualizer />)
    expect(screen.getByText(/aguardando primeiro snapshot/i)).toBeInTheDocument()
    expect(screen.queryByRole('img')).not.toBeInTheDocument()
  })

  it('renders real snapshot metadata, accessible fallback, and animation control', async () => {
    const actor = userEvent.setup()
    render(<NeuralNetworkVisualizer snapshot={snapshot} />)
    expect(screen.getByText(/Probe profile #1234.*pseudônimo/i)).toBeInTheDocument()
    expect(screen.getByText('Probe article #5678')).toBeInTheDocument()
    expect(screen.getByRole('img', { name: /ativações e pesos reais/i })).toBeInTheDocument()
    expect(screen.getByText('0,712')).toBeInTheDocument()
    expect(screen.getByText(/valores reais, movimento ilustrativo/i)).toBeInTheDocument()
    expect(screen.getByText(/tabela acessível do snapshot/i)).toBeInTheDocument()
    await actor.click(screen.getByRole('button', { name: /pausar animação/i }))
    expect(screen.getByRole('button', { name: /reproduzir animação/i })).toHaveAttribute('aria-pressed', 'true')
  })
})
