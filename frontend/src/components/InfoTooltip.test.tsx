import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { GlossaryPanel } from './GlossaryPanel'
import { InfoTooltip } from './InfoTooltip'

describe('ajuda de termos técnicos', () => {
  it('opens by focus or touch and closes with Escape', async () => {
    const actor = userEvent.setup()
    render(<InfoTooltip term="Score" label="Pontuação" />)
    const button = screen.getByRole('button', { name: /explicação: pontuação/i })
    await actor.click(button)
    const tooltip = screen.getByRole('tooltip')
    expect(tooltip).toHaveTextContent(/não é probabilidade de compra/i)
    expect(tooltip).toHaveClass('fixed')
    expect(button).toHaveAttribute('aria-describedby', tooltip.id)
    await actor.keyboard('{Escape}')
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
    fireEvent.blur(button)
    fireEvent.focus(button)
    expect(screen.getByRole('tooltip')).toBeInTheDocument()
  })

  it('renders glossary as an accessible dialog', async () => {
    const actor = userEvent.setup()
    const close = vi.fn()
    render(<GlossaryPanel open onClose={close} />)
    expect(screen.getByRole('dialog', { name: 'Glossário do Lab' })).toHaveClass('overflow-hidden')
    expect(screen.getByText('HitRate@10')).toBeInTheDocument()
    expect(screen.getByText('Fallback')).toBeInTheDocument()
    await actor.keyboard('{Escape}')
    expect(close).toHaveBeenCalledOnce()
  })
})
