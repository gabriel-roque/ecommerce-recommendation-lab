import { useEffect, useId, useLayoutEffect, useRef, useState } from 'react'
import { glossary, type GlossaryTerm } from '../glossary'
import { InfoIcon } from './Icons'

export function InfoTooltip({ term, label }: { term: GlossaryTerm; label?: string }) {
  const [open, setOpen] = useState(false)
  const [position, setPosition] = useState({ left: 0, top: 0, ready: false })
  const id = useId()
  const root = useRef<HTMLSpanElement>(null)
  const trigger = useRef<HTMLButtonElement>(null)
  const tooltip = useRef<HTMLSpanElement>(null)
  const entry = glossary[term]

  useLayoutEffect(() => {
    if (!open) return
    const updatePosition = () => {
      if (!trigger.current || !tooltip.current) return
      const anchor = trigger.current.getBoundingClientRect()
      const popup = tooltip.current.getBoundingClientRect()
      const edge = 16
      const gap = 8
      const centeredLeft = anchor.left + anchor.width / 2 - popup.width / 2
      const left = Math.min(Math.max(edge, centeredLeft), window.innerWidth - popup.width - edge)
      const below = anchor.bottom + gap
      const top = below + popup.height <= window.innerHeight - edge
        ? below
        : Math.max(edge, anchor.top - popup.height - gap)
      setPosition({ left, top, ready: true })
    }
    updatePosition()
    window.addEventListener('resize', updatePosition)
    window.addEventListener('scroll', updatePosition, true)
    return () => {
      window.removeEventListener('resize', updatePosition)
      window.removeEventListener('scroll', updatePosition, true)
    }
  }, [open])

  useEffect(() => {
    if (!open) return
    const close = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setOpen(false)
        trigger.current?.focus()
      }
    }
    const outside = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false)
    }
    document.addEventListener('keydown', close)
    document.addEventListener('pointerdown', outside)
    return () => {
      document.removeEventListener('keydown', close)
      document.removeEventListener('pointerdown', outside)
    }
  }, [open])

  return (
    <span ref={root} className="relative inline-flex align-middle" onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      <button
        ref={trigger}
        type="button"
        className="grid size-11 shrink-0 place-items-center text-cobalt transition-colors hover:bg-mist"
        aria-label={`Explicação: ${label ?? term}`}
        aria-describedby={open ? id : undefined}
        aria-expanded={open}
        onClick={() => setOpen(true)}
        onFocus={() => setOpen(true)}
        onBlur={(event) => { if (!root.current?.contains(event.relatedTarget)) setOpen(false) }}
      >
        <InfoIcon className="size-4" />
      </button>
      {open && <span ref={tooltip} id={id} role="tooltip" style={{ left: position.left, top: position.top, visibility: position.ready ? 'visible' : 'hidden' }} className="fixed z-50 w-80 max-w-[calc(100vw-2rem)] border-2 bg-ink p-4 text-left text-sm font-normal leading-5 text-white shadow-offset">
        <strong className="block text-acid">{label ?? term}</strong>
        <span className="mt-1 block">{entry.explanation}</span>
        {'example' in entry && <span className="mt-2 block text-mist">Exemplo: {entry.example}</span>}
      </span>}
    </span>
  )
}

export function TermHelp({ term, children }: { term: GlossaryTerm; children?: string }) {
  return <span className="inline-flex min-w-0 items-center gap-1"><span>{children ?? term}</span><InfoTooltip term={term} label={children} /></span>
}
