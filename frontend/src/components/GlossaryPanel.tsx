import { useEffect, useRef } from 'react'
import { glossary } from '../glossary'
import { CloseIcon } from './Icons'

export function GlossaryPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const closeButton = useRef<HTMLButtonElement>(null)
  const dialog = useRef<HTMLElement>(null)
  const previousFocus = useRef<HTMLElement | null>(null)

  useEffect(() => {
    if (!open) return
    previousFocus.current = document.activeElement as HTMLElement | null
    closeButton.current?.focus()
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
      if (event.key !== 'Tab') return
      const focusable = dialog.current?.querySelectorAll<HTMLElement>('button, a, input, select, textarea, [tabindex]:not([tabindex="-1"])')
      if (!focusable?.length) return
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus() }
      if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus() }
    }
    document.addEventListener('keydown', handleKey)
    return () => {
      document.removeEventListener('keydown', handleKey)
      previousFocus.current?.focus()
    }
  }, [open, onClose])

  if (!open) return null
  return <div className="fixed inset-0 z-50 grid grid-cols-[minmax(0,1fr)] items-end overflow-x-hidden bg-ink/70 p-0 sm:items-center sm:p-6" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <section ref={dialog} role="dialog" aria-modal="true" aria-labelledby="glossary-title" className="mx-auto flex max-h-[90dvh] min-w-0 w-full max-w-3xl flex-col overflow-hidden border-2 bg-paper shadow-offset">
      <header className="flex shrink-0 items-start justify-between gap-4 border-b-2 bg-paper p-5 sm:p-7">
        <div><p className="eyebrow">Referência rápida</p><h2 id="glossary-title" className="mt-1 text-2xl font-extrabold">Glossário do Lab</h2></div>
        <button ref={closeButton} type="button" className="grid size-11 shrink-0 place-items-center border-2 bg-white hover:bg-mist" onClick={onClose} aria-label="Fechar glossário"><CloseIcon className="size-5" /></button>
      </header>
      <div className="min-h-0 overflow-x-hidden overflow-y-auto overscroll-contain p-5 sm:p-7">
        <dl className="grid gap-3 sm:grid-cols-2">
          {Object.entries(glossary).map(([term, entry]) => <div key={term} className="border-2 bg-white p-4"><dt className="font-mono text-sm font-bold text-cobalt">{term}</dt><dd className="mt-2 text-sm leading-6 text-slate-700">{entry.explanation}</dd>{'example' in entry && <dd className="mt-2 text-xs leading-5 text-slate-600"><strong>Exemplo:</strong> {entry.example}</dd>}</div>)}
        </dl>
      </div>
    </section>
  </div>
}
