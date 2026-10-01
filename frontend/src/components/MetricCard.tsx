import type { ReactNode } from 'react'

export function MetricCard({ label, value, note, accent = 'cobalt', icon }: { label: string; value: string; note?: string; accent?: 'cobalt' | 'coral' | 'acid'; icon?: ReactNode }) {
  const colors = { cobalt: 'bg-cobalt text-white', coral: 'bg-coral text-ink', acid: 'bg-acid text-ink' }
  return (
    <article className="panel-flat relative overflow-hidden p-5">
      <span className={`absolute right-0 top-0 flex size-10 items-center justify-center border-b-2 border-l-2 ${colors[accent]}`}>{icon}</span>
      <p className="eyebrow pr-8">{label}</p>
      <p className="mt-3 font-mono text-2xl font-medium tracking-tight sm:text-3xl">{value}</p>
      {note && <p className="mt-2 text-xs leading-5 text-slate-600">{note}</p>}
    </article>
  )
}
