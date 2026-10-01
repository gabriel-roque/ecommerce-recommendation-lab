import type { ReactNode } from 'react'

export function PageHeader({ index, title, description, aside }: { index: string; title: string; description: string; aside?: ReactNode }) {
  return (
    <header className="mb-8 flex flex-col justify-between gap-5 border-b-2 pb-6 lg:flex-row lg:items-end">
      <div>
        <p className="eyebrow">Módulo {index} / 03</p>
        <h1 className="mt-2 text-3xl font-extrabold tracking-[-0.04em] sm:text-5xl">{title}</h1>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-600 sm:text-base">{description}</p>
      </div>
      {aside && <div className="shrink-0">{aside}</div>}
    </header>
  )
}
