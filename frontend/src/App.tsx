import { lazy, Suspense, useEffect, useState } from 'react'
import { DatabaseIcon, NetworkIcon, SparkIcon } from './components/Icons'
import { LoadingState } from './components/States'

const DatasetPage = lazy(() => import('./pages/DatasetPage').then((module) => ({ default: module.DatasetPage })))
const TrainingPage = lazy(() => import('./pages/TrainingPage').then((module) => ({ default: module.TrainingPage })))
const PlaygroundPage = lazy(() => import('./pages/PlaygroundPage').then((module) => ({ default: module.PlaygroundPage })))

type Area = 'dataset' | 'training' | 'playground'

const areas: Array<{ id: Area; label: string; shortLabel: string; icon: typeof DatabaseIcon }> = [
  { id: 'dataset', label: 'Dataset', shortLabel: 'Dados', icon: DatabaseIcon },
  { id: 'training', label: 'Training Lab', shortLabel: 'Treino', icon: NetworkIcon },
  { id: 'playground', label: 'Recommendation Playground', shortLabel: 'Recomendações', icon: SparkIcon },
]

function areaFromHash(): Area {
  const hash = window.location.hash.slice(1)
  return areas.some((area) => area.id === hash) ? hash as Area : 'dataset'
}

export default function App() {
  const [activeArea, setActiveArea] = useState<Area>(areaFromHash)

  useEffect(() => {
    const onHashChange = () => setActiveArea(areaFromHash())
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  const navigate = (area: Area) => {
    window.location.hash = area
    setActiveArea(area)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  return (
    <div className="paper-grid min-h-screen">
      <header className="border-b-2 bg-ink text-white">
        <div className="mx-auto flex max-w-[1500px] items-center justify-between gap-4 px-4 py-4 sm:px-6 lg:px-8">
          <button className="flex min-h-11 items-center gap-3 text-left" onClick={() => navigate('dataset')} aria-label="Ir para Dataset">
            <span className="grid size-10 place-items-center bg-coral font-serif text-lg font-black italic text-white shadow-[3px_3px_0_#fff]" aria-hidden="true">
              H&amp;M
            </span>
            <span>
              <span className="block text-sm font-extrabold uppercase tracking-wider sm:text-base">H&amp;M Recommendation Lab</span>
              <span className="hidden font-mono text-[10px] uppercase tracking-widest text-mist sm:block">Personalized fashion ranking</span>
            </span>
          </button>
          <div className="hidden items-center gap-2 border border-white/30 px-3 py-2 font-mono text-[10px] uppercase tracking-widest text-mist md:flex">
             <span className="status-dot text-acid" /> purchases · age · articles
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-[1500px] lg:grid-cols-[250px_1fr]">
        <aside className="border-b-2 bg-paper lg:min-h-[calc(100vh-74px)] lg:border-b-0 lg:border-r-2">
          <nav aria-label="Áreas principais" className="grid grid-cols-3 lg:sticky lg:top-0 lg:block lg:p-5">
            {areas.map((area, index) => {
              const Icon = area.icon
              const active = activeArea === area.id
              return (
                <button
                  key={area.id}
                  onClick={() => navigate(area.id)}
                  aria-current={active ? 'page' : undefined}
                  className={`group flex min-h-16 w-full items-center justify-center gap-2 border-r-2 px-2 py-3 text-center text-xs font-bold transition-colors last:border-r-0 lg:mb-3 lg:min-h-[72px] lg:justify-start lg:border-2 lg:px-4 lg:text-left lg:text-sm lg:last:border-r-2 ${active ? 'bg-acid shadow-none lg:shadow-offset-sm' : 'bg-cream hover:bg-mist'}`}
                >
                  <span className={`grid size-8 shrink-0 place-items-center border-2 lg:size-9 ${active ? 'bg-ink text-white' : 'bg-white text-cobalt'}`}><Icon className="size-4 lg:size-5" /></span>
                  <span><span className="hidden font-mono text-[9px] font-medium uppercase tracking-widest text-slate-500 lg:block">0{index + 1}</span><span className="lg:hidden">{area.shortLabel}</span><span className="hidden lg:block">{area.label}</span></span>
                </button>
              )
            })}
          </nav>
        </aside>

        <main id="main-content" className="min-w-0 px-4 py-7 sm:px-6 sm:py-10 lg:px-10 xl:px-14">
          <Suspense fallback={<LoadingState label="Abrindo módulo..." />}>
            {activeArea === 'dataset' && <DatasetPage />}
            {activeArea === 'training' && <TrainingPage />}
            {activeArea === 'playground' && <PlaygroundPage />}
          </Suspense>
        </main>
      </div>
    </div>
  )
}
