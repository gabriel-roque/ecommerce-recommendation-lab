import { AlertIcon, RefreshIcon } from './Icons'

export function LoadingState({ label = 'Carregando dados reais...' }: { label?: string }) {
  return (
    <div className="panel-flat flex min-h-52 flex-col items-center justify-center gap-4 p-8 text-center" role="status">
      <span className="relative block size-11">
        <span className="absolute inset-0 animate-ping rounded-full bg-cobalt/20" />
        <span className="absolute inset-2 rounded-full border-[3px] border-cobalt border-t-transparent motion-safe:animate-spin" />
      </span>
      <p className="font-mono text-sm text-slate-600">{label}</p>
    </div>
  )
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="border-2 border-coral bg-[#fff1ed] p-6" role="alert">
      <div className="flex items-start gap-3">
        <AlertIcon className="mt-0.5 size-6 shrink-0 text-coral" />
        <div>
          <p className="font-bold">Não foi possível concluir</p>
          <p className="mt-1 max-w-2xl text-sm leading-6 text-slate-700">{message}</p>
          {onRetry && <button className="button-secondary mt-4 inline-flex items-center gap-2" onClick={onRetry}><RefreshIcon className="size-4" /> Tentar novamente</button>}
        </div>
      </div>
    </div>
  )
}

export function EmptyState({ title, description }: { title: string; description: string }) {
  return (
    <div className="border-2 border-dashed bg-paper/60 p-8 text-center">
      <p className="font-bold">{title}</p>
      <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-slate-600">{description}</p>
    </div>
  )
}
