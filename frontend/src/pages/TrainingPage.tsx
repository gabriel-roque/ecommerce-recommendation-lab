import { FormEvent, useEffect, useState } from 'react'
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { CheckIcon, CpuIcon, NetworkIcon } from '../components/Icons'
import { NeuralNetworkVisualizer } from '../components/NeuralNetworkVisualizer'
import { PageHeader } from '../components/PageHeader'
import { ErrorState } from '../components/States'
import { api, subscribeToTraining } from '../services/api'
import type { DatasetStats, TrainingConfig, TrainingEvent, TrainingProfile, TrainingRun, TrainingStatus } from '../types/api'
import { errorMessage, formatDecimal, formatDuration, formatInteger } from '../utils'

const initialConfig: TrainingConfig = {
  data_profile: 'balanced',
  embedding_dim: 32,
  dense_1: 64,
  dense_2: 32,
  batch_size: 4096,
  epochs: 3,
  negative_ratio: 2,
  learning_rate: 0.001,
}

const fallbackProfiles: TrainingProfile[] = [
  { id: 'fast', label: 'Rápido', target_purchases: 25_000, description: 'Iteração curta para validar configuração.' },
  { id: 'balanced', label: 'Equilibrado', target_purchases: 100_000, description: 'Boa relação entre sinal e tempo. Recomendado.' },
  { id: 'extensive', label: 'Amplo', target_purchases: 250_000, description: 'Mais cobertura para comparação detalhada.' },
  { id: 'full', label: 'Completo', target_purchases: null, description: 'Usa todas as compras da fonte e pode levar horas.' },
]

const statusLabels: Record<TrainingStatus, string> = {
  QUEUED: 'Na fila', PREPARING_DATA: 'Preparando dados', TRAINING: 'Treinando modelo', EVALUATING: 'Avaliando ranking', SAVING_MODEL: 'Validando artefato', COMPLETED: 'Concluído', FAILED: 'Falhou',
}

const pipeline: TrainingStatus[] = ['QUEUED', 'PREPARING_DATA', 'TRAINING', 'EVALUATING', 'SAVING_MODEL', 'COMPLETED']

export function TrainingPage() {
  const [config, setConfig] = useState(initialConfig)
  const [dataset, setDataset] = useState<DatasetStats | null>(null)
  const [datasetError, setDatasetError] = useState('')
  const [run, setRun] = useState<TrainingRun | null>(null)
  const [activeRunId, setActiveRunId] = useState<string | null>(null)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState('')
  const [streamWarning, setStreamWarning] = useState('')
  const [profiles, setProfiles] = useState(fallbackProfiles)

  useEffect(() => {
    const controller = new AbortController()
    api.getDatasetStats({ signal: controller.signal }).then(setDataset).catch((cause) => {
      if (!(cause instanceof DOMException && cause.name === 'AbortError')) setDatasetError(errorMessage(cause))
    })
    api.getTrainingProfiles(controller.signal).then((response) => {
      if (response.profiles.length) setProfiles(response.profiles.map((profile) => {
        const localized = fallbackProfiles.find((item) => item.id === profile.id)
        return localized ? { ...profile, label: localized.label, description: localized.description } : profile
      }))
      setConfig((current) => ({ ...current, data_profile: response.default_profile || 'balanced' }))
    }).catch(() => undefined)
    api.getTrainingRuns(20, controller.signal).then(({ items }) => {
      const active = items.find((item) => item.status !== 'COMPLETED' && item.status !== 'FAILED')
      const selected = active ?? items[0]
      if (selected) {
        setRun(selected)
      }
      if (active) {
        setActiveRunId(active.id)
      }
    }).catch(() => undefined)
    return () => controller.abort()
  }, [])

  useEffect(() => {
    if (!activeRunId) return
    const unsubscribe = subscribeToTraining(activeRunId, handleEvent, () => {
      setStreamWarning('Conexão ao stream interrompida. O navegador tentará reconectar automaticamente.')
    })
    return unsubscribe
  }, [activeRunId])

  const handleEvent = (event: TrainingEvent) => {
    setStreamWarning('')
    if (event.type === 'completed') {
      setRun(event.run)
      setActiveRunId(null)
      return
    }
    if (event.type === 'failed') {
      setRun((current) => current ? { ...current, status: 'FAILED', error_message: event.error_message } : current)
      setActiveRunId(null)
      return
    }
    setRun((current) => {
      if (!current) return current
      if (event.type === 'status_changed') return { ...current, status: event.status, current_epoch: event.current_epoch, elapsed_seconds: event.elapsed_seconds ?? current.elapsed_seconds, hardware: event.hardware ?? current.hardware, error_message: event.error_message ?? current.error_message }
      if (event.type === 'network_snapshot') return { ...current, network_snapshot: event.snapshot }
      const metric = { epoch: event.epoch, timestamp: event.timestamp, loss: event.loss, val_loss: event.val_loss, learning_rate: event.learning_rate, epoch_duration_seconds: event.epoch_duration_seconds, examples_per_second: event.examples_per_second }
      const metrics = [...current.metrics.filter((item) => item.epoch !== metric.epoch), metric].sort((a, b) => a.epoch - b.epoch)
      return { ...current, status: 'TRAINING', current_epoch: metric.epoch, metrics, network_snapshot: event.snapshot ?? current.network_snapshot, elapsed_seconds: event.elapsed_seconds ?? current.elapsed_seconds }
    })
  }

  const startTraining = async (event: FormEvent) => {
    event.preventDefault()
    setStarting(true)
    setError('')
    try {
      const created = await api.createTrainingRun(config)
      setRun({ ...created, config: created.config || config, metrics: created.metrics || [], current_epoch: created.current_epoch || 0 })
      setActiveRunId(created.id)
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setStarting(false)
    }
  }

  const reset = () => {
    setRun(null)
    setActiveRunId(null)
    setError('')
    setStreamWarning('')
  }

  return (
    <section>
      <PageHeader index="02" title="H&M Training Lab" description="Treine ranking híbrido com compras, embeddings de perfil e artigo, contexto etário e metadados do catálogo." aside={run ? <StatusBadge status={run.status} /> : undefined} />
      {!run ? <TrainingSetup config={config} setConfig={setConfig} profiles={profiles} dataset={dataset} datasetError={datasetError} starting={starting} error={error} onSubmit={startTraining} /> : <TrainingDashboard run={run} streamWarning={streamWarning} onReset={reset} />}
    </section>
  )
}

function TrainingSetup({ config, setConfig, profiles, dataset, datasetError, starting, error, onSubmit }: { config: TrainingConfig; setConfig: (config: TrainingConfig) => void; profiles: TrainingProfile[]; dataset: DatasetStats | null; datasetError: string; starting: boolean; error: string; onSubmit: (event: FormEvent) => void }) {
  const numberField = (key: Exclude<keyof TrainingConfig, 'data_profile'>, value: number) => setConfig({ ...config, [key]: value })
  const selectedProfile = profiles.find((profile) => profile.id === config.data_profile) ?? fallbackProfiles[1]

  return (
    <form onSubmit={onSubmit} className="grid gap-6 xl:grid-cols-[1fr_360px]">
      <div className="panel p-5 sm:p-7">
        <fieldset className="mb-7 border-b-2 pb-7"><legend className="font-bold">Escopo do treino</legend><p className="mt-1 max-w-3xl text-sm leading-6 text-slate-600">Perfil de volume define quantas compras H&M entram no treino. Fonte original permanece intacta.</p><div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {profiles.map((profile) => <label key={profile.id} className={`relative min-h-[148px] cursor-pointer border-2 p-4 transition-colors ${config.data_profile === profile.id ? 'bg-acid shadow-offset-sm' : 'bg-white hover:bg-mist'}`}><input aria-label={`${profile.label}, ${profile.target_purchases == null ? 'fonte completa' : `${formatInteger(profile.target_purchases)} compras`}`} className="absolute right-3 top-3 size-5 accent-cobalt" type="radio" name="data-profile" value={profile.id} checked={config.data_profile === profile.id} onChange={() => setConfig({ ...config, data_profile: profile.id })} /><span className="block pr-7 font-extrabold">{profile.label}</span><span className="mt-1 block font-mono text-xs text-cobalt">{profile.target_purchases == null ? 'Fonte completa' : `${formatInteger(profile.target_purchases)} compras`}</span><span className="mt-3 block text-xs leading-5 text-slate-600">{profile.description}</span></label>)}
        </div>{config.data_profile === 'full' && <p className="mt-4 border-2 border-coral bg-[#fff1ed] p-3 text-sm font-bold">Perfil Completo pode levar horas, conforme hardware disponível.</p>}</fieldset>
        <div className="flex items-start gap-4 border-b-2 pb-5"><span className="grid size-11 shrink-0 place-items-center border-2 bg-cobalt text-white"><NetworkIcon className="size-6" /></span><div><p className="eyebrow">Arquitetura híbrida</p><h2 className="mt-1 text-xl font-extrabold">Compras + idade + catálogo</h2><p className="mt-1 text-sm text-slate-600">Embeddings de perfil e artigo combinam padrões colaborativos, faixa etária e metadados de produto em score de relevância.</p></div></div>
        <fieldset className="mt-6"><legend className="font-bold">Hiperparâmetros</legend><div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <NumberInput id="embedding" label="Embedding dimension" value={config.embedding_dim} min={4} max={256} onChange={(value) => numberField('embedding_dim', value)} />
          <NumberInput id="batch" label="Batch size" value={config.batch_size} min={1} onChange={(value) => numberField('batch_size', value)} />
          <NumberInput id="epochs" label="Epochs" value={config.epochs} min={1} max={1000} onChange={(value) => numberField('epochs', value)} />
          <NumberInput id="negative" label="Negative ratio" value={config.negative_ratio} min={1} max={20} onChange={(value) => numberField('negative_ratio', value)} />
          <NumberInput id="learning" label="Learning rate" value={config.learning_rate} min={0.000001} max={1} step={0.000001} onChange={(value) => numberField('learning_rate', value)} />
          <div className="grid grid-cols-2 gap-2"><NumberInput id="dense1" label="Dense 1" value={config.dense_1} min={1} onChange={(value) => numberField('dense_1', value)} /><NumberInput id="dense2" label="Dense 2" value={config.dense_2} min={1} onChange={(value) => numberField('dense_2', value)} /></div>
        </div></fieldset>
        <div className="mt-7 border-t-2 bg-mist/60 p-4"><p className="font-bold">Sinal único: compra</p><p className="mt-1 text-sm leading-6 text-slate-600">Amostras negativas ensinam ordenação. Faixa etária e catálogo enriquecem ranking sem inventar sinais de navegação.</p></div>
      </div>

      <aside className="space-y-5">
        <div className="panel bg-ink p-6 text-white"><p className="font-mono text-[11px] uppercase tracking-widest text-acid">Preflight</p><h2 className="mt-2 text-xl font-extrabold">Antes da partida</h2>
          <ul className="mt-5 space-y-3 text-sm">
            <CheckLine ok={Boolean(dataset)} label={dataset ? `${formatInteger(dataset.purchase_count)} compras disponíveis` : 'Dataset disponível'} />
            <CheckLine ok={dataset ? !dataset.sampled : false} label="Fonte completa · treino configurável" />
            <CheckLine ok={true} label={`${selectedProfile.label}: ${selectedProfile.target_purchases == null ? 'todas as compras' : `${formatInteger(selectedProfile.target_purchases)} compras`}`} />
            <CheckLine ok={Boolean(dataset)} label="Split temporal 80 / 10 / 10" />
            <CheckLine ok={true} label="Negativos gerados em tf.data" />
          </ul>
          {datasetError && <p className="mt-4 border border-coral bg-coral/10 p-3 text-xs leading-5 text-white">{datasetError}</p>}
          <div className="mt-6 border-t border-white/30 pt-5"><p className="text-xs leading-5 text-mist">Um único modelo treina com 80% inicial da sequência temporal. Validação (10%) e teste (10%) medem o mesmo modelo, sem segundo treino.</p></div>
        </div>
        {error && <ErrorState message={error} />}
        <button type="submit" disabled={starting || !dataset} className="button-primary flex w-full items-center justify-center gap-3 bg-coral text-ink"><CpuIcon className="size-5" />{starting ? 'Enviando ao trainer...' : 'Train model'}</button>
      </aside>
    </form>
  )
}

function TrainingDashboard({ run, streamWarning, onReset }: { run: TrainingRun; streamWarning: string; onReset: () => void }) {
  const lastMetric = run.metrics.at(-1)
  const epochTotal = run.config.epochs
  const terminal = run.status === 'COMPLETED' || run.status === 'FAILED'
  const profile = fallbackProfiles.find((item) => item.id === (run.config.data_profile || 'full')) ?? fallbackProfiles[3]
  return (
    <div className="space-y-6" aria-live="polite">
      {streamWarning && <div className="border-2 border-[#c18b00] bg-[#fff5cf] p-3 text-sm" role="status">{streamWarning}</div>}
      {run.status === 'FAILED' && <ErrorState message={run.error_message || 'Trainer encerrou sem detalhes.'} />}
      <div className="panel p-5 sm:p-6">
        <div className="flex flex-col justify-between gap-4 md:flex-row md:items-center"><div><p className="eyebrow">Run {run.id}</p><h2 className="mt-1 text-2xl font-extrabold">{statusLabels[run.status]}</h2>{run.status === 'COMPLETED' && <p className="mt-2 font-mono text-xs uppercase text-slate-600">Perfil {profile.label} · {profile.target_purchases == null ? 'fonte completa' : `${formatInteger(profile.target_purchases)} compras`}</p>}</div>{terminal && <button className="button-secondary" onClick={onReset}>Configurar novo treino</button>}</div>
        <div className="mt-5 grid gap-2 sm:grid-cols-3 lg:grid-cols-6">
          {pipeline.map((status, index) => {
            const currentIndex = pipeline.indexOf(run.status)
            const passed = run.status === 'COMPLETED' || (currentIndex > index && run.status !== 'FAILED')
            const current = status === run.status
            return <div key={status} className={`border-2 p-2.5 ${current ? 'bg-acid' : passed ? 'bg-mist' : 'bg-white'}`}><p className="font-mono text-[9px] text-slate-500">{String(index + 1).padStart(2, '0')}</p><p className="mt-1 text-[11px] font-bold leading-4">{statusLabels[status]}</p></div>
          })}
        </div>
        {run.status === 'TRAINING' && <div className="mt-5"><div className="mb-2 flex justify-between font-mono text-xs"><span>Epoch {run.current_epoch} / {epochTotal}</span><span>{formatDecimal((run.current_epoch / epochTotal) * 100, 0)}%</span></div><div className="h-3 border-2 bg-white"><div className="h-full bg-cobalt transition-[width] duration-300" style={{ width: `${Math.min(100, run.current_epoch / epochTotal * 100)}%` }} /></div></div>}
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        <LabMetric label="Epoch" value={`${run.current_epoch} / ${epochTotal}`} />
        <LabMetric label="Loss" value={lastMetric ? formatDecimal(lastMetric.loss, 4) : '—'} />
        <LabMetric label="Validation loss" value={lastMetric?.val_loss != null ? formatDecimal(lastMetric.val_loss, 4) : '—'} />
        <LabMetric label="Examples / sec" value={lastMetric?.examples_per_second ? formatInteger(lastMetric.examples_per_second) : '—'} />
        <LabMetric label="Elapsed time" value={formatDuration(run.elapsed_seconds)} />
      </div>

      <article className="panel p-5 sm:p-6"><div className="mb-5"><p className="eyebrow">Live probe</p><h2 className="mt-1 text-xl font-extrabold">Rede neural por dentro</h2></div><NeuralNetworkVisualizer snapshot={run.network_snapshot ?? undefined} denseSizes={[run.config.dense_1, run.config.dense_2]} /></article>

      <div className="grid gap-6 xl:grid-cols-[1.5fr_.5fr]">
        <figure className="panel min-w-0 p-5 sm:p-6"><figcaption><p className="eyebrow">Curva de aprendizado</p><h2 className="mt-1 text-xl font-extrabold">Loss por epoch</h2></figcaption>
          {run.metrics.length ? <><div className="mt-5 h-72" role="img" aria-label="Gráfico de loss de treino e validação"><ResponsiveContainer width="100%" height="100%"><LineChart data={run.metrics} margin={{ left: -10, right: 12 }}><CartesianGrid strokeDasharray="4 4" stroke="#e8e8e5" /><XAxis dataKey="epoch" allowDecimals={false} /><YAxis domain={['auto', 'auto']} /><Tooltip formatter={(value) => formatDecimal(Number(value), 5)} contentStyle={{ border: '2px solid #111', borderRadius: 0 }} /><Legend /><Line type="monotone" dataKey="loss" name="Training loss" stroke="#e50010" strokeWidth={3} dot={{ r: 4, fill: '#e50010', stroke: '#111' }} isAnimationActive={false} /><Line type="monotone" dataKey="val_loss" name="Validation loss" stroke="#777" strokeWidth={3} dot={{ r: 4, fill: '#777', stroke: '#111' }} isAnimationActive={false} /></LineChart></ResponsiveContainer></div><table className="sr-only"><caption>Loss por epoch</caption><thead><tr><th>Epoch</th><th>Training loss</th><th>Validation loss</th></tr></thead><tbody>{run.metrics.map((metric) => <tr key={metric.epoch}><td>{metric.epoch}</td><td>{metric.loss}</td><td>{metric.val_loss}</td></tr>)}</tbody></table></> : <div className="mt-5 grid h-72 place-items-center border-2 border-dashed text-center text-sm text-slate-600">Curva aparecerá após primeiro epoch.</div>}
        </figure>
        <aside className="panel bg-ink p-5 text-white sm:p-6"><p className="font-mono text-[11px] uppercase tracking-widest text-acid">Runtime</p><div className="mt-5 space-y-5"><RuntimeFact label="Device" value={run.hardware?.device || 'Aguardando'} /><RuntimeFact label="RAM disponível" value={run.hardware?.ram_gb ? `${formatDecimal(run.hardware.ram_gb, 1)} GB` : '—'} /><RuntimeFact label="TensorFlow" value={run.hardware?.tensorflow_version || '—'} /><RuntimeFact label="Python" value={run.hardware?.python_version || '—'} /><RuntimeFact label="Epoch duration" value={formatDuration(lastMetric?.epoch_duration_seconds)} /></div></aside>
      </div>

      {run.status === 'COMPLETED' && run.evaluation && <article className="border-2 bg-acid p-6"><p className="eyebrow">Benchmark concluído</p><h2 className="mt-1 text-2xl font-extrabold">Métricas de ranking</h2><div className="mt-5 grid gap-4 sm:grid-cols-3"><LabMetric label="Recall@10" value={formatDecimal(run.evaluation.recall_at_10, 4)} /><LabMetric label="HitRate@10" value={formatDecimal(run.evaluation.hit_rate_at_10, 4)} /><LabMetric label="NDCG@10" value={formatDecimal(run.evaluation.ndcg_at_10, 4)} /></div><p className="mt-4 font-mono text-xs">Candidate strategy: {run.evaluation.candidate_strategy}</p></article>}
    </div>
  )
}

function NumberInput({ id, label, value, min, max, step = 1, onChange }: { id: string; label: string; value: number; min?: number; max?: number; step?: number; onChange: (value: number) => void }) {
  return <label htmlFor={id} className="block"><span className="mb-1.5 block text-xs font-bold text-slate-700">{label}</span><input id={id} className="field font-mono text-sm" type="number" required min={min} max={max} step={step} value={value} onChange={(event) => onChange(event.target.valueAsNumber)} /></label>
}

function CheckLine({ ok, label }: { ok: boolean; label: string }) { return <li className="flex items-center gap-2"><span className={`grid size-5 shrink-0 place-items-center border ${ok ? 'bg-acid text-ink' : 'bg-white/10 text-white'}`}>{ok && <CheckIcon className="size-3" />}</span>{label}</li> }
function StatusBadge({ status }: { status: TrainingStatus }) { return <div className={`inline-flex items-center gap-2 border-2 px-4 py-3 font-mono text-xs font-medium uppercase tracking-wider ${status === 'FAILED' ? 'bg-coral' : status === 'COMPLETED' ? 'bg-acid' : 'bg-mist'}`}><span className={`status-dot ${status === 'FAILED' ? 'text-red-800' : 'text-cobalt'}`} />{statusLabels[status]}</div> }
function LabMetric({ label, value }: { label: string; value: string }) { return <div className="panel-flat bg-white p-4"><p className="font-mono text-[10px] uppercase tracking-widest text-slate-500">{label}</p><p className="mt-2 font-mono text-lg font-medium sm:text-xl">{value}</p></div> }
function RuntimeFact({ label, value }: { label: string; value: string }) { return <div className="border-b border-white/25 pb-3"><p className="font-mono text-[9px] uppercase tracking-widest text-mist">{label}</p><p className="mt-1 font-bold">{value}</p></div> }
