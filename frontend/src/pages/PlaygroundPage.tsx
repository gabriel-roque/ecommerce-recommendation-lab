import { FormEvent, useEffect, useRef, useState } from 'react'
import { RefreshIcon, SearchIcon, SparkIcon } from '../components/Icons'
import { InfoTooltip, TermHelp } from '../components/InfoTooltip'
import { PageHeader } from '../components/PageHeader'
import { RecommendationConnectionVisualizer } from '../components/RecommendationConnectionVisualizer'
import { EmptyState, ErrorState, LoadingState } from '../components/States'
import { api } from '../services/api'
import type { Customer, PurchaseHistory, Recommendation, RecommendationResult, RecommendationWeights } from '../types/api'
import { errorMessage, formatCurrency, formatDate, formatDecimal, formatInteger } from '../utils'

const PURCHASE_PAGE_SIZE = 8
const DEFAULT_WEIGHTS: RecommendationWeights = { model: 45, category: 30, popularity: 15, trend: 10 }
const WEIGHT_FIELDS: Array<{ key: keyof RecommendationWeights; label: string }> = [
  { key: 'model', label: 'Modelo' },
  { key: 'category', label: 'Categoria' },
  { key: 'popularity', label: 'Popularidade' },
  { key: 'trend', label: 'Tendência recente' },
]

export function PlaygroundPage() {
  const [query, setQuery] = useState('')
  const [customer, setCustomer] = useState<Customer | null>(null)
  const [history, setHistory] = useState<PurchaseHistory | null>(null)
  const [popular, setPopular] = useState<Recommendation[]>([])
  const [result, setResult] = useState<RecommendationResult | null>(null)
  const [draftWeights, setDraftWeights] = useState(DEFAULT_WEIGHTS)
  const [appliedWeights, setAppliedWeights] = useState(DEFAULT_WEIGHTS)
  const [customerLoading, setCustomerLoading] = useState(false)
  const [historyLoading, setHistoryLoading] = useState(false)
  const [recommendationLoading, setRecommendationLoading] = useState(false)
  const [popularLoading, setPopularLoading] = useState(true)
  const [error, setError] = useState('')
  const [historyError, setHistoryError] = useState('')
  const [recommendationError, setRecommendationError] = useState('')
  const [popularError, setPopularError] = useState('')
  const customerRequest = useRef<AbortController | null>(null)
  const historyRequest = useRef<AbortController | null>(null)
  const recommendationRequest = useRef<AbortController | null>(null)
  const popularRequest = useRef<AbortController | null>(null)

  const loadPopular = async () => {
    popularRequest.current?.abort()
    const controller = new AbortController()
    popularRequest.current = controller
    setPopularLoading(true)
    setPopularError('')
    try {
      const response = await api.getPopularRecommendations(10, controller.signal)
      if (popularRequest.current === controller) setPopular(response.recommendations)
    } catch (cause) {
      if (!(cause instanceof DOMException && cause.name === 'AbortError') && popularRequest.current === controller) setPopularError(errorMessage(cause))
    } finally {
      if (popularRequest.current === controller) setPopularLoading(false)
    }
  }

  useEffect(() => {
    void loadPopular()
    return () => {
      customerRequest.current?.abort()
      historyRequest.current?.abort()
      recommendationRequest.current?.abort()
      popularRequest.current?.abort()
    }
  }, [])

  const loadPurchases = async (profileId: string, offset: number) => {
    historyRequest.current?.abort()
    const controller = new AbortController()
    historyRequest.current = controller
    setHistoryLoading(true)
    setHistoryError('')
    try {
      const response = await api.getCustomerPurchases(profileId, PURCHASE_PAGE_SIZE, offset, controller.signal)
      if (historyRequest.current === controller) setHistory(response)
    } catch (cause) {
      if (!(cause instanceof DOMException && cause.name === 'AbortError') && historyRequest.current === controller) setHistoryError(errorMessage(cause))
    } finally {
      if (historyRequest.current === controller) setHistoryLoading(false)
    }
  }

  const selectCustomer = async (profileId: string) => {
    customerRequest.current?.abort()
    historyRequest.current?.abort()
    recommendationRequest.current?.abort()
    const controller = new AbortController()
    customerRequest.current = controller
    setCustomerLoading(true)
    setError('')
    setHistory(null)
    setResult(null)
    setRecommendationError('')
    try {
      const response = await api.getCustomer(profileId, controller.signal)
      if (customerRequest.current !== controller) return
      setCustomer(response)
      setQuery(response.profile_id)
      await loadPurchases(response.profile_id, 0)
    } catch (cause) {
      if (!(cause instanceof DOMException && cause.name === 'AbortError') && customerRequest.current === controller) {
        setCustomer(null)
        setError(errorMessage(cause))
      }
    } finally {
      if (customerRequest.current === controller) setCustomerLoading(false)
    }
  }

  const search = (event: FormEvent) => {
    event.preventDefault()
    const profileId = query.trim()
    if (!profileId) {
      setError('Informe um ID técnico de cliente válido.')
      return
    }
    void selectCustomer(profileId)
  }

  const randomCustomer = async () => {
    customerRequest.current?.abort()
    historyRequest.current?.abort()
    recommendationRequest.current?.abort()
    const controller = new AbortController()
    customerRequest.current = controller
    setCustomerLoading(true)
    setError('')
    setHistory(null)
    setResult(null)
    try {
      const response = await api.getRandomCustomer(controller.signal)
      if (customerRequest.current !== controller) return
      setCustomer(response)
      setQuery(response.profile_id)
      await loadPurchases(response.profile_id, 0)
    } catch (cause) {
      if (!(cause instanceof DOMException && cause.name === 'AbortError') && customerRequest.current === controller) setError(errorMessage(cause))
    } finally {
      if (customerRequest.current === controller) setCustomerLoading(false)
    }
  }

  const generate = async (weights = appliedWeights) => {
    if (!customer) return
    recommendationRequest.current?.abort()
    const controller = new AbortController()
    recommendationRequest.current = controller
    setRecommendationLoading(true)
    setRecommendationError('')
    try {
      const response = await api.getCustomerRecommendations(customer.profile_id, 10, weights, controller.signal)
      if (recommendationRequest.current === controller) setResult(response)
    } catch (cause) {
      if (!(cause instanceof DOMException && cause.name === 'AbortError') && recommendationRequest.current === controller) setRecommendationError(errorMessage(cause))
    } finally {
      if (recommendationRequest.current === controller) setRecommendationLoading(false)
    }
  }

  const applyWeights = () => {
    if (weightTotal(draftWeights) !== 100) return
    setAppliedWeights(draftWeights)
    void generate(draftWeights)
  }

  return (
    <section>
      <PageHeader index="03" title="Laboratório de recomendações" description="Explore clientes sintéticos, compras observadas e rankings por popularidade e estratégia híbrida." aside={<span className="inline-flex items-center gap-2 border-2 bg-coral px-4 py-3 font-mono text-xs font-medium uppercase tracking-wider"><SparkIcon className="size-4" /> Ranking de produtos</span>} />

      <form onSubmit={search} className="panel mb-7 p-4 sm:p-5">
        <div className="grid gap-3 md:grid-cols-[1fr_auto_auto] md:items-end">
          <label htmlFor="customer-id"><span className="mb-2 block text-sm font-extrabold">Buscar cliente pelo ID técnico</span><div className="relative"><SearchIcon className="absolute left-3.5 top-1/2 size-5 -translate-y-1/2 text-slate-500" /><input id="customer-id" aria-label="Buscar cliente pelo ID técnico" className="field pl-11 font-mono" placeholder="Ex.: cliente_0001" value={query} onChange={(event) => setQuery(event.target.value)} /></div></label>
          <button className="button-primary flex items-center justify-center gap-2" disabled={customerLoading} type="submit"><SearchIcon className="size-4" /> Buscar</button>
          <button className="button-secondary flex items-center justify-center gap-2" disabled={customerLoading} type="button" onClick={() => void randomCustomer()}><RefreshIcon className="size-4" /> Cliente do modelo</button>
        </div>
      </form>

      {customerLoading && <LoadingState label="Carregando cliente e compras..." />}
      {!customerLoading && error && <ErrorState message={error} />}
      {!customerLoading && !error && !customer && <WelcomeState popular={popular} loading={popularLoading} error={popularError} reload={loadPopular} />}
      {!customerLoading && customer && <CustomerWorkspace customer={customer} history={history} historyLoading={historyLoading} historyError={historyError} loadPurchases={loadPurchases} popular={popular} result={result} recommendationLoading={recommendationLoading} recommendationError={recommendationError} generate={() => generate()} draftWeights={draftWeights} setDraftWeights={setDraftWeights} applyWeights={applyWeights} />}
    </section>
  )
}

function WelcomeState({ popular, loading, error, reload }: { popular: Recommendation[]; loading: boolean; error: string; reload: () => Promise<void> }) {
  return <div className="grid gap-6 xl:grid-cols-[.8fr_1.2fr]">
    <div className="panel grid min-h-[340px] place-items-center bg-ink p-8 text-center text-white"><div><span className="mx-auto grid size-16 place-items-center border-2 border-white bg-coral text-white shadow-[5px_5px_0_#fff]"><SearchIcon className="size-8" /></span><h2 className="mt-6 text-2xl font-extrabold">Qual cliente vamos explorar?</h2><p className="mx-auto mt-3 max-w-sm text-sm leading-6 text-mist">Busque um ID conhecido ou selecione um cliente sintético coberto pelo modelo ativo.</p></div></div>
    <article className="panel p-5 sm:p-7"><p className="eyebrow">Referência global</p><h2 className="mt-1 text-2xl font-extrabold">Produtos populares</h2><p className="mt-2 text-sm text-slate-600">Mesmo ranking para todos os clientes. Pontuação mede popularidade normalizada, não afinidade personalizada.</p>{loading && <div className="mt-5"><LoadingState label="Buscando ranking popular..." /></div>}{!loading && error && <div className="mt-5"><ErrorState message={error} onRetry={() => void reload()} /></div>}{!loading && !error && !popular.length && <div className="mt-5"><EmptyState title="Ranking vazio" description="Nenhum produto popular retornado." /></div>}{!loading && popular.length > 0 && <RecommendationList recommendations={popular} compact />}</article>
  </div>
}

interface WorkspaceProps {
  customer: Customer
  history: PurchaseHistory | null
  historyLoading: boolean
  historyError: string
  loadPurchases: (profileId: string, offset: number) => Promise<void>
  popular: Recommendation[]
  result: RecommendationResult | null
  recommendationLoading: boolean
  recommendationError: string
  generate: () => Promise<void>
  draftWeights: RecommendationWeights
  setDraftWeights: (weights: RecommendationWeights) => void
  applyWeights: () => void
}

function CustomerWorkspace({ customer, history, historyLoading, historyError, loadPurchases, popular, result, recommendationLoading, recommendationError, generate, draftWeights, setDraftWeights, applyWeights }: WorkspaceProps) {
  const pageStart = history ? history.offset + 1 : 0
  const pageEnd = history ? Math.min(history.offset + history.purchases.length, history.total) : 0
  return <div className="space-y-7">
    <div className="grid gap-5 xl:grid-cols-[360px_1fr]">
      <aside className="panel bg-ink p-6 text-white"><p className="font-mono text-[11px] uppercase tracking-widest text-acid">Cliente sintético</p><h2 className="mt-2 break-words text-2xl font-extrabold">{customer.name}</h2><p className="mt-1 break-all font-mono text-[10px] uppercase tracking-wider text-mist">ID técnico {customer.profile_id}</p><div className="mt-6 grid grid-cols-2 gap-3"><CustomerMetric label="Compras" value={customer.purchase_count} /><CustomerMetric label="Faixa etária" value={customer.age_band} /><CustomerMetric label="Estado" value={customer.state} /><CustomerMetric label="Região" value={customer.region} /><CustomerMetric label="Última compra" value={customer.last_purchase_at ? formatDate(customer.last_purchase_at) : 'n/d'} /></div><button className="button-primary mt-6 flex w-full items-center justify-center gap-2 bg-coral text-white" onClick={() => void generate()} disabled={recommendationLoading}><SparkIcon className="size-5" />{recommendationLoading ? 'Calculando ranking...' : 'Gerar ranking híbrido'}</button><p className="mt-3 text-center font-mono text-[9px] uppercase tracking-wider text-mist">Produtos comprados são excluídos</p></aside>

      <article className="panel p-5 sm:p-6"><div className="flex flex-col justify-between gap-2 sm:flex-row sm:items-end"><div><p className="eyebrow">Histórico observado</p><h2 className="mt-1 text-xl font-extrabold">Compras do cliente</h2></div>{history && <span className="font-mono text-xs text-slate-500">{pageStart}-{pageEnd} de {history.total}</span>}</div>
        {historyLoading && <div className="mt-5"><LoadingState label="Carregando página de compras..." /></div>}
        {!historyLoading && historyError && <div className="mt-5"><ErrorState message={historyError} onRetry={() => void loadPurchases(customer.profile_id, history?.offset ?? 0)} /></div>}
        {!historyLoading && !historyError && history && history.purchases.length > 0 && <ol className="mt-5 space-y-2">{history.purchases.map((purchase, index) => <li key={`${purchase.article_id}-${purchase.purchased_at}-${index}`} className="grid grid-cols-[minmax(0,1fr)_auto] gap-3 border-2 bg-white p-3"><div className="min-w-0"><p className="truncate font-bold">{purchase.article.product_name || `Produto ${purchase.article_id}`}</p><p className="truncate text-xs text-slate-600">{purchase.article.category_name} · {purchase.article.brand}</p><p className="mt-1 font-mono text-[10px] text-slate-500">ID {purchase.article_id} · {formatCurrency(purchase.price)}</p></div><time className="text-right font-mono text-[10px] text-slate-500" dateTime={purchase.purchased_at}>{formatDate(purchase.purchased_at)}</time></li>)}</ol>}
        {!historyLoading && !historyError && history && !history.purchases.length && <div className="mt-5"><EmptyState title="Sem compras nesta página" description="Nenhuma compra retornada para este intervalo." /></div>}
        {history && history.total > PURCHASE_PAGE_SIZE && <nav aria-label="Paginação do histórico" className="mt-5 flex items-center justify-between gap-3 border-t-2 pt-4"><button type="button" className="button-secondary" disabled={historyLoading || history.offset === 0} onClick={() => void loadPurchases(customer.profile_id, Math.max(0, history.offset - history.limit))}>Anterior</button><span className="font-mono text-xs">Página {Math.floor(history.offset / history.limit) + 1}</span><button type="button" className="button-secondary" disabled={historyLoading || history.offset + history.limit >= history.total} onClick={() => void loadPurchases(customer.profile_id, history.offset + history.limit)}>Próxima</button></nav>}
      </article>
    </div>

    <WeightControls weights={draftWeights} onChange={setDraftWeights} onApply={applyWeights} loading={recommendationLoading} />
    {recommendationError && <ErrorState message={recommendationError} onRetry={() => void generate()} />}
    {recommendationLoading && <LoadingState label="Combinando modelo, categoria, popularidade e tendência..." />}
    {!recommendationLoading && !recommendationError && !result && <div className="border-2 border-dashed bg-mist/60 p-6 text-center text-sm text-slate-600">Gere o ranking para comparar popularidade global e estratégia híbrida. As escalas têm semânticas diferentes.</div>}
    {!recommendationLoading && result && <Comparison result={result} popular={popular} history={history} customerName={customer.name} />}
  </div>
}

function WeightControls({ weights, onChange, onApply, loading }: { weights: RecommendationWeights; onChange: (weights: RecommendationWeights) => void; onApply: () => void; loading: boolean }) {
  const total = weightTotal(weights)
  const valid = total === 100
  return <details className="panel p-5 sm:p-6">
    <summary className="min-h-11 cursor-pointer py-2 font-extrabold">Configuração avançada de pesos</summary>
    <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">Pesos controlam a contribuição máxima de cada componente. Alterações só recalculam o ranking ao selecionar “Aplicar pesos”.</p>
    <div className="mt-5 grid gap-5 md:grid-cols-2">
      {WEIGHT_FIELDS.map(({ key, label }) => <label key={key} htmlFor={`weight-${key}`} className="block"><span className="mb-2 flex items-center justify-between gap-3 text-sm font-bold"><span>{label}</span><output htmlFor={`weight-${key}`} className="font-mono">{formatDecimal(weights[key], 0)}</output></span><input id={`weight-${key}`} aria-label={`Peso de ${label}`} className="h-11 w-full accent-cobalt" type="range" min="0" max="100" step="1" value={weights[key]} onChange={(event) => onChange({ ...weights, [key]: event.target.valueAsNumber })} /></label>)}
    </div>
    <div className={`mt-5 flex flex-col justify-between gap-3 border-2 p-4 sm:flex-row sm:items-center ${valid ? 'bg-acid' : 'border-coral bg-[#fff1f2]'}`} role="status"><div><p className="font-bold">Soma: {formatDecimal(total, 0)} / 100</p>{!valid && <p className="mt-1 text-sm">A soma dos quatro pesos deve ser 100. Requisição bloqueada.</p>}</div><div className="flex flex-wrap gap-2"><button type="button" className="button-secondary" onClick={() => onChange(DEFAULT_WEIGHTS)}>Restaurar padrões</button><button type="button" className="button-primary" disabled={!valid || loading} onClick={onApply}>Aplicar pesos</button></div></div>
  </details>
}

function Comparison({ result, popular, history, customerName }: { result: RecommendationResult; popular: Recommendation[]; history: PurchaseHistory | null; customerName: string }) {
  return <section aria-live="polite">
    <div className="mb-4 flex flex-col justify-between gap-3 border-b-2 pb-4 sm:flex-row sm:items-end"><div><p className="eyebrow">Top 10</p><h2 className="mt-1 text-2xl font-extrabold">Popularidade e ranking híbrido</h2></div><div className="flex flex-wrap gap-2 font-mono text-[10px] uppercase"><span className="border bg-white px-2 py-1">{formatDecimal(result.inference_duration_ms, 1)} ms</span>{result.model_version && <span className="border bg-white px-2 py-1">modelo {result.model_version}</span>}</div></div>
    <div className="mb-5 border-2 border-cobalt bg-white p-4" role="note"><p className="flex items-center gap-1 font-bold"><TermHelp term="Score">Pontuação</TermHelp></p><p className="mt-1 text-sm leading-6 text-slate-700">Pontuação é força relativa de ranking. Não representa probabilidade de compra nem causalidade. Popularidade e ranking híbrido usam composições diferentes; compare ordens, não valores como se tivessem a mesma semântica.</p></div>
    {result.strategy === 'popularity' && <div className="mb-5 border-2 border-coral bg-[#fff1f2] p-4" role="status"><p className="flex items-center gap-1 font-bold"><span>Estratégia alternativa por popularidade</span><InfoTooltip term="Fallback" /></p><p className="mt-1 text-sm text-slate-700">{fallbackLabel(result.fallback_reason)}</p></div>}
    <div className="grid gap-6 lg:grid-cols-2">
      <article className="panel-flat min-w-0 p-5"><p className="eyebrow">A · Popularidade</p><h3 className="mt-1 font-extrabold">Ranking global</h3><p className="mt-1 text-xs leading-5 text-slate-600">Popularidade observada normalizada, sem contexto do cliente.</p>{popular.length ? <RecommendationList recommendations={popular} /> : <div className="mt-5"><EmptyState title="Popularidade indisponível" description="API não retornou ranking global." /></div>}</article>
      <article className="panel min-w-0 bg-white p-5"><p className="eyebrow">B · {result.strategy === 'model' ? 'Híbrido' : 'Alternativo'}</p><h3 className="mt-1 font-extrabold">{result.strategy === 'model' ? `Ranking para ${customerName}` : 'Popularidade global'}</h3><p className="mt-1 text-xs leading-5 text-slate-600">{result.score_semantics.description}</p>{result.recommendations.length ? <RecommendationList recommendations={result.recommendations} highlight /> : <div className="mt-5"><EmptyState title="Sem recomendações" description="API não retornou produtos candidatos." /></div>}</article>
    </div>
    {result.recommendations.length > 0 && <div className="mt-6"><RecommendationConnectionVisualizer profileId={result.profile_id} customerName={customerName} purchases={history?.purchases ?? []} recommendations={result.recommendations} strategy={result.strategy} /></div>}
  </section>
}

function RecommendationList({ recommendations, compact = false, highlight = false }: { recommendations: Recommendation[]; compact?: boolean; highlight?: boolean }) {
  return <ol className="mt-5 space-y-3">{recommendations.slice(0, 10).map((recommendation, index) => <li key={`${recommendation.article_id}-${index}`} className="border-2 bg-white">
    <div className="grid grid-cols-[38px_minmax(0,1fr)] gap-3 p-3"><span className={`grid size-8 place-items-center border-2 font-mono text-xs ${highlight && index < 3 ? 'bg-coral text-white' : 'bg-white'}`}>{recommendation.rank || index + 1}</span><div className="min-w-0"><div className="flex flex-col justify-between gap-2 sm:flex-row sm:items-start"><div className="min-w-0"><p className="break-words font-bold">{recommendation.article.product_name || `Produto ${recommendation.article_id}`}</p><p className="break-words text-xs text-slate-600">{recommendation.article.category_name || recommendation.article.department_name} · {recommendation.article.brand}</p></div><div className="shrink-0 sm:text-right"><p className="font-mono text-sm font-bold">{formatDecimal(recommendation.score, 1)} / 100</p><p className="text-xs text-slate-600">Grau: {gradeLabel(recommendation.grade)}</p></div></div>{!compact && <div className="mt-3 h-2 overflow-hidden border bg-mist" aria-label={`Pontuação ${formatDecimal(recommendation.score, 1)} de 100`}><div className="h-full bg-cobalt" style={{ width: `${Math.max(0, Math.min(100, recommendation.score))}%` }} /></div>}</div></div>
    <details className="border-t-2 px-3"><summary className="min-h-11 cursor-pointer py-3 text-sm font-bold">Por que foi recomendado?</summary><RecommendationExplanation recommendation={recommendation} /></details>
  </li>)}</ol>
}

function RecommendationExplanation({ recommendation }: { recommendation: Recommendation }) {
  const components = WEIGHT_FIELDS.map(({ key }) => recommendation.breakdown[key]).filter(Boolean)
  return <div className="pb-4">
    {components.length > 0 && <div className="grid gap-3 sm:grid-cols-2">{components.map((component) => component && <article key={component.label} className="border bg-paper p-3"><div className="flex items-start justify-between gap-2"><h4 className="text-sm font-bold">{component.label}</h4><span className="shrink-0 font-mono text-xs">{formatDecimal(component.contribution, 1)} / {formatDecimal(component.weight, 0)}</span></div><div className="mt-2 h-2 border bg-white"><div className="h-full bg-coral" style={{ width: `${Math.max(0, Math.min(100, component.raw_score * 100))}%` }} /></div><p className="mt-2 text-xs leading-5 text-slate-700">{component.explanation}</p></article>)}</div>}
    {recommendation.evidence.length > 0 && <div className="mt-3 border-l-4 border-cobalt pl-3"><p className="text-xs font-bold uppercase tracking-wide">Evidências factuais</p><ul className="mt-1 space-y-1 text-xs leading-5 text-slate-700">{recommendation.evidence.map((fact, index) => <li key={`${fact.kind}-${index}`}>{fact.description}</li>)}</ul></div>}
  </div>
}

function weightTotal(weights: RecommendationWeights) { return Object.values(weights).reduce((sum, weight) => sum + weight, 0) }
function gradeLabel(grade: Recommendation['grade']) { return ({ baixo: 'Baixo', medio: 'Médio', alto: 'Alto', muito_alto: 'Muito alto' })[grade] }
function fallbackLabel(reason: string | null) {
  if (reason === 'no_purchase_history') return 'Cliente sem histórico de compras; ranking usa popularidade global.'
  if (reason === 'no_active_model') return 'Não há modelo ativo; ranking usa popularidade global.'
  if (reason === 'profile_not_in_model') return 'Cliente não está coberto pelo modelo ativo; ranking usa popularidade global.'
  if (reason === 'model_unavailable') return 'Modelo indisponível; ranking usa popularidade global.'
  return reason || 'API não informou motivo adicional.'
}
function CustomerMetric({ label, value }: { label: string; value: number | string }) { return <div className="border border-white/30 p-3"><p className="font-mono text-[9px] uppercase tracking-wider text-mist">{label}</p><p className="mt-1 break-words text-base font-extrabold">{typeof value === 'number' ? formatInteger(value) : value}</p></div> }
