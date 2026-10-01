import { FormEvent, useEffect, useRef, useState } from 'react'
import { profileAlias } from '../aliases'
import { ArrowIcon, RefreshIcon, SearchIcon, SparkIcon } from '../components/Icons'
import { PageHeader } from '../components/PageHeader'
import { RecommendationConnectionVisualizer } from '../components/RecommendationConnectionVisualizer'
import { EmptyState, ErrorState, LoadingState } from '../components/States'
import { api } from '../services/api'
import type { Profile, PurchaseHistory, Recommendation, RecommendationResult } from '../types/api'
import { errorMessage, formatDate, formatDecimal, formatInteger } from '../utils'

const PURCHASE_PAGE_SIZE = 8

export function PlaygroundPage() {
  const [query, setQuery] = useState('')
  const [profile, setProfile] = useState<Profile | null>(null)
  const [history, setHistory] = useState<PurchaseHistory | null>(null)
  const [popular, setPopular] = useState<Recommendation[]>([])
  const [result, setResult] = useState<RecommendationResult | null>(null)
  const [profileLoading, setProfileLoading] = useState(false)
  const [historyLoading, setHistoryLoading] = useState(false)
  const [recommendationLoading, setRecommendationLoading] = useState(false)
  const [popularLoading, setPopularLoading] = useState(true)
  const [error, setError] = useState('')
  const [historyError, setHistoryError] = useState('')
  const [recommendationError, setRecommendationError] = useState('')
  const [popularError, setPopularError] = useState('')
  const profileRequest = useRef<AbortController | null>(null)
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
      profileRequest.current?.abort()
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
      const response = await api.getProfilePurchases(profileId, PURCHASE_PAGE_SIZE, offset, controller.signal)
      if (historyRequest.current === controller) setHistory(response)
    } catch (cause) {
      if (!(cause instanceof DOMException && cause.name === 'AbortError') && historyRequest.current === controller) setHistoryError(errorMessage(cause))
    } finally {
      if (historyRequest.current === controller) setHistoryLoading(false)
    }
  }

  const selectProfile = async (profileId: string) => {
    profileRequest.current?.abort()
    historyRequest.current?.abort()
    recommendationRequest.current?.abort()
    const controller = new AbortController()
    profileRequest.current = controller
    setProfileLoading(true)
    setError('')
    setHistory(null)
    setResult(null)
    setRecommendationError('')
    try {
      const response = await api.getProfile(profileId, controller.signal)
      if (profileRequest.current !== controller) return
      setProfile(response)
      setQuery(response.profile_id)
      await loadPurchases(response.profile_id, 0)
    } catch (cause) {
      if (!(cause instanceof DOMException && cause.name === 'AbortError') && profileRequest.current === controller) {
        setProfile(null)
        setError(errorMessage(cause))
      }
    } finally {
      if (profileRequest.current === controller) setProfileLoading(false)
    }
  }

  const search = (event: FormEvent) => {
    event.preventDefault()
    const profileId = query.trim()
    if (!profileId) {
      setError('Informe um profile_id válido.')
      return
    }
    void selectProfile(profileId)
  }

  const randomProfile = async () => {
    profileRequest.current?.abort()
    historyRequest.current?.abort()
    recommendationRequest.current?.abort()
    const controller = new AbortController()
    profileRequest.current = controller
    setProfileLoading(true)
    setError('')
    setHistory(null)
    setResult(null)
    try {
      const response = await api.getRandomProfile(controller.signal)
      if (profileRequest.current !== controller) return
      setProfile(response)
      setQuery(response.profile_id)
      await loadPurchases(response.profile_id, 0)
    } catch (cause) {
      if (!(cause instanceof DOMException && cause.name === 'AbortError') && profileRequest.current === controller) setError(errorMessage(cause))
    } finally {
      if (profileRequest.current === controller) setProfileLoading(false)
    }
  }

  const generate = async () => {
    if (!profile) return
    recommendationRequest.current?.abort()
    const controller = new AbortController()
    recommendationRequest.current = controller
    setRecommendationLoading(true)
    setRecommendationError('')
    try {
      const response = await api.getProfileRecommendations(profile.profile_id, 10, controller.signal)
      if (recommendationRequest.current === controller) setResult(response)
    } catch (cause) {
      if (!(cause instanceof DOMException && cause.name === 'AbortError') && recommendationRequest.current === controller) setRecommendationError(errorMessage(cause))
    } finally {
      if (recommendationRequest.current === controller) setRecommendationLoading(false)
    }
  }

  return (
    <section>
      <PageHeader index="03" title="H&M Recommendation Playground" description="Explore perfil pseudônimo, compras reais e diferença entre ranking popular e híbrido." aside={<span className="inline-flex items-center gap-2 border-2 bg-coral px-4 py-3 font-mono text-xs font-medium uppercase tracking-wider"><SparkIcon className="size-4" /> H&M ranking</span>} />

      <form onSubmit={search} className="panel mb-7 p-4 sm:p-5">
        <div className="grid gap-3 md:grid-cols-[1fr_auto_auto] md:items-end">
          <label htmlFor="profile-id"><span className="mb-2 block text-sm font-extrabold">Buscar perfil pseudônimo por profile_id</span><div className="relative"><SearchIcon className="absolute left-3.5 top-1/2 size-5 -translate-y-1/2 text-slate-500" /><input id="profile-id" aria-label="Buscar perfil pseudônimo por profile_id" className="field pl-11 font-mono" placeholder="Ex.: 000058a12d5b..." value={query} onChange={(event) => setQuery(event.target.value)} /></div></label>
          <button className="button-primary flex items-center justify-center gap-2" disabled={profileLoading} type="submit"><SearchIcon className="size-4" /> Buscar</button>
          <button className="button-secondary flex items-center justify-center gap-2" disabled={profileLoading} type="button" onClick={() => void randomProfile()}><RefreshIcon className="size-4" /> Perfil do modelo</button>
        </div>
      </form>

      {profileLoading && <LoadingState label="Carregando perfil e compras..." />}
      {!profileLoading && error && <ErrorState message={error} />}
      {!profileLoading && !error && !profile && <WelcomeState popular={popular} loading={popularLoading} error={popularError} reload={loadPopular} />}
      {!profileLoading && profile && <ProfileWorkspace profile={profile} history={history} historyLoading={historyLoading} historyError={historyError} loadPurchases={loadPurchases} popular={popular} result={result} recommendationLoading={recommendationLoading} recommendationError={recommendationError} generate={generate} />}
    </section>
  )
}

function WelcomeState({ popular, loading, error, reload }: { popular: Recommendation[]; loading: boolean; error: string; reload: () => Promise<void> }) {
  return <div className="grid gap-6 xl:grid-cols-[.8fr_1.2fr]">
    <div className="panel grid min-h-[340px] place-items-center bg-ink p-8 text-center text-white"><div><span className="mx-auto grid size-16 place-items-center border-2 border-white bg-coral text-white shadow-[5px_5px_0_#fff]"><SearchIcon className="size-8" /></span><h2 className="mt-6 text-2xl font-extrabold">Qual perfil vamos explorar?</h2><p className="mx-auto mt-3 max-w-sm text-sm leading-6 text-mist">Busque um profile_id conhecido ou selecione um perfil coberto pelo modelo ativo. Identidade real nunca aparece.</p></div></div>
    <article className="panel p-5 sm:p-7"><p className="eyebrow">Baseline H&M</p><h2 className="mt-1 text-2xl font-extrabold">Artigos populares</h2><p className="mt-2 text-sm text-slate-600">Mesmo ranking global para todos os perfis.</p>{loading && <div className="mt-5"><LoadingState label="Buscando ranking popular..." /></div>}{!loading && error && <div className="mt-5"><ErrorState message={error} onRetry={() => void reload()} /></div>}{!loading && !error && !popular.length && <div className="mt-5"><EmptyState title="Ranking vazio" description="Nenhum artigo popular retornado." /></div>}{!loading && popular.length > 0 && <RecommendationList recommendations={popular.slice(0, 6)} compact />}</article>
  </div>
}

interface WorkspaceProps {
  profile: Profile
  history: PurchaseHistory | null
  historyLoading: boolean
  historyError: string
  loadPurchases: (profileId: string, offset: number) => Promise<void>
  popular: Recommendation[]
  result: RecommendationResult | null
  recommendationLoading: boolean
  recommendationError: string
  generate: () => Promise<void>
}

function ProfileWorkspace({ profile, history, historyLoading, historyError, loadPurchases, popular, result, recommendationLoading, recommendationError, generate }: WorkspaceProps) {
  const alias = profileAlias(profile.profile_id)
  const pageStart = history ? history.offset + 1 : 0
  const pageEnd = history ? Math.min(history.offset + history.purchases.length, history.total) : 0
  return <div className="space-y-7">
    <div className="grid gap-5 xl:grid-cols-[340px_1fr]">
       <aside className="panel bg-ink p-6 text-white"><p className="font-mono text-[11px] uppercase tracking-widest text-acid">Perfil pseudônimo</p><h2 className="mt-2 break-words text-2xl font-extrabold">{alias}</h2><p className="mt-1 break-all font-mono text-[10px] uppercase tracking-wider text-mist">profile_id {profile.profile_id}</p><div className="mt-6 grid grid-cols-2 gap-3"><ProfileMetric label="Compras" value={profile.purchase_count} /><ProfileMetric label="Faixa etária" value={profile.age_band ?? 'n/d'} /><ProfileMetric label="Última compra" value={profile.last_purchase_at ? formatDate(profile.last_purchase_at) : 'n/d'} /></div><button className="button-primary mt-6 flex w-full items-center justify-center gap-2 bg-coral text-white" onClick={() => void generate()} disabled={recommendationLoading}><SparkIcon className="size-5" />{recommendationLoading ? 'Calculando relevância...' : 'Gerar ranking híbrido'}</button><p className="mt-3 text-center font-mono text-[9px] uppercase tracking-wider text-mist">Artigos comprados são excluídos</p></aside>

      <article className="panel p-5 sm:p-6"><div className="flex flex-col justify-between gap-2 sm:flex-row sm:items-end"><div><p className="eyebrow">Histórico real</p><h2 className="mt-1 text-xl font-extrabold">Compras observadas</h2></div>{history && <span className="font-mono text-xs text-slate-500">{pageStart}-{pageEnd} de {history.total}</span>}</div>
        {historyLoading && <div className="mt-5"><LoadingState label="Carregando página de compras..." /></div>}
        {!historyLoading && historyError && <div className="mt-5"><ErrorState message={historyError} onRetry={() => void loadPurchases(profile.profile_id, history?.offset ?? 0)} /></div>}
        {!historyLoading && !historyError && history && history.purchases.length > 0 && <ol className="mt-5 space-y-2">{history.purchases.map((purchase, index) => <li key={`${purchase.article_id}-${purchase.purchased_at}-${index}`} className="grid grid-cols-[minmax(0,1fr)_auto] gap-3 border-2 bg-white p-3"><div className="min-w-0"><p className="truncate font-bold">{purchase.article.product_name || `Artigo ${purchase.article_id}`}</p><p className="truncate text-xs text-slate-600">{purchase.article.product_type_name || 'Tipo não informado'} · {purchase.article.colour_group_name || 'Cor não informada'}</p><p className="mt-1 font-mono text-[10px] text-slate-500">article_id {purchase.article_id} · preço de origem {formatDecimal(purchase.source_price, 4)}</p></div><time className="text-right font-mono text-[10px] text-slate-500" dateTime={purchase.purchased_at}>{formatDate(purchase.purchased_at)}</time></li>)}</ol>}
        {!historyLoading && !historyError && history && !history.purchases.length && <div className="mt-5"><EmptyState title="Sem compras nesta página" description="Nenhuma compra retornada para este intervalo." /></div>}
        {history && history.total > PURCHASE_PAGE_SIZE && <nav aria-label="Paginação do histórico" className="mt-5 flex items-center justify-between gap-3 border-t-2 pt-4"><button type="button" className="button-secondary" disabled={historyLoading || history.offset === 0} onClick={() => void loadPurchases(profile.profile_id, Math.max(0, history.offset - history.limit))}>Anterior</button><span className="font-mono text-xs">Página {Math.floor(history.offset / history.limit) + 1}</span><button type="button" className="button-secondary" disabled={historyLoading || history.offset + history.limit >= history.total} onClick={() => void loadPurchases(profile.profile_id, history.offset + history.limit)}>Próxima</button></nav>}
      </article>
    </div>

    {recommendationError && <ErrorState message={recommendationError} onRetry={() => void generate()} />}
    {recommendationLoading && <LoadingState label="Combinando compras, faixa etária e catálogo..." />}
    {!recommendationLoading && !recommendationError && !result && <div className="border-2 border-dashed bg-mist/60 p-6 text-center text-sm text-slate-600">Gere ranking para comparar popularidade global e estratégia híbrida.</div>}
    {!recommendationLoading && result && <Comparison result={result} popular={popular} history={history} />}
  </div>
}

function Comparison({ result, popular, history }: { result: RecommendationResult; popular: Recommendation[]; history: PurchaseHistory | null }) {
  return <section aria-live="polite">
    <div className="mb-4 flex flex-col justify-between gap-3 border-b-2 pb-4 sm:flex-row sm:items-end"><div><p className="eyebrow">Top 10 por relevância</p><h2 className="mt-1 text-2xl font-extrabold">Popular vs. híbrido</h2></div><div className="flex flex-wrap gap-2 font-mono text-[10px] uppercase"><span className="border bg-white px-2 py-1">{formatDecimal(result.inference_duration_ms, 1)} ms</span>{result.model_version && <span className="border bg-white px-2 py-1">model {result.model_version}</span>}</div></div>
    {result.strategy === 'popular' && <div className="mb-5 border-2 border-coral bg-[#fff1f2] p-4" role="status"><p className="font-bold">Fallback para popularidade</p><p className="mt-1 text-sm text-slate-700">{result.fallback_reason || 'Estratégia híbrida indisponível para este perfil. API não informou motivo adicional.'}</p></div>}
    <div className="grid gap-6 lg:grid-cols-2">
      <article className="panel-flat p-5"><p className="eyebrow">A · Popular</p><h3 className="mt-1 font-extrabold">Ranking global</h3><p className="mt-1 text-xs text-slate-600">Sem contexto do perfil</p>{popular.length ? <RecommendationList recommendations={popular} /> : <div className="mt-5"><EmptyState title="Popular indisponível" description="API não retornou baseline." /></div>}</article>
      <article className="panel bg-white p-5"><p className="eyebrow">B · {result.strategy === 'hybrid' ? 'Híbrido' : 'Fallback'}</p><h3 className="mt-1 font-extrabold">{result.strategy === 'hybrid' ? 'Compras + idade + catálogo' : 'Popularidade global'}</h3><p className="mt-1 text-xs text-slate-600">Score indica relevância relativa, nunca probabilidade.</p>{result.recommendations.length ? <RecommendationList recommendations={result.recommendations} showEvidence highlight /> : <div className="mt-5"><EmptyState title="Sem recomendações" description="API não retornou candidatos." /></div>}</article>
    </div>
    {result.recommendations.length > 0 && <div className="mt-6"><RecommendationConnectionVisualizer profileId={result.profile_id} purchases={history?.purchases ?? []} recommendations={result.recommendations} strategy={result.strategy} /></div>}
  </section>
}

function RecommendationList({ recommendations, compact = false, highlight = false, showEvidence = false }: { recommendations: Recommendation[]; compact?: boolean; highlight?: boolean; showEvidence?: boolean }) {
  return <ol className="mt-5 divide-y-2 border-y-2">{recommendations.slice(0, compact ? 6 : 10).map((recommendation, index) => <li key={recommendation.article_id} className={`grid grid-cols-[38px_minmax(0,1fr)] gap-3 ${compact ? 'py-2.5' : 'py-3.5'}`}><span className={`grid size-8 place-items-center border-2 font-mono text-xs ${highlight && index < 3 ? 'bg-coral text-white' : 'bg-white'}`}>{recommendation.rank || index + 1}</span><div className="min-w-0"><div className="flex items-start justify-between gap-2"><div className="min-w-0"><p className="truncate font-bold">{recommendation.article.product_name || `Artigo ${recommendation.article_id}`}</p><p className="truncate text-xs text-slate-500">{recommendation.article.product_type_name || recommendation.article.department_name || `article_id ${recommendation.article_id}`}</p></div><span className="shrink-0 font-mono text-xs font-medium">relevância {formatDecimal(recommendation.relevance_score, 4)}</span>{highlight && <ArrowIcon className="mt-0.5 size-4 shrink-0 text-cobalt" />}</div>{showEvidence && <Evidence evidence={recommendation.evidence} />}</div></li>)}</ol>
}

function Evidence({ evidence }: { evidence: Recommendation['evidence'] }) {
  if (!evidence.length) return <p className="mt-2 text-xs text-slate-500">Sem evidência factual retornada.</p>
  return <dl className="mt-2 grid gap-1 border-l-2 border-coral pl-3 text-xs">{evidence.map((fact, index) => <div key={`${fact.reason}-${index}`}><dt className="font-bold">Evidência {index + 1}</dt><dd className="text-slate-600">{fact.reason}</dd>{fact.cohort_buyers > 0 && <dd className="font-mono text-[10px]">{formatInteger(fact.cohort_buyers)} compradores na coorte</dd>}{fact.related_history_articles > 0 && <dd className="font-mono text-[10px]">{formatInteger(fact.related_history_articles)} artigos relacionados no histórico</dd>}{fact.same_department && <dd className="font-mono text-[10px]">Mesmo departamento: sim</dd>}</div>)}</dl>
}

function ProfileMetric({ label, value }: { label: string; value: number | string }) {
  return <div className="border border-white/30 p-3"><p className="font-mono text-[9px] uppercase tracking-wider text-mist">{label}</p><p className="mt-1 break-words text-lg font-extrabold">{typeof value === 'number' ? formatInteger(value) : value}</p></div>
}
