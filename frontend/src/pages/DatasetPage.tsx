import { useEffect, useRef, useState } from 'react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { DatabaseIcon } from '../components/Icons'
import { MetricCard } from '../components/MetricCard'
import { PageHeader } from '../components/PageHeader'
import { EmptyState, ErrorState, LoadingState } from '../components/States'
import { api } from '../services/api'
import type { DatasetStats } from '../types/api'
import { errorMessage, formatDecimal, formatInteger } from '../utils'

export function DatasetPage() {
  const [stats, setStats] = useState<DatasetStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const activeRequest = useRef<AbortController | null>(null)

  const load = (force = false) => {
    activeRequest.current?.abort()
    const controller = new AbortController()
    activeRequest.current = controller
    setLoading(true)
    setError('')
    return api.getDatasetStats({ signal: controller.signal, force })
      .then((response) => { if (activeRequest.current === controller) setStats(response) })
      .catch((cause: unknown) => {
        if (!(cause instanceof DOMException && cause.name === 'AbortError') && activeRequest.current === controller) setError(errorMessage(cause))
      })
      .finally(() => { if (activeRequest.current === controller) setLoading(false) })
  }

  useEffect(() => {
    void load()
    return () => activeRequest.current?.abort()
  }, [])

  return (
    <section>
      <PageHeader
        index="01"
        title="Conjunto de dados sintético"
        description="Cobertura de clientes, produtos, compras e faixas etárias antes do treino. Uma única semântica: compra observada."
        aside={<div className="inline-flex items-center gap-2 border-2 bg-acid px-4 py-3 font-mono text-xs font-medium uppercase tracking-wider"><span className="status-dot" /> Sintético · compras · BRL</div>}
      />
      {loading && <LoadingState label="Carregando estatísticas..." />}
      {!loading && error && <ErrorState message={error} onRetry={() => void load(true)} />}
      {!loading && !error && !stats && <EmptyState title="Estatísticas indisponíveis" description="API respondeu sem dados do comércio eletrônico." />}
      {!loading && stats && <DatasetContent stats={stats} />}
    </section>
  )
}

function DatasetContent({ stats }: { stats: DatasetStats }) {
  const ageGroups = Object.entries(stats.age_coverage.groups).map(([ageBand, customers]) => ({ ageBand, customers }))
  return (
    <div className="space-y-8">
      <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
        <MetricCard label="Compras" value={formatInteger(stats.purchase_count)} note="Sinal positivo observado" icon={<DatabaseIcon className="size-5" />} />
        <MetricCard label="Clientes" value={formatInteger(stats.customer_count)} note={`${formatDecimal(stats.average_purchases_per_customer, 1)} compras / cliente`} accent="coral" />
        <MetricCard label="Produtos" value={formatInteger(stats.product_count)} note={`${formatDecimal(stats.average_purchases_per_product, 1)} compras / produto`} accent="acid" />
        <MetricCard label="Idade conhecida" value={`${formatDecimal(stats.age_coverage.known_ratio * 100, 1)}%`} note={`${formatInteger(stats.age_coverage.known)} clientes com idade`} />
      </div>

      <div className="grid gap-5 xl:grid-cols-[1.05fr_.95fr]">
        <figure className="panel min-w-0 p-5 sm:p-6">
          <figcaption><p className="eyebrow">Distribuição etária</p><h2 className="mt-1 text-xl font-extrabold">Clientes por faixa de idade</h2></figcaption>
          {ageGroups.length ? <>
            <div className="mt-5 h-72" role="img" aria-label="Gráfico de clientes por faixa de idade">
              <ResponsiveContainer width="100%" height="100%"><BarChart data={ageGroups} margin={{ left: -12 }}><CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e2e2e2" /><XAxis dataKey="ageBand" tick={{ fontSize: 11 }} /><YAxis tick={{ fontSize: 11 }} /><Tooltip formatter={(value) => formatInteger(Number(value))} contentStyle={{ border: '2px solid #111', borderRadius: 0 }} /><Bar dataKey="customers" name="Clientes" fill="#e50010" stroke="#111" strokeWidth={1.5} /></BarChart></ResponsiveContainer>
            </div>
            <table className="sr-only"><caption>Clientes por faixa de idade</caption><thead><tr><th>Faixa</th><th>Clientes</th></tr></thead><tbody>{ageGroups.map((group) => <tr key={group.ageBand}><td>{group.ageBand}</td><td>{group.customers}</td></tr>)}</tbody></table>
          </> : <div className="mt-5"><EmptyState title="Sem faixas etárias" description="API não retornou distribuição de idade." /></div>}
        </figure>

        <article className="panel bg-ink p-6 text-white">
          <p className="font-mono text-[11px] uppercase tracking-[.18em] text-acid">Leitura correta</p>
          <h2 className="mt-2 text-2xl font-extrabold">Compras, não intenção.</h2>
          <p className="mt-3 max-w-xl text-sm leading-6 text-mist">Cada linha representa produto comprado por cliente sintético. Idade adiciona contexto de grupo. Metadados do catálogo adicionam contexto de produto. Valores monetários usam BRL.</p>
          <dl className="mt-6 grid grid-cols-2 gap-3">
            <DataFact label="Idade ausente" value={formatInteger(stats.age_coverage.missing)} dark />
            <DataFact label="Mediana / cliente" value={formatDecimal(stats.median_purchases_per_customer, 1)} dark />
            <DataFact label="Início" value={formatDateOnly(stats.date_range.start)} dark />
            <DataFact label="Fim" value={formatDateOnly(stats.date_range.end)} dark />
          </dl>
        </article>
      </div>

      <div className="grid gap-5 xl:grid-cols-2">
        <figure className="panel min-w-0 p-5 sm:p-6">
          <figcaption><p className="eyebrow">Frequência</p><h2 className="mt-1 text-xl font-extrabold">Compras por cliente</h2></figcaption>
          {stats.purchases_per_customer.length ? <div className="mt-5 h-64" role="img" aria-label="Gráfico de compras por cliente"><ResponsiveContainer width="100%" height="100%"><BarChart data={stats.purchases_per_customer} margin={{ left: -12 }}><CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e2e2e2" /><XAxis dataKey="bucket" tick={{ fontSize: 11 }} /><YAxis tick={{ fontSize: 11 }} /><Tooltip formatter={(value) => formatInteger(Number(value))} contentStyle={{ border: '2px solid #111', borderRadius: 0 }} /><Bar dataKey="customers" name="Clientes" fill="#111" /></BarChart></ResponsiveContainer></div> : <div className="mt-5"><EmptyState title="Sem distribuição" description="Faixas de compra ainda não disponíveis." /></div>}
        </figure>

        <article className="panel p-5 sm:p-6">
          <p className="eyebrow">Catálogo</p><h2 className="mt-1 text-xl font-extrabold">Produtos mais comprados</h2>
          {stats.top_products.length ? <ol className="mt-5 divide-y-2 border-y-2">{stats.top_products.slice(0, 6).map((product, index) => <li key={product.article_id} className="grid grid-cols-[36px_minmax(0,1fr)_auto] items-center gap-3 py-3"><span className="font-mono text-xs text-slate-500">#{String(index + 1).padStart(2, '0')}</span><span className="min-w-0"><span className="block truncate font-bold">{product.product_name}</span><span className="block font-mono text-[10px] text-slate-500">{product.article_id}</span></span><span className="font-mono text-sm">{formatInteger(product.purchases)}</span></li>)}</ol> : <div className="mt-5"><EmptyState title="Sem produtos" description="Ranking ainda não disponível." /></div>}
        </article>
      </div>
    </div>
  )
}

function formatDateOnly(value: string | null) {
  if (!value) return 'n/d'
  const [year, month, day] = value.split('-')
  return `${day}/${month}/${year}`
}

function DataFact({ label, value, dark = false }: { label: string; value: string; dark?: boolean }) {
  return <div className={`border-l-2 pl-3 ${dark ? 'border-white/50' : ''}`}><dt className={`font-mono text-[10px] uppercase tracking-widest ${dark ? 'text-mist' : 'text-slate-600'}`}>{label}</dt><dd className="mt-1 font-bold">{value}</dd></div>
}
