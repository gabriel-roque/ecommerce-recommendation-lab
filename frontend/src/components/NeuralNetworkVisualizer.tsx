import { useRef, useState } from 'react'
import type { NetworkConnection, NetworkSnapshot } from '../types/api'
import { formatDate, formatDecimal } from '../utils'

const layerPoints = {
  profile_embedding: Array.from({ length: 6 }, (_, index) => ({ x: 92, y: 92 + index * 44 })),
  article_embedding: Array.from({ length: 6 }, (_, index) => ({ x: 238, y: 92 + index * 44 })),
  dense_64: Array.from({ length: 8 }, (_, index) => ({ x: 470, y: 48 + index * 42 })),
  dense_32: Array.from({ length: 6 }, (_, index) => ({ x: 700, y: 92 + index * 44 })),
  output: [{ x: 880, y: 202 }],
}

type Layer = keyof typeof layerPoints

function pointFor(layer: Layer, index: number) {
  return layerPoints[layer][index]
}

function connectionKey(connection: NetworkConnection) {
  return `${connection.source_layer}-${connection.source_index}-${connection.target_layer}-${connection.target_index}`
}

function Node({ x, y, activation, label, nodeKey, focused, onFocus }: { x: number; y: number; activation: number; label: string; nodeKey: string; focused: boolean; onFocus: (key: string | null) => void }) {
  const magnitude = Math.min(1, Math.abs(activation))
  const color = activation < 0 ? '#777777' : '#e50010'
  return (
    <g className="network-node cursor-pointer" tabIndex={0} role="button" aria-label={`${label}: ${formatDecimal(activation, 4)}`} onMouseEnter={() => onFocus(nodeKey)} onMouseLeave={() => onFocus(null)} onFocus={() => onFocus(nodeKey)} onBlur={() => onFocus(null)}>
      <circle className="network-node-halo" cx={x} cy={y} r={18 + magnitude * 8} fill={color} opacity={focused ? .34 : .12 + magnitude * .1} />
      <circle cx={x} cy={y} r={7 + magnitude * 7} fill={color} fillOpacity={.55 + magnitude * .45} stroke={focused ? '#f7a8ae' : '#f8fafc'} strokeWidth={focused ? 3 : 1.5} />
      <title>{label}: {formatDecimal(activation, 4)}</title>
    </g>
  )
}

export function NeuralNetworkVisualizer({ snapshot, denseSizes = [64, 32] }: { snapshot?: NetworkSnapshot; denseSizes?: [number, number] }) {
  const svgRef = useRef<SVGSVGElement>(null)
  const reducedMotion = typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
  const [playing, setPlaying] = useState(!reducedMotion)
  const [focusedNode, setFocusedNode] = useState<string | null>(null)

  if (!snapshot) {
    return (
      <div className="grid min-h-[330px] place-items-center border-2 border-dashed bg-paper/60 p-8 text-center">
        <div><p className="font-bold">Ativações aguardando primeira amostra técnica</p><p className="mt-2 max-w-md text-sm leading-6 text-slate-600">O desenho aparece somente quando o processo de treino enviar valores medidos pela conexão de eventos.</p></div>
      </div>
    )
  }

  const nodeExists = (layer: Layer, index: number) => {
    if (!pointFor(layer, index)) return false
    if (layer === 'output') return index === 0
    return index < snapshot[layer].length
  }
  const connections = snapshot.connections
    .filter((connection) => nodeExists(connection.source_layer, connection.source_index) && nodeExists(connection.target_layer, connection.target_index))
    .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
    .slice(0, 48)
  const animatedConnections = new Set(connections.slice(0, 8).map(connectionKey))
  const customerName = snapshot.probe_customer_name ?? null

  const toggleAnimation = () => {
    const next = !playing
    setPlaying(next)
    if (next) svgRef.current?.unpauseAnimations?.()
    else svgRef.current?.pauseAnimations?.()
  }

  const isConnectionFocused = (connection: NetworkConnection) => !focusedNode
    || focusedNode === `${connection.source_layer}-${connection.source_index}`
    || focusedNode === `${connection.target_layer}-${connection.target_index}`

  const renderNodes = (layer: Exclude<Layer, 'output'>, values: number[], label: string) => values.slice(0, layerPoints[layer].length).map((activation, index) => {
    const key = `${layer}-${index}`
    return <Node key={key} {...layerPoints[layer][index]} activation={activation} label={`${label} ${index + 1}`} nodeKey={key} focused={focusedNode === key} onFocus={setFocusedNode} />
  })

  return (
    <div className="min-w-0">
      <div className="mb-4 flex flex-col justify-between gap-3 sm:flex-row sm:items-center">
        <div className="flex min-w-0 flex-wrap gap-2 font-mono text-[10px] uppercase tracking-wider">
          <span className="max-w-full break-words border bg-white px-2 py-1">Cliente da amostra #{snapshot.probe_profile_id ?? 'não informado'}{customerName && ` · ${customerName} (sintético)`}</span>
          <span className="border bg-white px-2 py-1">Produto da amostra #{snapshot.probe_article_id ?? 'não informado'}</span>
          <span className="border bg-white px-2 py-1">Época {snapshot.epoch}</span>
          <time className="border bg-white px-2 py-1" dateTime={snapshot.timestamp}>{formatDate(snapshot.timestamp)}</time>
        </div>
        <button type="button" className="button-secondary shrink-0" onClick={toggleAnimation} disabled={reducedMotion} aria-pressed={!playing}>
          {reducedMotion ? 'Movimento reduzido' : playing ? 'Pausar animação' : 'Reproduzir animação'}
        </button>
      </div>

      <div className={`network-canvas overflow-hidden border-2 bg-[#080808] ${playing ? '' : 'network-paused'}`}>
        <svg ref={svgRef} className="block h-auto w-full" viewBox="0 0 960 405" role="img" aria-labelledby="network-title" aria-describedby="network-description">
          <title id="network-title">Ativações e pesos reais da rede neural</title>
           <desc id="network-description">Camadas com ativações da amostra técnica, até 48 pesos de maior magnitude e saída numérica. Partículas ilustram fluxo. Este retrato não explica recomendação individual.</desc>
          <defs>
            <pattern id="network-grid" width="32" height="32" patternUnits="userSpaceOnUse"><path d="M32 0H0V32" fill="none" stroke="#26304d" strokeWidth=".6" /></pattern>
            <radialGradient id="network-vignette"><stop offset="0" stopColor="#331015" /><stop offset="1" stopColor="#080808" /></radialGradient>
          </defs>
          <rect width="960" height="405" fill="url(#network-vignette)" />
          <rect width="960" height="405" fill="url(#network-grid)" opacity=".55" />

          {connections.map((connection) => {
            const start = pointFor(connection.source_layer, connection.source_index)!
            const end = pointFor(connection.target_layer, connection.target_index)!
            const key = connectionKey(connection)
            const focused = isConnectionFocused(connection)
            const color = connection.weight >= 0 ? '#e50010' : '#777777'
            return (
              <g key={key} opacity={focused ? 1 : .08} className="transition-opacity duration-200">
                <line x1={start.x} y1={start.y} x2={end.x} y2={end.y} stroke={color} strokeOpacity={.18 + Math.min(.58, Math.abs(connection.weight) * .4)} strokeWidth={.7 + Math.min(4, Math.abs(connection.weight) * 3)}><title>Peso real: {formatDecimal(connection.weight, 5)}</title></line>
                {playing && animatedConnections.has(key) && <circle r="2.8" fill="#f7a8ae" aria-hidden="true"><animateMotion dur={`${2.4 + (Math.abs(connection.weight) % 1)}s`} repeatCount="indefinite" path={`M${start.x},${start.y} L${end.x},${end.y}`} /></circle>}
              </g>
            )
          })}

          <g className="font-mono text-[10px] font-medium uppercase tracking-wider" fill="#dfe4f2">
            <text x="92" y="30" textAnchor="middle">Vetor cliente</text><text x="238" y="30" textAnchor="middle">Vetor produto</text>
             <text x="470" y="24" textAnchor="middle">Densa {denseSizes[0]}</text><text x="700" y="68" textAnchor="middle">Densa {denseSizes[1]}</text><text x="880" y="166" textAnchor="middle">Saída</text>
          </g>
          {renderNodes('profile_embedding', snapshot.profile_embedding, 'Vetor do cliente')}
          {renderNodes('article_embedding', snapshot.article_embedding, 'Vetor do produto')}
           {renderNodes('dense_64', snapshot.dense_64, `Camada densa ${denseSizes[0]}, ativação`)}
           {renderNodes('dense_32', snapshot.dense_32, `Camada densa ${denseSizes[1]}, ativação`)}
          <Node {...layerPoints.output[0]} activation={snapshot.output_score} label="Saída numérica da amostra" nodeKey="output-0" focused={focusedNode === 'output-0'} onFocus={setFocusedNode} />
          <text x="880" y="250" textAnchor="middle" className="font-mono text-xl font-medium" fill="#f8fafc">{formatDecimal(snapshot.output_score, 3)}</text>
        </svg>
      </div>

      <div className="mt-3 grid gap-3 text-xs leading-5 text-slate-600 sm:grid-cols-[1fr_auto] sm:items-start">
        <p className="border-l-4 border-cobalt pl-3"><strong>Amostra técnica, não explicação individual.</strong> Tamanho e brilho seguem magnitude da ativação. Linhas mostram até 48 pesos de maior magnitude. Movimento é ilustrativo.</p>
        <div className="flex gap-3 font-mono text-[10px] uppercase"><span><i className="mr-1 inline-block size-2 bg-[#e50010]" /> positivo</span><span><i className="mr-1 inline-block size-2 bg-[#777777]" /> negativo</span></div>
      </div>
      <details className="mt-4 border-2 bg-white p-3 text-sm">
        <summary className="min-h-11 cursor-pointer py-3 font-bold">Tabela acessível da amostra técnica</summary>
        <div className="overflow-x-auto"><table className="w-full min-w-[420px] text-left"><caption className="sr-only">Valores reais de ativação por camada</caption><thead><tr className="border-b-2"><th className="p-2">Camada</th><th className="p-2">Índice</th><th className="p-2">Ativação</th></tr></thead><tbody>{([
          ['Vetor do cliente', snapshot.profile_embedding], ['Vetor do produto', snapshot.article_embedding], [`Camada densa ${denseSizes[0]}`, snapshot.dense_64], [`Camada densa ${denseSizes[1]}`, snapshot.dense_32], ['Saída', [snapshot.output_score]],
        ] as Array<[string, number[]]>).flatMap(([layer, values]) => values.map((value, index) => <tr className="border-b" key={`${layer}-${index}`}><td className="p-2">{layer}</td><td className="p-2 font-mono">{index}</td><td className="p-2 font-mono">{formatDecimal(value, 5)}</td></tr>))}</tbody></table></div>
      </details>
    </div>
  )
}
