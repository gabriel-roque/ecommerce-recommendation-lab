import { useState } from 'react'
import type { Purchase, Recommendation, RecommendationResult } from '../types/api'
import { formatDecimal } from '../utils'

interface Props {
  profileId: string
  customerName: string
  purchases: Purchase[]
  recommendations: Recommendation[]
  strategy: RecommendationResult['strategy']
}

const yPosition = (index: number, total: number) => 64 + (index + 1) * (310 / (total + 1))

export function RecommendationConnectionVisualizer({ profileId, customerName, purchases, recommendations, strategy }: Props) {
  const reducedMotion = typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
  const [active, setActive] = useState('')
  const visiblePurchases = strategy === 'model' ? purchases.slice(0, 6) : []
  const visibleRecommendations = recommendations.slice(0, 8)
  const displayName = customerName || 'Cliente sintético'
  const selectedPurchase = active.startsWith('purchase-') ? visiblePurchases[Number(active.split('-')[1])] : undefined
  const selectedRecommendation = active.startsWith('recommendation-') ? visibleRecommendations[Number(active.split('-')[1])] : undefined

  return <figure className="panel overflow-hidden bg-ink text-white" aria-labelledby="connection-graph-heading">
    <figcaption className="border-b border-white/20 p-5 sm:p-6"><p className="font-mono text-[11px] uppercase tracking-[.18em] text-acid">Mapa factual</p><h3 id="connection-graph-heading" className="mt-1 text-xl font-extrabold">Compras e ranking em contexto</h3><p className="mt-2 max-w-2xl text-sm leading-6 text-mist">Esquerda mostra compras observadas. Direita mostra produtos ordenados. Linhas organizam leitura. Não representam causa, influência individual ou caminho de decisão do modelo.</p></figcaption>
    <div className="overflow-hidden"><svg className="block h-auto min-h-[310px] w-full" viewBox="0 0 900 430" role="img" aria-labelledby="recommendation-map-title" aria-describedby="recommendation-map-description">
      <title id="recommendation-map-title">Relações visuais entre compras, cliente e recomendações</title>
      <desc id="recommendation-map-description">{strategy === 'model' ? `${displayName}, cliente sintético com ID ${profileId}, aparece entre compras observadas e ranking híbrido. Linhas não indicam causalidade.` : 'Popularidade global organiza produtos recomendados. Nenhuma relação personalizada aparece.'}</desc>
      <defs><radialGradient id="recommendation-map-bg"><stop offset="0" stopColor="#331015" /><stop offset="1" stopColor="#080808" /></radialGradient></defs><rect width="900" height="430" fill="url(#recommendation-map-bg)" />
      {visiblePurchases.map((purchase, index) => { const y = yPosition(index, visiblePurchases.length); return <g key={`purchase-edge-${purchase.article_id}-${index}`}><line x1="178" y1={y} x2="416" y2="215" stroke="#f7a8ae" strokeWidth="2" strokeOpacity={active && active !== `purchase-${index}` ? .12 : .7} />{!reducedMotion && <circle r="3" fill="#f7a8ae"><animateMotion dur={`${2.8 + index * .17}s`} repeatCount="indefinite" path={`M178,${y} L416,215`} /></circle>}</g> })}
      {visibleRecommendations.map((recommendation, index) => { const y = yPosition(index, visibleRecommendations.length); const relevance = Math.max(0, Math.min(1, recommendation.score / 100)); return <g key={`recommendation-edge-${recommendation.article_id}`}><line x1="484" y1="215" x2="720" y2={y} stroke="#fff" strokeWidth={1 + relevance * 5} strokeOpacity={active && active !== `recommendation-${index}` ? .12 : .3 + relevance * .6} />{!reducedMotion && index < 5 && <circle r="3" fill="#fff"><animateMotion dur={`${2.2 + index * .2}s`} repeatCount="indefinite" path={`M484,215 L720,${y}`} /></circle>}</g> })}
      {visiblePurchases.map((purchase, index) => { const y = yPosition(index, visiblePurchases.length); return <g key={`purchase-${purchase.article_id}-${index}`} tabIndex={0} role="button" aria-label={`Compra observada do produto ${purchase.article_id}`} className="cursor-pointer" onMouseEnter={() => setActive(`purchase-${index}`)} onMouseLeave={() => setActive('')} onFocus={() => setActive(`purchase-${index}`)} onBlur={() => setActive('')}><circle cx="150" cy={y} r="27" fill="#111" stroke="#f7a8ae" strokeWidth="3" /><text x="150" y={y + 4} textAnchor="middle" fill="#fff" className="font-mono text-[9px]">{purchase.article_id.slice(-8)}</text></g> })}
      <g tabIndex={0} role="button" aria-label={strategy === 'model' ? `${displayName}, cliente sintético com ID ${profileId}` : 'Popularidade global sem personalização'} className="cursor-pointer"><circle cx="450" cy="215" r="47" fill={strategy === 'model' ? '#e50010' : '#555'} stroke="#fff" strokeWidth="3" /><text x="450" y="210" textAnchor="middle" fill="#fff" className="font-sans text-[11px] font-bold">{strategy === 'model' ? displayName.split(' ')[0] : 'Global'}</text><text x="450" y="228" textAnchor="middle" fill="#eee" className="font-mono text-[8px] uppercase">{strategy === 'model' ? 'cliente sintético' : 'popular'}</text></g>
      {visibleRecommendations.map((recommendation, index) => { const y = yPosition(index, visibleRecommendations.length); return <g key={`recommendation-${recommendation.article_id}`} tabIndex={0} role="button" aria-label={`Produto recomendado ${recommendation.article_id}, pontuação ${formatDecimal(recommendation.score, 1)} de 100`} className="cursor-pointer" onMouseEnter={() => setActive(`recommendation-${index}`)} onMouseLeave={() => setActive('')} onFocus={() => setActive(`recommendation-${index}`)} onBlur={() => setActive('')}><circle cx="750" cy={y} r="25" fill="#111" stroke="#fff" strokeWidth={active === `recommendation-${index}` ? 4 : 2} /><text x="750" y={y + 4} textAnchor="middle" fill="#fff" className="font-mono text-[9px]">{recommendation.article_id.slice(-8)}</text></g> })}
      <g className="font-mono text-[9px] uppercase tracking-wider" fill="#ddd"><text x="150" y="28" textAnchor="middle">Compras observadas</text><text x="450" y="28" textAnchor="middle">{strategy === 'model' ? 'Contexto híbrido' : 'Alternativa global'}</text><text x="750" y="28" textAnchor="middle">Produtos ordenados</text></g>
    </svg></div>
    <div className="border-t border-white/20 p-5 text-xs text-mist" aria-live="polite">{selectedPurchase && <>Compra observada: {selectedPurchase.article.product_name || `produto ${selectedPurchase.article_id}`}.</>}{selectedRecommendation && <>Produto {selectedRecommendation.article_id}, pontuação {formatDecimal(selectedRecommendation.score, 1)} de 100.</>}{!selectedPurchase && !selectedRecommendation && (strategy === 'model' ? `${displayName} identifica cliente sintético. Mapa não explica recomendação individual.` : 'Estratégia alternativa usa popularidade global e não representa cliente selecionado.')}</div>
    <details className="border-t border-white/20 p-5 text-sm"><summary className="min-h-11 cursor-pointer py-3 font-bold text-white">Lista acessível de relações</summary><ul className="space-y-2 text-mist">{visiblePurchases.map((purchase, index) => <li key={`accessible-purchase-${purchase.article_id}-${index}`}>Compra observada do produto {purchase.article_id}.</li>)}{visibleRecommendations.map((recommendation) => <li key={`accessible-recommendation-${recommendation.article_id}`}>{strategy === 'model' ? displayName : 'Popularidade global'} aparece no mesmo mapa que o produto {recommendation.article_id}, com pontuação {formatDecimal(recommendation.score, 1)} de 100. Isso não indica causalidade.</li>)}</ul></details>
  </figure>
}
