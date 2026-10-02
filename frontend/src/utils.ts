export const formatInteger = (value: number) => new Intl.NumberFormat('pt-BR').format(value)

export const formatDecimal = (value: number, digits = 2) =>
  new Intl.NumberFormat('pt-BR', { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(value)

export const formatCurrency = (value: number) =>
  new Intl.NumberFormat('pt-BR', { style: 'currency', currency: 'BRL' }).format(value)

export const formatDuration = (seconds?: number | null) => {
  if (seconds == null) return '—'
  if (seconds < 60) return `${formatDecimal(seconds, 1)} s`
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)
  return hours ? `${hours}h ${minutes}min` : `${minutes}min ${secs}s`
}

export const formatDate = (value: string) => {
  if (/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    return new Intl.DateTimeFormat('pt-BR', { dateStyle: 'medium' }).format(
      new Date(`${value}T12:00:00`),
    )
  }
  return new Intl.DateTimeFormat('pt-BR', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value))
}

export const errorMessage = (error: unknown) => error instanceof Error ? error.message : 'Erro inesperado.'
