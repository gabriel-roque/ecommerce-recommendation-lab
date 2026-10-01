export const formatInteger = (value: number) => new Intl.NumberFormat('pt-BR').format(value)

export const formatDecimal = (value: number, digits = 2) =>
  new Intl.NumberFormat('pt-BR', { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(value)

export const formatDuration = (seconds?: number | null) => {
  if (seconds == null) return '—'
  if (seconds < 60) return `${formatDecimal(seconds, 1)} s`
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)
  return hours ? `${hours}h ${minutes}min` : `${minutes}min ${secs}s`
}

export const formatDate = (value: string) =>
  new Intl.DateTimeFormat('pt-BR', { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(value))

export const errorMessage = (error: unknown) => error instanceof Error ? error.message : 'Erro inesperado.'
