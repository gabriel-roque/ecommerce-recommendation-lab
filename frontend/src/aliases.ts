export function profileAlias(profileId: string) {
  let hash = 0
  for (const character of profileId) hash = (hash * 31 + character.charCodeAt(0)) | 0
  return `Profile ${String(Math.abs(hash) % 100_000).padStart(5, '0')}`
}
