const ACCESS_TOKEN_KEY = 'xai-cdss.access-token'

export function getAccessToken(): string | null {
  try {
    return typeof window === 'undefined' ? null : window.localStorage.getItem(ACCESS_TOKEN_KEY)
  } catch {
    return null
  }
}

export function storeAccessToken(token: string): void {
  window.localStorage.setItem(ACCESS_TOKEN_KEY, token)
}

export function clearAccessToken(): void {
  try {
    window.localStorage.removeItem(ACCESS_TOKEN_KEY)
  } catch {
    // Keep sign-out state usable even when browser storage is unavailable.
  }
}
