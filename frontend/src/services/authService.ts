import { apiClient } from './api'
import { clearAccessToken, getAccessToken, storeAccessToken } from './authToken'
import type { AuthLoginRequest, AuthLoginResponse, User } from '../types/api'

export async function login(username: string, password: string): Promise<User> {
  const credentials: AuthLoginRequest = { username, password }
  const { data } = await apiClient.post<AuthLoginResponse>('/auth/login', credentials)

  if (!data.access_token || data.token_type.toLowerCase() !== 'bearer') {
    throw new Error('The authentication service returned an unsupported token response.')
  }

  storeAccessToken(data.access_token)
  try {
    return await fetchCurrentUser()
  } catch (error) {
    clearAccessToken()
    throw error
  }
}

export async function fetchCurrentUser(): Promise<User> {
  const { data } = await apiClient.get<User>('/auth/me')
  return data
}

export function logout(): void {
  clearAccessToken()
}

export { getAccessToken }
