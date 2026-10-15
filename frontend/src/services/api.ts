import axios from 'axios'
import { clearAccessToken, getAccessToken } from './authToken'

// Keep API configuration centralized so later service modules share the same base URL.
export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000',
  headers: { 'Content-Type': 'application/json' },
  timeout: 15_000,
})

// Centralize bearer token attachment so feature services do not manage auth headers.
apiClient.interceptors.request.use((config) => {
  const token = getAccessToken()
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// A 401 invalidates the browser session, except for login failures themselves.
apiClient.interceptors.response.use(
  (response) => response,
  (error: unknown) => {
    if (axios.isAxiosError(error) && error.response?.status === 401) {
      const path = error.config?.url ?? ''
      if (!path.replace(/\/$/, '').endsWith('/auth/login')) {
        clearAccessToken()
        if (typeof window !== 'undefined') {
          window.dispatchEvent(new Event('xai-cdss:unauthorized'))
        }
      }
    }
    return Promise.reject(error)
  },
)

export const apiBaseUrl = apiClient.defaults.baseURL
