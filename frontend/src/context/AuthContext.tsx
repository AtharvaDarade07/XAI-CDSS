import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import type { AuthState, User } from '../types/api'
import { fetchCurrentUser, getAccessToken, login as loginRequest, logout as clearSession } from '../services/authService'

const AuthContext = createContext<AuthState | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [token, setToken] = useState<string | null>(() => getAccessToken())
  const [isLoading, setIsLoading] = useState(true)

  const logout = useCallback(() => {
    clearSession()
    setToken(null)
    setUser(null)
  }, [])

  useEffect(() => {
    let active = true
    const storedToken = getAccessToken()
    if (!storedToken) {
      setToken(null)
      setUser(null)
      setIsLoading(false)
      return () => { active = false }
    }

    setToken(storedToken)
    fetchCurrentUser()
      .then((currentUser) => {
        if (active) setUser(currentUser)
      })
      .catch(() => {
        clearSession()
        if (active) {
          setToken(null)
          setUser(null)
        }
      })
      .finally(() => {
        if (active) setIsLoading(false)
      })

    return () => { active = false }
  }, [])

  useEffect(() => {
    const handleUnauthorized = () => {
      setToken(null)
      setUser(null)
      setIsLoading(false)
    }
    window.addEventListener('xai-cdss:unauthorized', handleUnauthorized)
    return () => window.removeEventListener('xai-cdss:unauthorized', handleUnauthorized)
  }, [])

  const login = useCallback(async (username: string, password: string) => {
    try {
      const authenticatedUser = await loginRequest(username, password)
      setToken(getAccessToken())
      setUser(authenticatedUser)
      return authenticatedUser
    } catch (error) {
      clearSession()
      setToken(null)
      setUser(null)
      throw error
    }
  }, [])

  const value = useMemo<AuthState>(() => ({
    user,
    token,
    isAuthenticated: Boolean(user && token),
    isLoading,
    login,
    logout,
  }), [user, token, isLoading, login, logout])

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthState {
  const context = useContext(AuthContext)
  if (!context) throw new Error('useAuth must be used within AuthProvider')
  return context
}
