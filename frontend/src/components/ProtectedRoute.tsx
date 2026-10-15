import type { ReactNode } from 'react'
import { Navigate, Outlet, useLocation } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { LoadingState } from './StatePanels'

export function ProtectedRoute({ children }: { children?: ReactNode }) {
  const { isAuthenticated, isLoading } = useAuth()
  const location = useLocation()

  if (isLoading) return <LoadingState title="Restoring your session" />
  if (!isAuthenticated) return <Navigate to="/login" replace state={{ from: location }} />
  return children ?? <Outlet />
}

export function GuestRoute({ children }: { children: ReactNode }) {
  const { isAuthenticated, isLoading } = useAuth()
  if (isLoading) return <LoadingState title="Checking your session" />
  if (isAuthenticated) return <Navigate to="/dashboard" replace />
  return children
}
