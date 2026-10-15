import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { AxiosError, type AxiosAdapter } from 'axios'
import { MemoryRouter } from 'react-router-dom'
import App from '../App'
import { AuthProvider } from '../context/AuthContext'
import { apiClient } from '../services/api'

const TOKEN_KEY = 'xai-cdss.access-token'
const demoUser = {
  id: 7,
  username: 'clinician-one',
  email: 'clinician@example.test',
  role: 'clinician',
  is_active: true,
}

function makeAdapter(): AxiosAdapter {
  return async (config) => {
    let data: unknown
    let status = 200
    if (config.url === '/auth/login') {
      const credentials = typeof config.data === 'string' ? JSON.parse(config.data) as Record<string, string> : {}
      if (credentials.username === 'clinician-one' && credentials.password === 'test-password') {
        data = { access_token: 'demo-token', token_type: 'bearer', expires_in: 1800 }
      } else {
        status = 401
        data = { detail: 'Invalid username or password' }
      }
    } else if (config.url === '/auth/me') {
      const authorization = config.headers.get('Authorization')
      if (authorization === 'Bearer demo-token') data = demoUser
      else {
        status = 401
        data = { detail: 'Invalid or missing authentication credentials' }
      }
    } else if (config.url === '/patients') {
      data = { items: [], skip: 0, limit: 5, total: 0 }
    } else {
      data = { status: 'ok' }
    }

    const response = {
      data,
      status,
      statusText: status === 200 ? 'OK' : 'Unauthorized',
      headers: {},
      config,
    }
    if (status === 401) {
      throw new AxiosError('Request failed with status code 401', AxiosError.ERR_BAD_REQUEST, config, undefined, response)
    }
    return response
  }
}

function renderAt(path: string) {
  return render(
    <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }} initialEntries={[path]}>
      <AuthProvider><App /></AuthProvider>
    </MemoryRouter>,
  )
}

beforeEach(() => {
  window.localStorage.clear()
  apiClient.defaults.adapter = makeAdapter()
})

afterEach(() => {
  cleanup()
  window.localStorage.clear()
  vi.restoreAllMocks()
})

describe('authentication and protected routes', () => {
  it.each(['/dashboard', '/patients'] as const)('redirects logged-out users from %s to login', async (path) => {
    renderAt(path)
    expect(await screen.findByRole('heading', { name: 'Sign in to your workspace' })).toBeTruthy()
  })

  it('validates blank credentials without making a request', async () => {
    const post = vi.spyOn(apiClient, 'post')
    renderAt('/login')
    await userEvent.click(await screen.findByRole('button', { name: 'Sign in' }))
    expect(screen.getByText('Enter both your username and password.')).toBeTruthy()
    expect(post).not.toHaveBeenCalled()
  })

  it('shows a useful message for invalid credentials', async () => {
    const user = userEvent.setup()
    renderAt('/login')
    await user.type(await screen.findByLabelText('Username'), 'unknown')
    await user.type(screen.getByLabelText('Password'), 'incorrect')
    await user.click(screen.getByRole('button', { name: 'Sign in' }))
    expect(await screen.findByText('The username or password was not accepted.')).toBeTruthy()
    expect(window.localStorage.getItem(TOKEN_KEY)).toBeNull()
  })

  it('logs in, stores the bearer token, and redirects to dashboard', async () => {
    const user = userEvent.setup()
    renderAt('/login')
    await user.type(await screen.findByLabelText('Username'), 'clinician-one')
    await user.type(screen.getByLabelText('Password'), 'test-password')
    await user.click(screen.getByRole('button', { name: 'Sign in' }))

    expect(await screen.findByRole('heading', { name: 'Clinical Decision Support Dashboard' })).toBeTruthy()
    expect(window.localStorage.getItem(TOKEN_KEY)).toBe('demo-token')
    expect(screen.getByText('clinician@example.test')).toBeTruthy()
    expect(screen.getByText('clinician', { selector: '.role-badge' })).toBeTruthy()
  })

  it('restores the session on refresh using /auth/me', async () => {
    window.localStorage.setItem(TOKEN_KEY, 'demo-token')
    renderAt('/dashboard')
    expect(await screen.findByRole('heading', { name: 'Clinical Decision Support Dashboard' })).toBeTruthy()
    expect(screen.getByText('clinician-one')).toBeTruthy()
  })

  it('logs out, clears storage, and redirects to login', async () => {
    window.localStorage.setItem(TOKEN_KEY, 'demo-token')
    const user = userEvent.setup()
    renderAt('/dashboard')
    await screen.findByRole('heading', { name: 'Clinical Decision Support Dashboard' })
    await user.click(screen.getByRole('button', { name: 'Logout' }))
    expect(await screen.findByRole('heading', { name: 'Sign in to your workspace' })).toBeTruthy()
    expect(window.localStorage.getItem(TOKEN_KEY)).toBeNull()
  })

  it('adds the stored bearer token to authenticated API requests', async () => {
    window.localStorage.setItem(TOKEN_KEY, 'demo-token')
    const { data } = await apiClient.get('/auth/me')
    expect(data).toEqual(demoUser)
  })

  it('clears an expired token and redirects out of the protected shell', async () => {
    window.localStorage.setItem(TOKEN_KEY, 'expired-token')
    renderAt('/patients')
    expect(await screen.findByRole('heading', { name: 'Sign in to your workspace' })).toBeTruthy()
    await waitFor(() => expect(window.localStorage.getItem(TOKEN_KEY)).toBeNull())
  })

  it('redirects authenticated users away from login', async () => {
    window.localStorage.setItem(TOKEN_KEY, 'demo-token')
    renderAt('/login')
    expect(await screen.findByRole('heading', { name: 'Clinical Decision Support Dashboard' })).toBeTruthy()
    expect(screen.queryByRole('heading', { name: 'Sign in to your workspace' })).toBeNull()
  })
})
