import { useState, type FormEvent } from 'react'
import { isAxiosError } from 'axios'
import { useLocation, useNavigate } from 'react-router-dom'
import { ErrorState } from '../components/StatePanels'
import { useAuth } from '../context/AuthContext'

export default function LoginPage() {
  const { login, isLoading } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [errorMessage, setErrorMessage] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setErrorMessage('')
    if (!username.trim() || !password) {
      setErrorMessage('Enter both your username and password.')
      return
    }

    setIsSubmitting(true)
    try {
      await login(username.trim(), password)
      const state = location.state as { from?: { pathname?: string } } | null
      navigate(state?.from?.pathname || '/dashboard', { replace: true })
    } catch (error: unknown) {
      if (isAxiosError(error) && error.response?.status === 401) {
        setErrorMessage('The username or password was not accepted.')
      } else if (isAxiosError(error) && !error.response) {
        setErrorMessage('The sign-in service could not be reached. Check the connection and try again.')
      } else if (isAxiosError(error) && error.response?.status === 403) {
        setErrorMessage('This account is inactive. Contact the system administrator.')
      } else {
        setErrorMessage('Sign-in could not be completed. Please try again.')
      }
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <main className="login-page">
      <section className="login-card" aria-labelledby="login-title">
        <div className="brand-mark" aria-hidden="true">X</div>
        <p className="eyebrow">Clinician access · research prototype</p>
        <h1 id="login-title">Sign in to your workspace</h1>
        <p className="login-intro">Use your provisioned account to access the experimental decision-support workspace.</p>
        <form className="login-form" onSubmit={handleSubmit} noValidate>
          <div className="form-field">
            <label htmlFor="username">Username</label>
            <input
              id="username"
              name="username"
              type="text"
              autoComplete="username"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              aria-describedby={errorMessage ? 'login-error' : undefined}
              required
            />
          </div>
          <div className="form-field">
            <label htmlFor="password">Password</label>
            <input
              id="password"
              name="password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              aria-describedby={errorMessage ? 'login-error' : undefined}
              required
            />
          </div>
          {errorMessage && <div id="login-error"><ErrorState title="Sign-in unsuccessful">{errorMessage}</ErrorState></div>}
          <button className="login-submit" type="submit" disabled={isSubmitting || isLoading}>
            {isSubmitting ? 'Signing in…' : 'Sign in'}
          </button>
        </form>
        <p className="login-security-note">This research prototype stores its access token in browser storage. Do not use sensitive accounts or credentials.</p>
      </section>
    </main>
  )
}
