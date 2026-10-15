import { Link, NavLink, Outlet, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function AppLayout() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  function handleLogout() {
    logout()
    navigate('/login', { replace: true })
  }

  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="Main navigation">
        <Link className="brand" to="/dashboard" aria-label="XAI CDSS home">
          <span className="brand-mark" aria-hidden="true">X</span>
          <span><strong>XAI CDSS</strong><small>Research prototype</small></span>
        </Link>
        <nav className="primary-nav">
          <NavLink to="/dashboard" end className={({ isActive }) => `nav-link${isActive ? ' is-active' : ''}`}>
            <span aria-hidden="true">D</span> Dashboard
          </NavLink>
          <NavLink to="/patients" className={({ isActive }) => `nav-link${isActive ? ' is-active' : ''}`}>
            <span aria-hidden="true">P</span> Patients
          </NavLink>
        </nav>
        <div className="sidebar-note">
          <span className="status-dot" aria-hidden="true" />
          <p>Academic decision-support prototype</p>
        </div>
      </aside>

      <div className="workspace">
        <header className="topbar">
          <p>Explainable AI <span aria-hidden="true">/</span> Clinical research</p>
          <div className="account-controls">
            <div className="account-identity">
              <strong>{user?.username}</strong>
              <span>{user?.email}</span>
            </div>
            <span className="role-badge">{user?.role}</span>
            <button className="logout-button" type="button" onClick={handleLogout}>Logout</button>
          </div>
        </header>
        <main className="main-content" id="main-content" tabIndex={-1}>
          <Outlet />
          <footer className="disclaimer">
            Model output is experimental and supports research exploration only. It is not a diagnosis or treatment recommendation.
          </footer>
        </main>
      </div>
    </div>
  )
}
