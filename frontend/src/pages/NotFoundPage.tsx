import { Link } from 'react-router-dom'

export default function NotFoundPage() {
  return (
    <main className="not-found">
      <p className="eyebrow">Page not found</p>
      <h1>This route is not part of the current prototype.</h1>
      <Link to="/dashboard">Return to dashboard</Link>
    </main>
  )
}
