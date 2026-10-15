interface StatePanelProps {
  title: string
  children: string
}

export function LoadingState({ title = 'Loading' }: { title?: string }) {
  return <p className="state-panel" role="status">{title}…</p>
}

export function ErrorState({ title = 'Something went wrong', children }: StatePanelProps) {
  return <section className="state-panel state-panel--error" role="alert"><h2>{title}</h2><p>{children}</p></section>
}

export function EmptyState({ title, children }: StatePanelProps) {
  return <section className="state-panel"><h2>{title}</h2><p>{children}</p></section>
}
