interface PageHeadingProps {
  eyebrow?: string
  title: string
  description: string
}

export default function PageHeading({ eyebrow, title, description }: PageHeadingProps) {
  return (
    <header className="page-heading">
      {eyebrow && <p className="eyebrow">{eyebrow}</p>}
      <h1>{title}</h1>
      <p className="page-description">{description}</p>
    </header>
  )
}
