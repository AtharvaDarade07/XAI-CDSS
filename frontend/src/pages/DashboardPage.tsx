import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import axios from 'axios'
import PageHeading from '../components/PageHeading'
import { EmptyState, ErrorState, LoadingState } from '../components/StatePanels'
import { listPatients } from '../services/patientService'
import type { PatientListResponse } from '../types/api'
import { formatDate } from './PatientsPage'

const DASHBOARD_PATIENT_LIMIT = 5

function dashboardError(error: unknown): string {
  if (axios.isAxiosError(error) && error.response?.status === 403) return 'Your account does not have permission to view patient information.'
  if (axios.isAxiosError(error) && error.response?.status === 401) return 'Your session has expired. Sign in again to continue.'
  return 'Patient overview could not be loaded. Please try again.'
}

export default function DashboardPage() {
  const [patients, setPatients] = useState<PatientListResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const loadOverview = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      setPatients(await listPatients(0, DASHBOARD_PATIENT_LIMIT))
    } catch (cause) {
      setError(dashboardError(cause))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { void loadOverview() }, [loadOverview])

  return <>
    <PageHeading
      eyebrow="Workspace"
      title="Clinical Decision Support Dashboard"
      description="Overview of patient records, model predictions, and explainability activity."
    />

    <section className="dashboard-quick-actions" aria-label="Quick actions">
      <div><h2>Continue working</h2><p>Open the patient directory to view records or add a patient.</p></div>
      <div className="dashboard-action-links">
        <Link className="secondary-button dashboard-link-button" to="/patients">View Patients</Link>
        <Link className="primary-button dashboard-link-button" to="/patients?action=add">Add Patient</Link>
      </div>
    </section>

    {loading ? <LoadingState title="Loading dashboard overview" /> : error ? <>
      <ErrorState title="Dashboard overview unavailable">{error}</ErrorState>
      <button className="secondary-button" type="button" onClick={() => void loadOverview()}>Retry</button>
    </> : patients && <>
      <section className="dashboard-summary-grid" aria-label="System overview">
        <article className="dashboard-summary-card">
          <span>Total patients</span>
          <strong>{patients.total}</strong>
          <small>Reported by the patient API</small>
        </article>
        <article className="dashboard-summary-card">
          <span>Patients loaded</span>
          <strong>{patients.items.length}</strong>
          <small>First page shown below</small>
        </article>
        <article className="dashboard-summary-card dashboard-model-card">
          <span>Model information</span>
          <strong>Available with each prediction</strong>
          <small>No active-model summary endpoint is available</small>
        </article>
      </section>

      <section className="dashboard-patients-section" aria-labelledby="dashboard-patients-title">
        <header className="dashboard-section-heading">
          <div>
            <p className="eyebrow">Patient directory</p>
            <h2 id="dashboard-patients-title">Patients</h2>
            <p>Showing the first page in the order returned by the patient API.</p>
          </div>
          <Link className="text-action" to="/patients">Open patient directory</Link>
        </header>
        {patients.items.length === 0 ? <EmptyState title="No patients have been added yet.">Add a patient to begin a research record.</EmptyState> : <div className="table-card"><div className="table-scroll">
          <table className="data-table dashboard-patient-table">
            <caption className="sr-only">Patients on the first page of the patient directory</caption>
            <thead><tr><th scope="col">Patient code</th><th scope="col">Patient ID</th><th scope="col">Created</th><th scope="col"><span className="sr-only">Actions</span></th></tr></thead>
            <tbody>{patients.items.map((patient) => <tr key={patient.id}>
              <td>{patient.patient_code}</td>
              <td>{patient.id}</td>
              <td>{formatDate(patient.created_at)}</td>
              <td><Link className="text-action" to={`/patients/${patient.id}`}>View Patient</Link></td>
            </tr>)}</tbody>
          </table>
        </div></div>}
      </section>

      <aside className="dashboard-responsible-notice" aria-label="Responsible use notice">
        <strong>Decision-support only</strong>
        <p>Predictions and explanations describe model behavior and are not a diagnosis or treatment recommendation. Clinical judgment and appropriate validation remain necessary.</p>
      </aside>
    </>}
  </>
}
