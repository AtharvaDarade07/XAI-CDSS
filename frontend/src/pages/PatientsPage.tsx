import { useCallback, useEffect, useState, type FormEvent } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import axios from 'axios'
import PageHeading from '../components/PageHeading'
import { EmptyState, ErrorState, LoadingState } from '../components/StatePanels'
import { createPatient, listPatients } from '../services/patientService'
import type { Patient, PatientListResponse } from '../types/api'

const PAGE_SIZE = 20

function patientError(error: unknown): string {
  if (axios.isAxiosError(error) && error.response?.status === 409) return 'That patient code is already in use. Choose a different code.'
  if (axios.isAxiosError(error) && error.response?.status === 401) return 'Your session has expired. Sign in again to continue.'
  return 'Patients could not be loaded or saved. Please try again.'
}

export default function PatientsPage() {
  const [skip, setSkip] = useState(0)
  const [data, setData] = useState<PatientListResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [searchParams, setSearchParams] = useSearchParams()
  const [showForm, setShowForm] = useState(() => searchParams.get('action') === 'add')
  const [patientCode, setPatientCode] = useState('')
  const [formError, setFormError] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const navigate = useNavigate()

  const loadPatients = useCallback(async (offset: number) => {
    setLoading(true)
    setError('')
    try {
      setData(await listPatients(offset, PAGE_SIZE))
    } catch (cause) {
      setError(patientError(cause))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { void loadPatients(skip) }, [loadPatients, skip])

  async function submitPatient(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setFormError('')
    if (!patientCode.trim()) {
      setFormError('Patient code is required and cannot contain only spaces.')
      return
    }
    if (patientCode.length > 100) {
      setFormError('Patient code must be 100 characters or fewer.')
      return
    }
    setSubmitting(true)
    try {
      const patient: Patient = await createPatient({ patient_code: patientCode })
      setShowForm(false)
      setPatientCode('')
      navigate(`/patients/${patient.id}`)
    } catch (cause) {
      setFormError(patientError(cause))
    } finally {
      setSubmitting(false)
    }
  }

  function togglePatientForm() {
    setShowForm((open) => !open)
    setFormError('')
    if (searchParams.has('action')) {
      const nextParams = new URLSearchParams(searchParams)
      nextParams.delete('action')
      setSearchParams(nextParams, { replace: true })
    }
  }

  return (
    <>
      <PageHeading eyebrow="Workspace" title="Patients" description="Browse privacy-minimal patient records and their stored clinical data." />
      <section className="patient-toolbar" aria-label="Patient actions">
        <p>{data ? `${data.total} patient${data.total === 1 ? '' : 's'} in this workspace` : 'Patient directory'}</p>
        <button className="primary-button" type="button" onClick={togglePatientForm}>
          {showForm ? 'Cancel' : 'Add Patient'}
        </button>
      </section>

      {showForm && <section className="content-card patient-form-card" aria-labelledby="create-patient-title">
        <h2 id="create-patient-title">Add a patient</h2>
        <p className="muted-copy">Use a project-level code. Do not enter personally identifying information.</p>
        <form className="data-form compact-form" onSubmit={submitPatient} noValidate>
          <div className="form-field">
            <label htmlFor="patient-code">Patient code</label>
            <input id="patient-code" name="patient_code" value={patientCode} maxLength={100} onChange={(event) => setPatientCode(event.target.value)} aria-describedby={formError ? 'patient-code-error' : undefined} autoFocus />
          </div>
          {formError && <p className="inline-error" id="patient-code-error" role="alert">{formError}</p>}
          <button className="primary-button" type="submit" disabled={submitting}>{submitting ? 'Saving…' : 'Create patient'}</button>
        </form>
      </section>}

      {loading ? <LoadingState title="Loading patients…" /> : error ? <ErrorState title="Patient list unavailable">{error}</ErrorState> : !data || data.items.length === 0 ? (
        <EmptyState title="No patients yet">Add a patient code to begin a research record.</EmptyState>
      ) : (
        <>
          <div className="table-card">
            <div className="table-scroll">
              <table className="data-table">
                <caption className="sr-only">Patient directory</caption>
                <thead><tr><th scope="col">Patient code</th><th scope="col">Patient ID</th><th scope="col">Created</th><th scope="col">Updated</th><th scope="col"><span className="sr-only">Actions</span></th></tr></thead>
                <tbody>{data.items.map((patient) => <tr key={patient.id}>
                  <td><Link className="table-primary-link" to={`/patients/${patient.id}`}>{patient.patient_code}</Link></td>
                  <td>{patient.id}</td><td>{formatDate(patient.created_at)}</td><td>{formatDate(patient.updated_at)}</td>
                  <td><Link className="text-action" to={`/patients/${patient.id}`}>View patient</Link></td>
                </tr>)}</tbody>
              </table>
            </div>
          </div>
          <nav className="pagination" aria-label="Patient list pages">
            <span>Showing {skip + 1}–{skip + data.items.length} of {data.total}</span>
            <div>
              <button type="button" className="secondary-button" disabled={skip === 0} onClick={() => setSkip(Math.max(0, skip - PAGE_SIZE))}>Previous</button>
              <button type="button" className="secondary-button" disabled={skip + data.items.length >= data.total} onClick={() => setSkip(skip + PAGE_SIZE)}>Next</button>
            </div>
          </nav>
        </>
      )}
    </>
  )
}

export function formatDate(value: string): string {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString()
}
