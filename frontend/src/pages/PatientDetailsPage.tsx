import { useCallback, useEffect, useState, type FormEvent } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import axios from 'axios'
import PageHeading from '../components/PageHeading'
import { EmptyState, ErrorState, LoadingState } from '../components/StatePanels'
import { createPatientRecord, getPatient, listPatientRecords } from '../services/patientService'
import type { Patient, PatientRecord, PatientRecordCreateRequest } from '../types/api'
import { formatDate } from './PatientsPage'

const RECORD_PAGE_SIZE = 20
const inputs = [
  { name: 'pregnancies', label: 'Pregnancies', min: 0, max: 17, step: 1, integer: true },
  { name: 'glucose', label: 'Glucose', min: 0, max: 199, step: 0.1 },
  { name: 'blood_pressure', label: 'Blood pressure', min: 0, max: 122, step: 0.1 },
  { name: 'skin_thickness', label: 'Skin thickness', min: 0, max: 99, step: 0.1 },
  { name: 'insulin', label: 'Insulin', min: 0, max: 846, step: 0.1 },
  { name: 'bmi', label: 'BMI', min: 0, max: 67.1, step: 0.1 },
  { name: 'diabetes_pedigree_function', label: 'Diabetes pedigree function', min: 0.078, max: 2.42, step: 0.001 },
  { name: 'age', label: 'Age', min: 21, max: 81, step: 1, integer: true },
] as const

type InputName = typeof inputs[number]['name']
type FormValues = Record<InputName, string>
const blankValues = (): FormValues => Object.fromEntries(inputs.map(({ name }) => [name, ''])) as FormValues

function explainError(error: unknown): string {
  if (axios.isAxiosError(error) && error.response?.status === 404) return 'The requested patient could not be found.'
  if (axios.isAxiosError(error) && error.response?.status === 401) return 'Your session has expired. Sign in again to continue.'
  return 'Patient information could not be loaded or saved. Please try again.'
}

function toPayload(values: FormValues): PatientRecordCreateRequest {
  const payload = {} as PatientRecordCreateRequest
  for (const field of inputs) {
    const raw = values[field.name]
    payload[field.name] = raw.trim() === '' ? null : Number(raw)
  }
  return payload
}

export default function PatientDetailsPage() {
  const { patientId: patientIdParam } = useParams()
  const patientId = Number(patientIdParam)
  const navigate = useNavigate()
  const validId = Number.isInteger(patientId) && patientId > 0
  const [patient, setPatient] = useState<Patient | null>(null)
  const [records, setRecords] = useState<PatientRecord[]>([])
  const [skip, setSkip] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [showForm, setShowForm] = useState(false)
  const [values, setValues] = useState<FormValues>(blankValues)
  const [formError, setFormError] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const load = useCallback(async (offset: number) => {
    if (!validId) { setError('The patient identifier is invalid.'); setLoading(false); return }
    setLoading(true)
    setError('')
    try {
      const [patientResult, recordResult] = await Promise.all([getPatient(patientId), listPatientRecords(patientId, offset, RECORD_PAGE_SIZE)])
      setPatient(patientResult)
      setRecords(recordResult)
    } catch (cause) { setError(explainError(cause)) }
    finally { setLoading(false) }
  }, [patientId, validId])

  useEffect(() => { void load(skip) }, [load, skip])

  function validate(valuesToCheck: FormValues): string {
    for (const field of inputs) {
      const raw = valuesToCheck[field.name].trim()
      if (!raw) continue
      const value = Number(raw)
      if (!Number.isFinite(value)) return `${field.label} must be a valid number.`
      if (value < field.min || value > field.max) return `${field.label} must be between ${field.min} and ${field.max} for this project's dataset validation.`
      if ('integer' in field && field.integer && !Number.isInteger(value)) return `${field.label} must be a whole number.`
    }
    return ''
  }

  async function submitRecord(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const validation = validate(values)
    if (validation) { setFormError(validation); return }
    setFormError('')
    setSubmitting(true)
    try {
      const record = await createPatientRecord(patientId, toPayload(values))
      setValues(blankValues())
      setShowForm(false)
      // Show the server-created row directly, including when older records fill the first page.
      navigate(`/patients/${patientId}/records/${record.id}`)
    } catch (cause) { setFormError(explainError(cause)) }
    finally { setSubmitting(false) }
  }

  return <>
    <div className="breadcrumbs"><Link to="/patients">Patients</Link><span aria-hidden="true">/</span><span>Patient details</span></div>
    {loading ? <LoadingState title="Loading patient…" /> : error || !patient ? <ErrorState title="Patient unavailable">{error || 'The requested patient could not be found.'}</ErrorState> : <>
      <PageHeading eyebrow="Patient workspace" title={patient.patient_code} description={`Patient ID ${patient.id} · Created ${formatDate(patient.created_at)} · Updated ${formatDate(patient.updated_at)}`} />
      <section className="patient-toolbar"><div><h2 className="section-title">Clinical records</h2><p>Stored measurements as entered. Dataset validation ranges are project-specific and are not universal clinical reference ranges.</p></div>
        <button className="primary-button" type="button" onClick={() => { setShowForm((open) => !open); setFormError('') }}>{showForm ? 'Cancel' : 'Add Clinical Record'}</button>
      </section>

      {showForm && <section className="content-card record-form-card" aria-labelledby="add-record-title">
        <h2 id="add-record-title">Add clinical record</h2>
        <p className="muted-copy">Fields may be left blank. Zero values are stored as entered. Bounds reflect this project's dataset schema, not universal clinical ranges.</p>
        <form className="data-form record-form" onSubmit={submitRecord} noValidate>
          {inputs.map((field) => <div className="form-field" key={field.name}>
            <label htmlFor={`record-${field.name}`}>{field.label}</label>
            <input id={`record-${field.name}`} name={field.name} type="number" min={field.min} max={field.max} step={field.step} value={values[field.name]} onChange={(event) => setValues((previous) => ({ ...previous, [field.name]: event.target.value }))} />
          </div>)}
          {formError && <p className="inline-error form-wide" role="alert">{formError}</p>}
          <div className="form-actions form-wide"><button className="primary-button" type="submit" disabled={submitting}>{submitting ? 'Saving…' : 'Save clinical record'}</button></div>
        </form>
      </section>}

      {records.length === 0 ? <EmptyState title="No clinical records">Add a clinical record to store measurements for this patient.</EmptyState> : <>
        <div className="table-card"><div className="table-scroll"><table className="data-table">
          <caption className="sr-only">Clinical records for {patient.patient_code}</caption>
          <thead><tr><th scope="col">Record</th><th scope="col">Created</th>{inputs.map((field) => <th scope="col" key={field.name}>{field.label}</th>)}<th scope="col"><span className="sr-only">Actions</span></th></tr></thead>
          <tbody>{records.map((record) => <tr key={record.id}><td>{record.id}</td><td>{formatDate(record.created_at)}</td>{inputs.map((field) => <td key={field.name}>{record[field.name] ?? '—'}</td>)}<td><Link className="text-action" to={`/patients/${patient.id}/records/${record.id}`}>View record</Link></td></tr>)}</tbody>
        </table></div></div>
        <nav className="pagination" aria-label="Patient record pages"><span>Showing {skip + 1}–{skip + records.length} records</span><div>
          <button className="secondary-button" type="button" disabled={skip === 0} onClick={() => setSkip(Math.max(0, skip - RECORD_PAGE_SIZE))}>Previous</button>
          <button className="secondary-button" type="button" disabled={records.length < RECORD_PAGE_SIZE} onClick={() => setSkip(skip + RECORD_PAGE_SIZE)}>Next</button>
        </div></nav>
      </>}
    </>}
  </>
}
