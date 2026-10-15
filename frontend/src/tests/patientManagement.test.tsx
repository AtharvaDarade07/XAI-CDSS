import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { AxiosAdapter } from 'axios'
import { MemoryRouter } from 'react-router-dom'
import App from '../App'
import { AuthProvider } from '../context/AuthContext'
import { apiClient } from '../services/api'
import type { Patient, PatientListResponse, PatientRecord } from '../types/api'

const mocks = vi.hoisted(() => ({
  listPatients: vi.fn(), getPatient: vi.fn(), createPatient: vi.fn(),
  listPatientRecords: vi.fn(), getPatientRecord: vi.fn(), createPatientRecord: vi.fn(),
}))

vi.mock('../services/patientService', () => mocks)

const tokenKey = 'xai-cdss.access-token'
const userAccount = { id: 17, username: 'test-clinician', email: 'clinician@example.test', role: 'clinician', is_active: true }
const patientOne: Patient = { id: 8, patient_code: 'RESEARCH-008', created_at: '2026-02-01T10:00:00Z', updated_at: '2026-02-01T10:00:00Z' }
const emptyList: PatientListResponse = { items: [], skip: 0, limit: 20, total: 0 }
const recordOne: PatientRecord = {
  id: 13, patient_id: 8, pregnancies: 0, glucose: 0, blood_pressure: 72, skin_thickness: 35,
  insulin: 0, bmi: 33.6, diabetes_pedigree_function: 0.627, age: 50, created_at: '2026-02-02T11:30:00Z',
}

function authAdapter(): AxiosAdapter {
  return async (config) => {
    const response = { data: userAccount, status: 200, statusText: 'OK', headers: {}, config }
    return response
  }
}

function renderAt(path: string, authenticated = true) {
  if (authenticated) window.localStorage.setItem(tokenKey, 'test-bearer')
  return render(<MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }} initialEntries={[path]}>
    <AuthProvider><App /></AuthProvider>
  </MemoryRouter>)
}

beforeEach(() => {
  window.localStorage.clear()
  apiClient.defaults.adapter = authAdapter()
  mocks.listPatients.mockResolvedValue(emptyList)
  mocks.getPatient.mockResolvedValue(patientOne)
  mocks.createPatient.mockResolvedValue(patientOne)
  mocks.listPatientRecords.mockResolvedValue([])
  mocks.getPatientRecord.mockResolvedValue(recordOne)
  mocks.createPatientRecord.mockResolvedValue(recordOne)
})

afterEach(() => {
  cleanup()
  window.localStorage.clear()
  vi.clearAllMocks()
})

describe('patient management UI', () => {
  it('loads and displays patients from the patient service', async () => {
    mocks.listPatients.mockResolvedValue({ items: [patientOne], skip: 0, limit: 20, total: 1 })
    renderAt('/patients')
    expect(await screen.findByText('RESEARCH-008')).toBeTruthy()
    expect(mocks.listPatients).toHaveBeenCalledWith(0, 20)
    expect(screen.getByRole('link', { name: 'View patient' })).toBeTruthy()
  })

  it('shows a useful error state when patient list loading fails', async () => {
    mocks.listPatients.mockRejectedValue(new Error('private server internals'))
    renderAt('/patients')
    expect(await screen.findByText('Patients could not be loaded or saved. Please try again.')).toBeTruthy()
    expect(screen.queryByText('private server internals')).toBeNull()
  })

  it('uses backend skip/limit pagination and disables unavailable directions', async () => {
    const second: Patient = { ...patientOne, id: 9, patient_code: 'RESEARCH-009' }
    mocks.listPatients
      .mockResolvedValueOnce({ items: [patientOne], skip: 0, limit: 1, total: 2 })
      .mockResolvedValueOnce({ items: [second], skip: 1, limit: 1, total: 2 })
    renderAt('/patients')
    const next = await screen.findByRole('button', { name: 'Next' })
    expect(screen.getByRole('button', { name: 'Previous' }).hasAttribute('disabled')).toBe(true)
    await userEvent.click(next)
    expect(await screen.findByText('RESEARCH-009')).toBeTruthy()
    expect(mocks.listPatients).toHaveBeenLastCalledWith(20, 20)
  })

  it('validates patient code before submitting', async () => {
    const user = userEvent.setup()
    renderAt('/patients')
    await user.click(await screen.findByRole('button', { name: 'Add Patient' }))
    await user.click(screen.getByRole('button', { name: 'Create patient' }))
    expect(await screen.findByText('Patient code is required and cannot contain only spaces.')).toBeTruthy()
    expect(mocks.createPatient).not.toHaveBeenCalled()
  })

  it('creates patient then navigates to and loads its detail', async () => {
    const user = userEvent.setup()
    renderAt('/patients')
    await user.click(await screen.findByRole('button', { name: 'Add Patient' }))
    await user.type(screen.getByLabelText('Patient code'), 'RESEARCH-008')
    await user.click(screen.getByRole('button', { name: 'Create patient' }))
    expect(await screen.findByRole('heading', { name: 'RESEARCH-008' })).toBeTruthy()
    expect(mocks.createPatient).toHaveBeenCalledWith({ patient_code: 'RESEARCH-008' })
    expect(mocks.getPatient).toHaveBeenCalledWith(8)
    expect(mocks.listPatientRecords).toHaveBeenCalledWith(8, 0, 20)
  })

  it('shows a clean duplicate patient code error', async () => {
    const user = userEvent.setup()
    mocks.createPatient.mockRejectedValue(Object.assign(new Error('conflict'), { isAxiosError: true, response: { status: 409 } }))
    renderAt('/patients')
    await user.click(await screen.findByRole('button', { name: 'Add Patient' }))
    await user.type(screen.getByLabelText('Patient code'), 'RESEARCH-008')
    await user.click(screen.getByRole('button', { name: 'Create patient' }))
    expect(await screen.findByText('That patient code is already in use. Choose a different code.')).toBeTruthy()
  })

  it('loads the selected patient and its records', async () => {
    mocks.listPatientRecords.mockResolvedValue([recordOne])
    renderAt('/patients/8')
    expect(await screen.findByRole('heading', { name: 'RESEARCH-008' })).toBeTruthy()
    expect(screen.getByText('Clinical records')).toBeTruthy()
    expect(screen.getByRole('link', { name: 'View record' })).toBeTruthy()
    expect(mocks.getPatient).toHaveBeenCalledWith(8)
    expect(mocks.listPatientRecords).toHaveBeenCalledWith(8, 0, 20)
  })

  it('shows an actionable empty state when a patient has no clinical records', async () => {
    mocks.listPatientRecords.mockResolvedValue([])
    renderAt('/patients/8')
    expect(await screen.findByRole('heading', { name: 'No clinical records' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Add Clinical Record' })).toBeTruthy()
  })

  it('opens an existing clinical record from patient details', async () => {
    const user = userEvent.setup()
    mocks.listPatientRecords.mockResolvedValue([recordOne])
    renderAt('/patients/8')
    await user.click(await screen.findByRole('link', { name: 'View record' }))
    expect(await screen.findByRole('heading', { name: 'Record 13' })).toBeTruthy()
    expect(screen.getByRole('link', { name: 'Back to Patient' })).toBeTruthy()
    expect(screen.getByRole('link', { name: 'Back to Patients' })).toBeTruthy()
  })

  it('validates record fields against project boundaries', async () => {
    const user = userEvent.setup()
    renderAt('/patients/8')
    await user.click(await screen.findByRole('button', { name: 'Add Clinical Record' }))
    await user.type(screen.getByLabelText('Glucose'), '200')
    await user.click(screen.getByRole('button', { name: 'Save clinical record' }))
    expect(await screen.findByText("Glucose must be between 0 and 199 for this project's dataset validation.")).toBeTruthy()
    expect(mocks.createPatientRecord).not.toHaveBeenCalled()
  })

  it('creates a record and preserves accepted zero values', async () => {
    const user = userEvent.setup()
    mocks.listPatientRecords.mockResolvedValue([recordOne])
    renderAt('/patients/8')
    await user.click(await screen.findByRole('button', { name: 'Add Clinical Record' }))
    await user.type(screen.getByLabelText('Pregnancies'), '0')
    await user.type(screen.getByLabelText('Glucose'), '0')
    await user.type(screen.getByLabelText('Insulin'), '0')
    await user.click(screen.getByRole('button', { name: 'Save clinical record' }))
    await waitFor(() => expect(mocks.createPatientRecord).toHaveBeenCalled())
    expect(mocks.createPatientRecord).toHaveBeenCalledWith(8, expect.objectContaining({ pregnancies: 0, glucose: 0, insulin: 0 }))
    expect(await screen.findByRole('heading', { name: 'Record 13' })).toBeTruthy()
  })

  it('shows the selected stored record and patient context', async () => {
    renderAt('/patients/8/records/13')
    expect(await screen.findByRole('heading', { name: 'Record 13' })).toBeTruthy()
    expect(screen.getByText('RESEARCH-008 · Patient ID 8', { exact: false })).toBeTruthy()
    const values = screen.getAllByText('0')
    expect(values.length).toBeGreaterThan(0)
    expect(screen.getByRole('heading', { name: 'Prediction' })).toBeTruthy()
    expect(screen.getByRole('heading', { name: 'Generate SHAP Explanation' })).toBeTruthy()
    expect((screen.getByRole('button', { name: 'Generate SHAP Explanation' }) as HTMLButtonElement).disabled).toBe(true)
    expect(mocks.getPatientRecord).toHaveBeenCalledWith(8, 13)
  })

  it('keeps protected patient routes behind authentication', async () => {
    renderAt('/patients', false)
    expect(await screen.findByRole('heading', { name: 'Sign in to your workspace' })).toBeTruthy()
    expect(mocks.listPatients).not.toHaveBeenCalled()
  })
})

describe('clinician dashboard', () => {
  it('renders the authenticated dashboard and fetches the first patient page', async () => {
    mocks.listPatients.mockResolvedValue({ items: [patientOne], skip: 0, limit: 5, total: 37 })
    renderAt('/dashboard')
    expect(await screen.findByRole('heading', { name: 'Clinical Decision Support Dashboard' })).toBeTruthy()
    expect(mocks.listPatients).toHaveBeenCalledWith(0, 5)
    expect(screen.getByText('RESEARCH-008')).toBeTruthy()
  })

  it('shows the API total and loaded count without presenting loaded patients as the total', async () => {
    mocks.listPatients.mockResolvedValue({ items: [patientOne], skip: 0, limit: 5, total: 137 })
    renderAt('/dashboard')
    await screen.findByText('RESEARCH-008')
    const totalCard = screen.getByText('Total patients').parentElement
    const loadedCard = screen.getByText('Patients loaded').parentElement
    expect(totalCard?.textContent).toContain('137')
    expect(loadedCard?.textContent).toContain('1')
    expect(screen.queryByText('Predictions available')).toBeNull()
  })

  it('does not show zero summary values while dashboard data is loading', async () => {
    let resolveList: (data: PatientListResponse) => void = () => undefined
    mocks.listPatients.mockReturnValue(new Promise<PatientListResponse>((resolve) => { resolveList = resolve }))
    renderAt('/dashboard')
    await screen.findByText('Continue working')
    expect(screen.getByRole('status').textContent).toContain('Loading dashboard overview')
    expect(screen.queryByText('Total patients')).toBeNull()
    await act(async () => { resolveList({ items: [patientOne], skip: 0, limit: 5, total: 1 }) })
    expect(await screen.findByText('Total patients')).toBeTruthy()
  })

  it('shows the empty patient state and provides an Add Patient route', async () => {
    const user = userEvent.setup()
    mocks.listPatients.mockResolvedValue(emptyList)
    renderAt('/dashboard')
    expect(await screen.findByText('No patients have been added yet.')).toBeTruthy()
    await user.click(screen.getByRole('link', { name: 'Add Patient' }))
    expect(await screen.findByRole('heading', { name: 'Patients' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeTruthy()
    expect(screen.getByRole('heading', { name: 'Add a patient' })).toBeTruthy()
    expect(screen.getByLabelText('Patient code')).toBeTruthy()
  })

  it('shows a recoverable error and retries the patient request', async () => {
    const user = userEvent.setup()
    mocks.listPatients.mockRejectedValueOnce(new Error('private backend details')).mockResolvedValueOnce({ items: [patientOne], skip: 0, limit: 5, total: 1 })
    renderAt('/dashboard')
    expect(await screen.findByText('Patient overview could not be loaded. Please try again.')).toBeTruthy()
    expect(screen.queryByText('private backend details')).toBeNull()
    await user.click(screen.getByRole('button', { name: 'Retry' }))
    expect(await screen.findByText('RESEARCH-008')).toBeTruthy()
    expect(mocks.listPatients).toHaveBeenCalledTimes(2)
  })

  it('navigates to the selected patient details', async () => {
    const user = userEvent.setup()
    mocks.listPatients.mockResolvedValue({ items: [patientOne], skip: 0, limit: 5, total: 1 })
    mocks.listPatientRecords.mockResolvedValue([recordOne])
    renderAt('/dashboard')
    await user.click(await screen.findByRole('link', { name: 'View Patient' }))
    expect(await screen.findByRole('heading', { name: 'RESEARCH-008' })).toBeTruthy()
    expect(mocks.getPatient).toHaveBeenCalledWith(patientOne.id)
  })

  it('opens the patient directory from the dashboard navigation action', async () => {
    const user = userEvent.setup()
    mocks.listPatients.mockResolvedValue(emptyList)
    renderAt('/dashboard')
    await user.click(await screen.findByRole('link', { name: 'Open patient directory' }))
    expect(await screen.findByRole('heading', { name: 'Patients' })).toBeTruthy()
    expect(screen.getByRole('heading', { name: 'No patients yet' })).toBeTruthy()
  })

  it('uses generic model information rather than inventing a version or metrics', async () => {
    mocks.listPatients.mockResolvedValue(emptyList)
    renderAt('/dashboard')
    await screen.findByText('Model information')
    expect(screen.getByText('Available with each prediction')).toBeTruthy()
    expect(screen.queryByText(/xgboost-baseline-v1/)).toBeNull()
    expect(screen.queryByText(/accuracy|AUC|precision/i)).toBeNull()
  })

  it('shows the responsible-use notice', async () => {
    mocks.listPatients.mockResolvedValue(emptyList)
    renderAt('/dashboard')
    expect(await screen.findByText('Decision-support only')).toBeTruthy()
    expect(screen.getByText(/not a diagnosis or treatment recommendation/, { selector: '.dashboard-responsible-notice p' })).toBeTruthy()
  })
})
