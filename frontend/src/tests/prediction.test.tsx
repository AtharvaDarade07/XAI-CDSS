import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { AxiosError, type AxiosAdapter, type AxiosResponse, type InternalAxiosRequestConfig } from 'axios'
import { MemoryRouter } from 'react-router-dom'
import App from '../App'
import { AuthProvider } from '../context/AuthContext'
import { apiClient } from '../services/api'
import type { Patient, PatientRecord } from '../types/api'

const patientMocks = vi.hoisted(() => ({ getPatient: vi.fn(), getPatientRecord: vi.fn() }))
vi.mock('../services/patientService', () => patientMocks)

const tokenKey = 'xai-cdss.access-token'
const currentUser = { id: 6, username: 'test-clinician', email: 'clinician@example.test', role: 'clinician', is_active: true }
const patient: Patient = { id: 4, patient_code: 'SYNTHETIC-004', created_at: '2026-03-01T00:00:00Z', updated_at: '2026-03-01T00:00:00Z' }
const record: PatientRecord = {
  id: 12, patient_id: 4, pregnancies: 2, glucose: 120, blood_pressure: 70, skin_thickness: 25,
  insulin: 80, bmi: 29.5, diabetes_pedigree_function: 0.5, age: 40, created_at: '2026-03-02T00:00:00Z',
}
const prediction = {
  id: 51, patient_id: 4, patient_record_id: 12, model_version: 'xgboost-baseline-v1',
  predicted_class: 1 as const, predicted_probability: 0.8078943490982056, created_at: '2026-03-03T12:30:00Z',
}
const shapFeatures = [
  { feature: 'Age', feature_value: 50, contribution: 0.754732, direction: 'positive', rank: 1 },
  { feature: 'Glucose', feature_value: 148, contribution: 0.611457, direction: 'positive', rank: 2 },
  { feature: 'Pregnancies', feature_value: 6, contribution: -0.398355, direction: 'negative', rank: 3 },
  { feature: 'SkinThickness', feature_value: 35, contribution: 0.383024, direction: 'positive', rank: 4 },
  { feature: 'Insulin', feature_value: 0, contribution: 0.311468, direction: 'positive', rank: 5 },
  { feature: 'BloodPressure', feature_value: 72, contribution: 0.15426, direction: 'positive', rank: 6 },
  { feature: 'DiabetesPedigreeFunction', feature_value: 0.627, contribution: 0.148399, direction: 'positive', rank: 7 },
  { feature: 'BMI', feature_value: 33.6, contribution: 0.076615, direction: 'positive', rank: 8 },
]
const shapResult = {
  prediction_id: prediction.id, patient_id: prediction.patient_id, patient_record_id: prediction.patient_record_id,
  method: 'SHAP', predicted_class: prediction.predicted_class, predicted_probability: prediction.predicted_probability,
  features: shapFeatures, created_at: '2026-03-03T12:40:00Z',
  summary: 'Model contribution summary.', output_space: 'raw tree-model output (margin; not probability)',
  shap_base_value: 0.12, shap_contribution_sum: 1.64, shap_model_output: 1.76,
  shap_additivity_error: 0.000001, lime_configuration: null,
}
const limeFeatures = shapFeatures.map((feature, index) => ({
  ...feature,
  contribution: index === 0 ? 0.46 : index === 1 ? -0.09 : feature.contribution / 10,
}))
const limeResult = {
  prediction_id: prediction.id, patient_id: prediction.patient_id, patient_record_id: prediction.patient_record_id,
  method: 'LIME', predicted_class: prediction.predicted_class, predicted_probability: prediction.predicted_probability,
  features: limeFeatures, created_at: '2026-03-03T12:45:00Z', summary: 'Local surrogate summary.',
  output_space: 'local surrogate weight for positive-class probability', shap_base_value: null,
  shap_contribution_sum: null, shap_model_output: null, shap_additivity_error: null,
  lime_configuration: { random_state: 42, num_features: 8, num_samples: 5000 },
}
const consistencyResult = {
  prediction_id: prediction.id, patient_id: prediction.patient_id, patient_record_id: prediction.patient_record_id,
  top3_overlap_count: 2, top3_overlap_percentage: 66.6666666667,
  top5_overlap_count: 4, top5_overlap_percentage: 80,
  spearman_correlation: 0.4524, spearman_p_value: 0.2604,
  direction_agreement_count: 8, direction_disagreement_count: 0,
  direction_agreement_percentage: 100, shared_feature_count: 8,
  shap_top_features: ['Age', 'Glucose', 'Pregnancies'],
  lime_top_features: ['Age', 'Glucose', 'SkinThickness'],
  interpretation: 'This compares SHAP and LIME for this explanation instance only.',
  created_at: '2026-03-03T12:50:00Z',
}

let predictionMode: 'success' | 'unavailable' | 'unauthorized' | 'deferred'
let shapMode: 'success' | 'unavailable' | 'unauthorized' | 'deferred'
let limeMode: 'success' | 'unavailable' | 'unauthorized' | 'deferred'
let consistencyMode: 'success' | 'unavailable' | 'unauthorized' | 'deferred'
let predictionRequests: Array<{ url?: string; method?: string; data?: unknown; headers: InternalAxiosRequestConfig['headers'] }> = []
let shapRequests: Array<{ url?: string; method?: string; data?: unknown; headers: InternalAxiosRequestConfig['headers'] }> = []
let limeRequests: Array<{ url?: string; method?: string; data?: unknown; headers: InternalAxiosRequestConfig['headers'] }> = []
let consistencyRequests: Array<{ url?: string; method?: string; data?: unknown; headers: InternalAxiosRequestConfig['headers'] }> = []
let finishDeferred: (() => void) | null
let finishShapDeferred: (() => void) | null
let finishLimeDeferred: (() => void) | null
let finishConsistencyDeferred: (() => void) | null

function response(config: InternalAxiosRequestConfig, status: number, data: unknown): AxiosResponse {
  return { config, status, statusText: status === 201 ? 'Created' : 'OK', headers: {}, data }
}

function testAdapter(): AxiosAdapter {
  return async (config) => {
    if (config.url === '/auth/me') return response(config, 200, currentUser)
    if (config.url?.endsWith('/predict')) {
      predictionRequests.push({ url: config.url, method: config.method, data: config.data, headers: config.headers })
      if (predictionMode === 'deferred') {
        return new Promise<AxiosResponse>((resolve) => {
          finishDeferred = () => resolve(response(config, 201, prediction))
        })
      }
      if (predictionMode === 'unavailable' || predictionMode === 'unauthorized') {
        const status = predictionMode === 'unavailable' ? 503 : 401
        const failed = response(config, status, { detail: 'internal detail must not be shown' })
        throw new AxiosError(`Request failed with status code ${status}`, AxiosError.ERR_BAD_RESPONSE, config, undefined, failed)
      }
      return response(config, 201, prediction)
    }
    if (config.url?.endsWith('/explanation/shap')) {
      shapRequests.push({ url: config.url, method: config.method, data: config.data, headers: config.headers })
      if (shapMode === 'deferred') {
        return new Promise<AxiosResponse>((resolve) => {
          finishShapDeferred = () => resolve(response(config, 201, shapResult))
        })
      }
      if (shapMode === 'unavailable' || shapMode === 'unauthorized') {
        const status = shapMode === 'unavailable' ? 503 : 401
        const failed = response(config, status, { detail: 'internal explanation detail must not be shown' })
        throw new AxiosError(`Request failed with status code ${status}`, AxiosError.ERR_BAD_RESPONSE, config, undefined, failed)
      }
      return response(config, 201, shapResult)
    }
    if (config.url?.endsWith('/explanation/lime')) {
      limeRequests.push({ url: config.url, method: config.method, data: config.data, headers: config.headers })
      if (limeMode === 'deferred') return new Promise<AxiosResponse>((resolve) => { finishLimeDeferred = () => resolve(response(config, 201, limeResult)) })
      if (limeMode === 'unavailable' || limeMode === 'unauthorized') {
        const status = limeMode === 'unavailable' ? 503 : 401
        const failed = response(config, status, { detail: 'internal LIME detail must not be shown' })
        throw new AxiosError(`Request failed with status code ${status}`, AxiosError.ERR_BAD_RESPONSE, config, undefined, failed)
      }
      return response(config, 201, limeResult)
    }
    if (config.url?.endsWith('/explanation/consistency')) {
      consistencyRequests.push({ url: config.url, method: config.method, data: config.data, headers: config.headers })
      if (consistencyMode === 'deferred') return new Promise<AxiosResponse>((resolve) => { finishConsistencyDeferred = () => resolve(response(config, 200, consistencyResult)) })
      if (consistencyMode === 'unavailable' || consistencyMode === 'unauthorized') {
        const status = consistencyMode === 'unavailable' ? 503 : 401
        const failed = response(config, status, { detail: 'internal consistency detail must not be shown' })
        throw new AxiosError(`Request failed with status code ${status}`, AxiosError.ERR_BAD_RESPONSE, config, undefined, failed)
      }
      return response(config, 200, consistencyResult)
    }
    return response(config, 200, {})
  }
}

function renderAt(path = '/patients/4/records/12') {
  window.localStorage.setItem(tokenKey, 'test-token')
  return render(<MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }} initialEntries={[path]}>
    <AuthProvider><App /></AuthProvider>
  </MemoryRouter>)
}

beforeEach(() => {
  window.localStorage.clear()
  predictionMode = 'success'
  shapMode = 'success'
  limeMode = 'success'
  consistencyMode = 'success'
  predictionRequests = []
  shapRequests = []
  limeRequests = []
  consistencyRequests = []
  finishDeferred = null
  finishShapDeferred = null
  finishLimeDeferred = null
  finishConsistencyDeferred = null
  apiClient.defaults.adapter = testAdapter()
  patientMocks.getPatient.mockResolvedValue(patient)
  patientMocks.getPatientRecord.mockResolvedValue(record)
})

afterEach(() => {
  cleanup()
  window.localStorage.clear()
  vi.clearAllMocks()
})

describe('record prediction UI', () => {
  it('loads record detail and offers an explicit Run Prediction action', async () => {
    renderAt()
    expect(await screen.findByRole('heading', { name: 'Record 12' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Run Prediction' })).toBeTruthy()
    expect(screen.getByText(/Previous predictions are not loaded automatically after a page refresh/)).toBeTruthy()
    expect((screen.getByRole('button', { name: 'Generate SHAP Explanation' }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole('button', { name: 'Generate LIME Explanation' }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole('button', { name: 'Analyze Explanation Consistency' }) as HTMLButtonElement).disabled).toBe(true)
    expect(predictionRequests).toHaveLength(0)
  })

  it('posts to the matching record endpoint with the centralized bearer token', async () => {
    const user = userEvent.setup()
    renderAt()
    await user.click(await screen.findByRole('button', { name: 'Run Prediction' }))
    await screen.findByRole('heading', { name: 'Prediction result' })
    expect(predictionRequests).toHaveLength(1)
    expect(predictionRequests[0].url).toBe('/patients/4/records/12/predict')
    expect(predictionRequests[0].method).toBe('post')
    expect(predictionRequests[0].data).toBeUndefined()
    expect(predictionRequests[0].headers.get('Authorization')).toBe('Bearer test-token')
  })

  it('displays the returned class, formatted and raw probability, and model version', async () => {
    const user = userEvent.setup()
    renderAt()
    await user.click(await screen.findByRole('button', { name: 'Run Prediction' }))
    expect(await screen.findByText('80.8%')).toBeTruthy()
    expect(screen.getByText('Raw value: 0.807894')).toBeTruthy()
    expect(screen.getByText('xgboost-baseline-v1')).toBeTruthy()
    expect(screen.getByText('Positive', { selector: 'dd' })).toBeTruthy()
    expect(screen.getByText('Experimental model output — not a medical diagnosis. Final clinical decisions require qualified professional review.')).toBeTruthy()
    expect(screen.getByRole('heading', { name: 'Generate SHAP Explanation' })).toBeTruthy()
    expect((screen.getByRole('button', { name: 'Generate SHAP Explanation' }) as HTMLButtonElement).disabled).toBe(false)
  })

  it('disables submission and announces processing while the request is pending', async () => {
    predictionMode = 'deferred'
    const user = userEvent.setup()
    renderAt()
    const button = await screen.findByRole('button', { name: 'Run Prediction' })
    await user.click(button)
    expect((await screen.findByRole('button', { name: 'Running prediction…' }) as HTMLButtonElement).disabled).toBe(true)
    expect(screen.getByRole('status').textContent).toContain('processing this stored clinical record')
    await waitFor(() => expect(finishDeferred).not.toBeNull())
    await act(async () => { finishDeferred?.() })
    expect(await screen.findByRole('heading', { name: 'Prediction result' })).toBeTruthy()
  })

  it('shows a concise message when the model service is unavailable', async () => {
    predictionMode = 'unavailable'
    const user = userEvent.setup()
    renderAt()
    await user.click(await screen.findByRole('button', { name: 'Run Prediction' }))
    expect(await screen.findByText('The prediction model is currently unavailable. Please try again later.')).toBeTruthy()
    expect(screen.queryByText('internal detail must not be shown')).toBeNull()
  })

  it('uses the existing 401 session handling and redirects to login', async () => {
    predictionMode = 'unauthorized'
    const user = userEvent.setup()
    renderAt()
    await user.click(await screen.findByRole('button', { name: 'Run Prediction' }))
    expect(await screen.findByRole('heading', { name: 'Sign in to your workspace' })).toBeTruthy()
    await waitFor(() => expect(window.localStorage.getItem(tokenKey)).toBeNull())
  })

  it('disables SHAP until a prediction exists and then enables it', async () => {
    const user = userEvent.setup()
    renderAt()
    const shapButton = await screen.findByRole('button', { name: 'Generate SHAP Explanation' }) as HTMLButtonElement
    expect(shapButton.disabled).toBe(true)
    expect(screen.getByText('Run a prediction before generating an explanation.')).toBeTruthy()
    expect(shapRequests).toHaveLength(0)
    await user.click(screen.getByRole('button', { name: 'Run Prediction' }))
    await screen.findByRole('heading', { name: 'Prediction result' })
    expect(shapButton.disabled).toBe(false)
  })

  it('requests SHAP for the returned prediction ID and renders every ranked feature', async () => {
    const user = userEvent.setup()
    renderAt()
    await user.click(await screen.findByRole('button', { name: 'Run Prediction' }))
    await screen.findByRole('heading', { name: 'Prediction result' })
    await user.click(screen.getByRole('button', { name: 'Generate SHAP Explanation' }))
    expect(await screen.findByRole('heading', { name: 'SHAP Explanation details' })).toBeTruthy()
    expect(shapRequests).toHaveLength(1)
    expect(shapRequests[0].url).toBe('/patients/4/records/12/predictions/51/explanation/shap')
    expect(shapRequests[0].method).toBe('post')
    expect(shapRequests[0].data).toBeUndefined()
    expect(shapRequests[0].headers.get('Authorization')).toBe('Bearer test-token')
    const shapTable = screen.getByRole('table', { name: /Ranked SHAP feature contributions/ })
    for (const feature of shapFeatures) expect(within(shapTable).getByText(feature.feature)).toBeTruthy()
    expect(within(shapTable).getByText('8', { selector: 'tbody tr:last-child td:first-child' })).toBeTruthy()
    expect(within(shapTable).getByText('50', { selector: 'tbody tr:first-child td:nth-child(3)' })).toBeTruthy()
    expect(within(shapTable).getByText('+0.754732')).toBeTruthy()
    expect(within(shapTable).getAllByText('Pushes model output upward').length).toBe(7)
    expect(within(shapTable).getByText('Pushes model output downward')).toBeTruthy()
    expect(screen.getByText('+0.754732').textContent).not.toContain('%')
    expect(screen.getByText('SHAP baseline model output:').parentElement?.textContent).toContain('raw tree-model output (margin; not probability)')
  })

  it('announces SHAP loading and disables the explanation action until complete', async () => {
    shapMode = 'deferred'
    const user = userEvent.setup()
    renderAt()
    await user.click(await screen.findByRole('button', { name: 'Run Prediction' }))
    await screen.findByRole('heading', { name: 'Prediction result' })
    await user.click(screen.getByRole('button', { name: 'Generate SHAP Explanation' }))
    const loadingButton = await screen.findByRole('button', { name: 'Generating SHAP explanation…' }) as HTMLButtonElement
    expect(loadingButton.disabled).toBe(true)
    expect(screen.getByRole('status').textContent).toContain('Generating the SHAP feature contribution explanation')
    await waitFor(() => expect(finishShapDeferred).not.toBeNull())
    await act(async () => { finishShapDeferred?.() })
    expect(await screen.findByRole('heading', { name: 'SHAP Explanation details' })).toBeTruthy()
  })

  it('shows a friendly error when the SHAP service is unavailable', async () => {
    shapMode = 'unavailable'
    const user = userEvent.setup()
    renderAt()
    await user.click(await screen.findByRole('button', { name: 'Run Prediction' }))
    await screen.findByRole('heading', { name: 'Prediction result' })
    await user.click(screen.getByRole('button', { name: 'Generate SHAP Explanation' }))
    expect(await screen.findByText('Unable to generate the SHAP explanation because the model is unavailable or incompatible.')).toBeTruthy()
    expect(screen.queryByText('internal explanation detail must not be shown')).toBeNull()
  })

  it('routes SHAP 401 through the existing authentication session handling', async () => {
    shapMode = 'unauthorized'
    const user = userEvent.setup()
    renderAt()
    await user.click(await screen.findByRole('button', { name: 'Run Prediction' }))
    await screen.findByRole('heading', { name: 'Prediction result' })
    await user.click(screen.getByRole('button', { name: 'Generate SHAP Explanation' }))
    expect(await screen.findByRole('heading', { name: 'Sign in to your workspace' })).toBeTruthy()
    await waitFor(() => expect(window.localStorage.getItem(tokenKey)).toBeNull())
  })

  it('disables LIME until a prediction exists and enables it after prediction', async () => {
    const user = userEvent.setup()
    renderAt()
    const limeButton = await screen.findByRole('button', { name: 'Generate LIME Explanation' }) as HTMLButtonElement
    expect(limeButton.disabled).toBe(true)
    expect(screen.getByText('Run a prediction before generating an explanation.')).toBeTruthy()
    expect(limeRequests).toHaveLength(0)
    await user.click(screen.getByRole('button', { name: 'Run Prediction' }))
    await screen.findByRole('heading', { name: 'Prediction result' })
    expect(limeButton.disabled).toBe(false)
  })

  it('requests LIME for the current prediction and renders all ranked values and config', async () => {
    const user = userEvent.setup()
    renderAt()
    await user.click(await screen.findByRole('button', { name: 'Run Prediction' }))
    await screen.findByRole('heading', { name: 'Prediction result' })
    await user.click(screen.getByRole('button', { name: 'Generate LIME Explanation' }))
    expect(await screen.findByRole('heading', { name: 'LIME Explanation' })).toBeTruthy()
    expect(limeRequests).toHaveLength(1)
    expect(limeRequests[0].url).toBe('/patients/4/records/12/predictions/51/explanation/lime')
    expect(limeRequests[0].method).toBe('post')
    expect(limeRequests[0].data).toBeUndefined()
    expect(limeRequests[0].headers.get('Authorization')).toBe('Bearer test-token')
    expect(document.querySelector('.explanation-context')?.textContent).toContain('Both methods below explain prediction #51')
    const limeTable = screen.getByRole('table', { name: /Ranked LIME feature contributions/ })
    for (const feature of limeFeatures) expect(within(limeTable).getByText(feature.feature)).toBeTruthy()
    expect(within(limeTable).getByText('8', { selector: 'tbody tr:last-child td:first-child' })).toBeTruthy()
    expect(within(limeTable).getByText('50', { selector: 'tbody tr:first-child td:nth-child(3)' })).toBeTruthy()
    expect(within(limeTable).getByText('+0.46')).toBeTruthy()
    expect(within(limeTable).getByText('-0.09')).toBeTruthy()
    expect(within(limeTable).getAllByText("Supports the model's positive prediction").length).toBeGreaterThan(0)
    expect(within(limeTable).getByText("Pushes away from the model's positive prediction")).toBeTruthy()
    expect(screen.getByText('Random seed').parentElement?.textContent).toContain('42')
    expect(screen.getByText('Number of features').parentElement?.textContent).toContain('8')
    expect(screen.getByText('Number of samples').parentElement?.textContent).toContain('5000')
    expect(screen.getByText('LIME contribution values are local explanation units, not percentage-point changes in probability.')).toBeTruthy()
    expect(screen.getByText('+0.46').textContent).not.toContain('%')
  })

  it('keeps LIME and SHAP requests independent while one explanation is pending', async () => {
    limeMode = 'deferred'
    const user = userEvent.setup()
    renderAt()
    await user.click(await screen.findByRole('button', { name: 'Run Prediction' }))
    await screen.findByRole('heading', { name: 'Prediction result' })
    await user.click(screen.getByRole('button', { name: 'Generate LIME Explanation' }))
    const limeLoadingButton = await screen.findByRole('button', { name: 'Generating LIME explanation…' }) as HTMLButtonElement
    expect(limeLoadingButton.disabled).toBe(true)
    expect(screen.getByRole('status').textContent).toContain('Generating the LIME feature contribution explanation')
    expect((screen.getByRole('button', { name: 'Generate SHAP Explanation' }) as HTMLButtonElement).disabled).toBe(false)
    expect((screen.getByRole('button', { name: 'Run Prediction' }) as HTMLButtonElement).disabled).toBe(true)
    await waitFor(() => expect(finishLimeDeferred).not.toBeNull())
    await act(async () => { finishLimeDeferred?.() })
    expect(await screen.findByRole('heading', { name: 'LIME Explanation' })).toBeTruthy()
  })

  it('shows a concise LIME service error', async () => {
    limeMode = 'unavailable'
    const user = userEvent.setup()
    renderAt()
    await user.click(await screen.findByRole('button', { name: 'Run Prediction' }))
    await screen.findByRole('heading', { name: 'Prediction result' })
    await user.click(screen.getByRole('button', { name: 'Generate LIME Explanation' }))
    expect(await screen.findByText('Unable to generate the LIME explanation because the model is unavailable or incompatible.')).toBeTruthy()
    expect(screen.queryByText('internal LIME detail must not be shown')).toBeNull()
  })

  it('routes LIME 401 through existing authentication handling', async () => {
    limeMode = 'unauthorized'
    const user = userEvent.setup()
    renderAt()
    await user.click(await screen.findByRole('button', { name: 'Run Prediction' }))
    await screen.findByRole('heading', { name: 'Prediction result' })
    await user.click(screen.getByRole('button', { name: 'Generate LIME Explanation' }))
    expect(await screen.findByRole('heading', { name: 'Sign in to your workspace' })).toBeTruthy()
    await waitFor(() => expect(window.localStorage.getItem(tokenKey)).toBeNull())
  })

  it('shows consistency action disabled until a prediction exists', async () => {
    const user = userEvent.setup()
    renderAt()
    const button = await screen.findByRole('button', { name: 'Analyze Explanation Consistency' }) as HTMLButtonElement
    expect(button.disabled).toBe(true)
    expect(consistencyRequests).toHaveLength(0)
    await user.click(screen.getByRole('button', { name: 'Run Prediction' }))
    await screen.findByRole('heading', { name: 'Prediction result' })
    expect(button.disabled).toBe(false)
  })

  it('requests backend consistency metrics for the displayed prediction and renders the returned fields', async () => {
    const user = userEvent.setup()
    renderAt()
    await user.click(await screen.findByRole('button', { name: 'Run Prediction' }))
    await screen.findByRole('heading', { name: 'Prediction result' })
    await user.click(screen.getByRole('button', { name: 'Analyze Explanation Consistency' }))
    expect(await screen.findByText('2 / 3')).toBeTruthy()
    expect(consistencyRequests).toHaveLength(1)
    expect(consistencyRequests[0].url).toBe('/patients/4/records/12/predictions/51/explanation/consistency')
    expect(consistencyRequests[0].method).toBe('get')
    expect(consistencyRequests[0].data).toBeUndefined()
    expect(consistencyRequests[0].headers.get('Authorization')).toBe('Bearer test-token')
    expect(document.querySelector('.consistency-context')?.textContent).toContain('prediction #51')
    expect(screen.getByText('Top-3 feature overlap').parentElement?.textContent).toContain('66.7%')
    expect(screen.getByText('Top-5 feature overlap').parentElement?.textContent).toContain('80.0%')
    expect(screen.getByText('Spearman correlation').parentElement?.textContent).toContain('0.4524')
    expect(screen.getByText(/Spearman p-value: 0.2604/)).toBeTruthy()
    expect(screen.getByText('Direction agreement').parentElement?.textContent).toContain('8 / 8')
    expect(screen.getByText('Direction agreement').parentElement?.textContent).toContain('100.0%')
    expect(screen.getByText('This compares SHAP and LIME for this explanation instance only.')).toBeTruthy()
    expect(screen.getByText('Explanation agreement does not establish clinical validity, diagnostic accuracy, or general model reliability.')).toBeTruthy()
    expect(screen.getAllByText('Age', { selector: '.consistency-feature-lists li' })).toHaveLength(2)
  })

  it('announces consistency loading and prevents duplicate analysis requests', async () => {
    consistencyMode = 'deferred'
    const user = userEvent.setup()
    renderAt()
    await user.click(await screen.findByRole('button', { name: 'Run Prediction' }))
    await screen.findByRole('heading', { name: 'Prediction result' })
    await user.click(screen.getByRole('button', { name: 'Analyze Explanation Consistency' }))
    const loadingButton = await screen.findByRole('button', { name: 'Analyzing explanation consistency…' }) as HTMLButtonElement
    expect(loadingButton.disabled).toBe(true)
    expect(screen.getByRole('status').textContent).toContain('Comparing SHAP and LIME')
    expect(consistencyRequests).toHaveLength(1)
    await waitFor(() => expect(finishConsistencyDeferred).not.toBeNull())
    await act(async () => { finishConsistencyDeferred?.() })
    expect(await screen.findByText('0.4524')).toBeTruthy()
  })

  it('shows a concise consistency API error without exposing backend details', async () => {
    consistencyMode = 'unavailable'
    const user = userEvent.setup()
    renderAt()
    await user.click(await screen.findByRole('button', { name: 'Run Prediction' }))
    await screen.findByRole('heading', { name: 'Prediction result' })
    await user.click(screen.getByRole('button', { name: 'Analyze Explanation Consistency' }))
    expect(await screen.findByText('The consistency analysis is temporarily unavailable. Please try again later.')).toBeTruthy()
    expect(screen.queryByText('internal consistency detail must not be shown')).toBeNull()
  })

  it('clears stale consistency metrics when a new prediction is requested', async () => {
    const user = userEvent.setup()
    renderAt()
    await user.click(await screen.findByRole('button', { name: 'Run Prediction' }))
    await screen.findByRole('heading', { name: 'Prediction result' })
    await user.click(screen.getByRole('button', { name: 'Analyze Explanation Consistency' }))
    await screen.findByText('0.4524')
    await user.click(screen.getByRole('button', { name: 'Run Prediction' }))
    await waitFor(() => expect(screen.queryByText('0.4524')).toBeNull())
  })

  it('uses the current prediction ID for SHAP, LIME, and consistency in one workflow', async () => {
    const user = userEvent.setup()
    renderAt()
    await user.click(await screen.findByRole('button', { name: 'Run Prediction' }))
    await screen.findByRole('heading', { name: 'Prediction result' })
    await user.click(screen.getByRole('button', { name: 'Generate SHAP Explanation' }))
    await screen.findByRole('heading', { name: 'SHAP Explanation details' })
    await user.click(screen.getByRole('button', { name: 'Generate LIME Explanation' }))
    await screen.findByRole('heading', { name: 'LIME Explanation' })
    await user.click(screen.getByRole('button', { name: 'Analyze Explanation Consistency' }))
    await screen.findByText('0.4524')

    expect(predictionRequests[0].url).toBe('/patients/4/records/12/predict')
    expect(shapRequests[0].url).toBe('/patients/4/records/12/predictions/51/explanation/shap')
    expect(limeRequests[0].url).toBe('/patients/4/records/12/predictions/51/explanation/lime')
    expect(consistencyRequests[0].url).toBe('/patients/4/records/12/predictions/51/explanation/consistency')
    expect(document.querySelector('.explanation-context')?.textContent).toContain('prediction #51')
    expect(document.querySelector('.consistency-context')?.textContent).toContain('prediction #51')
  })

  it('hides old prediction and explanation results as soon as a replacement prediction starts', async () => {
    const user = userEvent.setup()
    renderAt()
    await user.click(await screen.findByRole('button', { name: 'Run Prediction' }))
    await screen.findByRole('heading', { name: 'Prediction result' })
    await user.click(screen.getByRole('button', { name: 'Generate SHAP Explanation' }))
    await screen.findByRole('heading', { name: 'SHAP Explanation details' })
    await user.click(screen.getByRole('button', { name: 'Generate LIME Explanation' }))
    await screen.findByRole('heading', { name: 'LIME Explanation' })
    await user.click(screen.getByRole('button', { name: 'Analyze Explanation Consistency' }))
    await screen.findByText('0.4524')

    predictionMode = 'deferred'
    await user.click(screen.getByRole('button', { name: 'Run Prediction' }))
    await screen.findByRole('status')
    expect(screen.queryByRole('heading', { name: 'Prediction result' })).toBeNull()
    expect(screen.queryByRole('heading', { name: 'SHAP Explanation details' })).toBeNull()
    expect(screen.queryByRole('heading', { name: 'LIME Explanation' })).toBeNull()
    expect(screen.queryByText('0.4524')).toBeNull()
    await waitFor(() => expect(finishDeferred).not.toBeNull())
    await act(async () => { finishDeferred?.() })
    expect(await screen.findByRole('heading', { name: 'Prediction result' })).toBeTruthy()
  })
})
