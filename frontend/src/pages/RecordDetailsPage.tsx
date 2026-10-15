import { Fragment, useEffect, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import axios from 'axios'
import PageHeading from '../components/PageHeading'
import { ErrorState, LoadingState } from '../components/StatePanels'
import { getPatient, getPatientRecord } from '../services/patientService'
import { predictPatientRecord } from '../services/predictionService'
import { generateLimeExplanation, generateShapExplanation, getExplanationConsistency } from '../services/explanationService'
import type { ExplanationConsistencyResponse, ExplanationResponse, Patient, PatientRecord, Prediction } from '../types/api'
import { formatDate } from './PatientsPage'

const fields = [
  ['pregnancies', 'Pregnancies'], ['glucose', 'Glucose'], ['blood_pressure', 'Blood pressure'],
  ['skin_thickness', 'Skin thickness'], ['insulin', 'Insulin'], ['bmi', 'BMI'],
  ['diabetes_pedigree_function', 'Diabetes pedigree function'], ['age', 'Age'],
] as const satisfies readonly (readonly [keyof PatientRecord, string])[]

export default function RecordDetailsPage() {
  const { patientId: patientParam, recordId: recordParam } = useParams()
  const patientId = Number(patientParam)
  const recordId = Number(recordParam)
  const validIds = Number.isInteger(patientId) && patientId > 0 && Number.isInteger(recordId) && recordId > 0
  const currentRecord = useRef({ patientId, recordId })
  currentRecord.current = { patientId, recordId }
  const [patient, setPatient] = useState<Patient | null>(null)
  const [record, setRecord] = useState<PatientRecord | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [predictionLoading, setPredictionLoading] = useState(false)
  const [predictionError, setPredictionError] = useState('')
  const [shapExplanation, setShapExplanation] = useState<ExplanationResponse | null>(null)
  const [shapLoading, setShapLoading] = useState(false)
  const [shapError, setShapError] = useState('')
  const [limeExplanation, setLimeExplanation] = useState<ExplanationResponse | null>(null)
  const [limeLoading, setLimeLoading] = useState(false)
  const [limeError, setLimeError] = useState('')
  const [consistencyResult, setConsistencyResult] = useState<ExplanationConsistencyResponse | null>(null)
  const [consistencyLoading, setConsistencyLoading] = useState(false)
  const [consistencyError, setConsistencyError] = useState('')

  useEffect(() => {
    let active = true
    async function load() {
      if (!validIds) { setError('The patient or record identifier is invalid.'); setLoading(false); return }
      setLoading(true)
      setError('')
      setPredictionLoading(false)
      setShapLoading(false)
      setLimeLoading(false)
      setConsistencyLoading(false)
      setPrediction(null)
      setPredictionError('')
      setShapExplanation(null)
      setShapError('')
      setLimeExplanation(null)
      setLimeError('')
      setConsistencyResult(null)
      setConsistencyError('')
      try {
        const [patientResult, recordResult] = await Promise.all([getPatient(patientId), getPatientRecord(patientId, recordId)])
        if (active) { setPatient(patientResult); setRecord(recordResult) }
      } catch (cause) {
        if (active) setError(axios.isAxiosError(cause) && cause.response?.status === 404 ? 'The requested patient record could not be found.' : 'Patient record details could not be loaded. Please try again.')
      } finally { if (active) setLoading(false) }
    }
    void load()
    return () => { active = false }
  }, [patientId, recordId, validIds])

  async function runPrediction() {
    if (predictionLoading || shapLoading || limeLoading || consistencyLoading) return
    // A new request creates a distinct prediction. Hide the previous prediction
    // and its downstream explanations as soon as the new request begins.
    setPrediction(null)
    setShapExplanation(null)
    setShapError('')
    setLimeExplanation(null)
    setLimeError('')
    setConsistencyResult(null)
    setConsistencyError('')
    setPredictionLoading(true)
    setPredictionError('')
    try {
      // Each successful request creates a new prediction; keep only the latest response in view.
      const latestPrediction = await predictPatientRecord(patientId, recordId)
      if (currentRecord.current.patientId !== patientId || currentRecord.current.recordId !== recordId) return
      setPrediction(latestPrediction)
    } catch (cause) {
      if (currentRecord.current.patientId !== patientId || currentRecord.current.recordId !== recordId) return
      setPredictionError(predictionErrorMessage(cause))
    } finally {
      if (currentRecord.current.patientId === patientId && currentRecord.current.recordId === recordId) setPredictionLoading(false)
    }
  }

  async function runShapExplanation() {
    if (!prediction || predictionLoading || shapLoading) return
    const requestedPredictionId = prediction.id
    setShapExplanation(null)
    setConsistencyResult(null)
    setConsistencyError('')
    setShapLoading(true)
    setShapError('')
    try {
      const result = await generateShapExplanation(patientId, recordId, requestedPredictionId)
      if (currentRecord.current.patientId !== patientId || currentRecord.current.recordId !== recordId) return
      if (result.prediction_id !== requestedPredictionId) {
        setShapError('The returned explanation does not match the current prediction. Please try again.')
        return
      }
      setShapExplanation(result)
    } catch (cause) {
      if (currentRecord.current.patientId !== patientId || currentRecord.current.recordId !== recordId) return
      setShapError(explanationErrorMessage(cause, 'SHAP'))
    } finally {
      if (currentRecord.current.patientId === patientId && currentRecord.current.recordId === recordId) setShapLoading(false)
    }
  }

  async function runLimeExplanation() {
    if (!prediction || predictionLoading || limeLoading) return
    const requestedPredictionId = prediction.id
    setLimeExplanation(null)
    setConsistencyResult(null)
    setConsistencyError('')
    setLimeLoading(true)
    setLimeError('')
    try {
      const result = await generateLimeExplanation(patientId, recordId, requestedPredictionId)
      if (currentRecord.current.patientId !== patientId || currentRecord.current.recordId !== recordId) return
      if (result.prediction_id !== requestedPredictionId) {
        setLimeError('The returned explanation does not match the current prediction. Please try again.')
        return
      }
      setLimeExplanation(result)
    } catch (cause) {
      if (currentRecord.current.patientId !== patientId || currentRecord.current.recordId !== recordId) return
      setLimeError(explanationErrorMessage(cause, 'LIME'))
    } finally {
      if (currentRecord.current.patientId === patientId && currentRecord.current.recordId === recordId) setLimeLoading(false)
    }
  }

  async function analyzeConsistency() {
    if (!prediction || predictionLoading || consistencyLoading) return
    const requestedPredictionId = prediction.id
    setConsistencyResult(null)
    setConsistencyLoading(true)
    setConsistencyError('')
    try {
      const result = await getExplanationConsistency(patientId, recordId, requestedPredictionId)
      if (currentRecord.current.patientId !== patientId || currentRecord.current.recordId !== recordId) return
      if (result.prediction_id !== requestedPredictionId || result.patient_id !== patientId || result.patient_record_id !== recordId) {
        setConsistencyError('The returned comparison does not match the current prediction and record. Please try again.')
        return
      }
      setConsistencyResult(result)
    } catch (cause) {
      if (currentRecord.current.patientId !== patientId || currentRecord.current.recordId !== recordId) return
      setConsistencyError(consistencyErrorMessage(cause))
    } finally {
      if (currentRecord.current.patientId === patientId && currentRecord.current.recordId === recordId) setConsistencyLoading(false)
    }
  }

  return <>
    <div className="breadcrumbs"><Link to="/patients">Patients</Link><span aria-hidden="true">/</span>{patient && <><Link to={`/patients/${patient.id}`}>{patient.patient_code}</Link><span aria-hidden="true">/</span></>}<span>Record details</span></div>
    {loading ? <LoadingState title="Loading clinical record…" /> : error || !patient || !record ? <ErrorState title="Record unavailable">{error || 'The requested patient record could not be found.'}</ErrorState> : <>
      <PageHeading eyebrow="Stored clinical data" title={`Record ${record.id}`} description={`${patient.patient_code} · Patient ID ${patient.id} · Created ${formatDate(record.created_at)}`} />
      <section className="content-card" aria-labelledby="record-measurements-title">
        <h2 id="record-measurements-title">Record measurements</h2>
        <dl className="measurement-grid">{fields.map(([key, label]) => <div className="measurement-item" key={key}><dt>{label}</dt><dd>{record[key] ?? 'Not provided'}</dd></div>)}</dl>
      </section>
      <section className="content-card prediction-action-card" aria-labelledby="prediction-action-title">
        <p className="eyebrow">Stored record inference</p>
        <h2 id="prediction-action-title">Prediction</h2>
        <p className="muted-copy">Run the experimental model using this stored record. Each request creates a new prediction.</p>
        <button className="primary-button" type="button" onClick={() => void runPrediction()} disabled={predictionLoading || shapLoading || limeLoading || consistencyLoading}>
          {predictionLoading ? 'Running prediction…' : 'Run Prediction'}
        </button>
        {predictionLoading && <p className="prediction-loading" role="status">The model is processing this stored clinical record…</p>}
        {predictionError && <p className="inline-error prediction-error" role="alert">{predictionError}</p>}
        {!prediction && !predictionLoading && <p className="prediction-state-note">No prediction is loaded for this page view. Run a prediction to enable SHAP, LIME, and consistency analysis. Previous predictions are not loaded automatically after a page refresh.</p>}
      </section>

      {prediction && <section className="content-card prediction-result-card" aria-labelledby="prediction-result-title" aria-live="polite">
        <p className="eyebrow">Experimental model output</p>
        <h2 id="prediction-result-title">Prediction result</h2>
        <dl className="prediction-grid">
          <div><dt>Predicted class</dt><dd>{prediction.predicted_class}</dd></div>
          <div><dt>Model classification</dt><dd>{prediction.predicted_class === 1 ? 'Positive' : 'Negative'}</dd></div>
          <div><dt>Positive-class model probability</dt><dd>{formatProbability(prediction.predicted_probability)}</dd><small>Raw value: {prediction.predicted_probability.toFixed(6)}</small></div>
          <div><dt>Model version</dt><dd>{prediction.model_version}</dd></div>
          <div><dt>Generated</dt><dd>{formatDate(prediction.created_at)}</dd></div>
          <div><dt>Prediction ID</dt><dd>{prediction.id}</dd></div>
        </dl>
        <p className="prediction-disclaimer">Experimental model output — not a medical diagnosis. Final clinical decisions require qualified professional review.</p>
      </section>}

      <section className="content-card explanation-methods-card" aria-labelledby="explanation-methods-title">
        <p className="eyebrow">Post-hoc model explanations</p>
        <h2 id="explanation-methods-title">Explanation methods</h2>
        {prediction ? <p className="muted-copy explanation-context">Both methods below explain prediction <strong>#{prediction.id}</strong> · {prediction.model_version} · class {prediction.predicted_class} · positive-class model probability {formatProbability(prediction.predicted_probability)}.</p> : <p className="shap-prerequisite">Run a prediction before generating an explanation.</p>}
        <p className="muted-copy">SHAP and LIME explanations describe how the trained model used the available features for this prediction. They do not establish clinical causation and should not be used as standalone diagnosis or treatment recommendations.</p>
        <div className="explanation-methods-grid">
          <div className="method-panel method-panel--shap">
            <h3>Generate SHAP Explanation</h3>
            <p>Feature contributions in the model output space.</p>
            <button className="primary-button" type="button" onClick={() => void runShapExplanation()} disabled={!prediction || predictionLoading || shapLoading}>
              {shapLoading ? 'Generating SHAP explanation…' : shapExplanation ? 'Generate SHAP Explanation Again' : 'Generate SHAP Explanation'}
            </button>
            {shapLoading && <p className="prediction-loading" role="status">Generating the SHAP feature contribution explanation…</p>}
            {shapError && <p className="inline-error prediction-error" role="alert">{shapError}</p>}
          </div>
          <div className="method-panel method-panel--lime">
            <h3>Generate LIME Explanation</h3>
            <p>Local surrogate explanation for this prediction.</p>
            <button className="primary-button" type="button" onClick={() => void runLimeExplanation()} disabled={!prediction || predictionLoading || limeLoading}>
              {limeLoading ? 'Generating LIME explanation…' : limeExplanation ? 'Generate LIME Explanation Again' : 'Generate LIME Explanation'}
            </button>
            {limeLoading && <p className="prediction-loading" role="status">Generating the LIME feature contribution explanation…</p>}
            {limeError && <p className="inline-error prediction-error" role="alert">{limeError}</p>}
          </div>
        </div>
      </section>

      {shapExplanation && <ExplanationResult explanation={shapExplanation} method="SHAP" />}
      {limeExplanation && <ExplanationResult explanation={limeExplanation} method="LIME" />}
      <section className="content-card consistency-action-card" aria-labelledby="consistency-title">
        <p className="eyebrow">Instance-level comparison</p>
        <h2 id="consistency-title">Explanation Consistency</h2>
        {prediction ? <p className="muted-copy">Analyze how SHAP and LIME compare for prediction <strong>#{prediction.id}</strong>. Metrics are calculated by the backend for this prediction.</p> : <p className="muted-copy">Run a prediction before analyzing explanation consistency.</p>}
        <button className="primary-button" type="button" onClick={() => void analyzeConsistency()} disabled={!prediction || consistencyLoading || predictionLoading}>
          {consistencyLoading ? 'Analyzing explanation consistency…' : consistencyResult ? 'Analyze Explanation Consistency Again' : 'Analyze Explanation Consistency'}
        </button>
        {consistencyLoading && <p className="prediction-loading" role="status">Comparing SHAP and LIME explanations for prediction {prediction?.id}…</p>}
        {consistencyError && <p className="inline-error prediction-error" role="alert">{consistencyError}</p>}
        {consistencyResult && <ConsistencyResult result={consistencyResult} />}
      </section>
      <nav className="record-return-links" aria-label="Record navigation"><Link className="text-action" to={`/patients/${patient.id}`}>Back to Patient</Link><Link className="text-action" to="/patients">Back to Patients</Link></nav>
    </>}
  </>
}

function ExplanationResult({ explanation, method }: { explanation: ExplanationResponse; method: 'SHAP' | 'LIME' }) {
  const maxContribution = Math.max(...explanation.features.map((item) => Math.abs(item.contribution)), Number.EPSILON)
  const configurationLabels: Record<string, string> = { random_state: 'Random seed', num_features: 'Number of features', num_samples: 'Number of samples' }
  return <section className={`content-card shap-result-card ${method === 'LIME' ? 'lime-result-card' : ''}`} aria-labelledby={`${method.toLowerCase()}-result-title`} aria-live="polite">
    <p className="eyebrow">Model behavior for prediction {explanation.prediction_id}</p>
    <h2 id={`${method.toLowerCase()}-result-title`}>{method} Explanation{method === 'SHAP' ? ' details' : ''}</h2>
    <p className="muted-copy">Top model contributing factors, in the rank order returned by the backend:</p>
    <ol className="top-shap-factors">{explanation.features.slice(0, 3).map((feature) => <li key={feature.rank}><strong>{feature.feature}</strong><span>Rank {feature.rank}</span></li>)}</ol>
    <div className="table-card shap-table-card"><div className="table-scroll"><table className="data-table shap-table">
      <caption className="sr-only">Ranked {method} feature contributions for prediction {explanation.prediction_id}</caption>
      <thead><tr><th scope="col">Rank</th><th scope="col">Feature</th><th scope="col">Patient value</th><th scope="col">Contribution</th><th scope="col">Visual contribution</th><th scope="col">Direction</th></tr></thead>
      <tbody>{explanation.features.map((feature) => {
        const barWidth = Math.abs(feature.contribution) / maxContribution * 50
        const positive = feature.contribution >= 0
        return <tr key={`${feature.rank}-${feature.feature}`}>
          <td>{feature.rank}</td><td className="shap-feature-name">{feature.feature}</td>
          <td>{feature.feature_value === null ? 'Not provided' : String(feature.feature_value)}</td>
          <td className="shap-number">{formatContribution(feature.contribution)}</td>
          <td><div className="contribution-track" aria-hidden="true"><span className={`contribution-bar ${positive ? 'contribution-bar--positive' : 'contribution-bar--negative'}`} style={{ left: positive ? '50%' : `${50 - barWidth}%`, width: `${barWidth}%` }} /></div><span className="sr-only">{positive ? 'Positive' : 'Negative'} {method} contribution</span></td>
          <td>{directionLabel(feature.direction, method)}</td>
        </tr>
      })}</tbody>
    </table></div></div>
    {method === 'SHAP' && explanation.shap_base_value !== null && <p className="shap-base-value"><strong>SHAP baseline model output:</strong> {formatContribution(explanation.shap_base_value)} <span>({explanation.output_space})</span></p>}
    {method === 'LIME' && <>
      <p className="shap-base-value"><strong>LIME output space:</strong> <span>{explanation.output_space}</span></p>
      <p className="shap-base-value lime-units-note">LIME contribution values are local explanation units, not percentage-point changes in probability.</p>
      {explanation.lime_configuration && <dl className="lime-configuration"><dt className="lime-config-title">LIME configuration</dt>{Object.entries(explanation.lime_configuration).map(([key, value]) => <Fragment key={key}><dt>{configurationLabels[key] ?? key}</dt><dd>{value}</dd></Fragment>)}</dl>}
    </>}
  </section>
}

function ConsistencyResult({ result }: { result: ExplanationConsistencyResponse }) {
  const metrics = [
    { label: 'Top-3 feature overlap', count: `${result.top3_overlap_count} / 3`, percentage: result.top3_overlap_percentage },
    { label: 'Top-5 feature overlap', count: `${result.top5_overlap_count} / 5`, percentage: result.top5_overlap_percentage },
    { label: 'Direction agreement', count: `${result.direction_agreement_count} / ${result.shared_feature_count}`, percentage: result.direction_agreement_percentage },
  ]
  return <div className="consistency-result" aria-live="polite">
    <p className="consistency-context">Consistency comparison for prediction <strong>#{result.prediction_id}</strong></p>
    <div className="consistency-metric-grid">
      {metrics.map((metric) => <article className="consistency-metric" key={metric.label}>
        <h3>{metric.label}</h3>
        <p className="consistency-count">{metric.count}</p>
        <p className="consistency-percentage">{formatPercentage(metric.percentage)}</p>
        <progress aria-label={`${metric.label} percentage`} max={100} value={metric.percentage}>{formatPercentage(metric.percentage)}</progress>
      </article>)}
      <article className="consistency-metric consistency-metric--correlation">
        <h3>Spearman correlation</h3>
        <p className="consistency-count">{result.spearman_correlation.toFixed(4)}</p>
        <span className="consistency-secondary">Spearman p-value: {result.spearman_p_value.toFixed(4)}</span>
      </article>
    </div>
    <div className="consistency-feature-lists">
      <div><h3>Top SHAP features</h3><ol>{result.shap_top_features.map((feature) => <li key={feature}>{feature}</li>)}</ol></div>
      <div><h3>Top LIME features</h3><ol>{result.lime_top_features.map((feature) => <li key={feature}>{feature}</li>)}</ol></div>
    </div>
    <p className="consistency-interpretation">{result.interpretation}</p>
    <p className="consistency-limitation">Explanation agreement does not establish clinical validity, diagnostic accuracy, or general model reliability.</p>
  </div>
}

function formatProbability(probability: number): string {
  return `${(probability * 100).toFixed(1)}%`
}

function formatContribution(contribution: number): string {
  return `${contribution >= 0 ? '+' : ''}${String(contribution)}`
}

function directionLabel(direction: string, method: 'SHAP' | 'LIME'): string {
  if (method === 'LIME') {
    if (direction === 'positive') return "Supports the model's positive prediction"
    if (direction === 'negative') return "Pushes away from the model's positive prediction"
    return 'Direction returned by the model explanation'
  }
  if (direction === 'positive') return 'Pushes model output upward'
  if (direction === 'negative') return 'Pushes model output downward'
  return 'Direction returned by the model explanation'
}

function predictionErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const status = error.response?.status
    if (status === 401) return 'Your session has expired. Sign in again to continue.'
    if (status === 403) return 'Your account does not have permission to run predictions.'
    if (status === 404) return 'The patient or clinical record could not be found.'
    if (status === 409) return 'The model version is currently in conflict. Please try again later.'
    if (status === 503) return 'The prediction model is currently unavailable. Please try again later.'
    if (!error.response) return 'The prediction service could not be reached. Check your connection and try again.'
  }
  return 'The prediction could not be completed. Please try again later.'
}

function explanationErrorMessage(error: unknown, method: 'SHAP' | 'LIME'): string {
  if (axios.isAxiosError(error)) {
    const status = error.response?.status
    if (status === 401) return 'Your session has expired. Please sign in again.'
    if (status === 403) return 'Your account does not have permission to generate this explanation.'
    if (status === 404) return 'The patient, record, or prediction was not found. Run a prediction for this record first.'
    if (status === 409) return 'The saved prediction no longer matches this record and model. Run a new prediction, then try again.'
    if (status === 503) return `Unable to generate the ${method} explanation because the model is unavailable or incompatible.`
    if (!error.response) return 'The explanation service could not be reached. Check your connection and try again.'
  }
  return `The ${method} explanation could not be generated. Please try again later.`
}

function formatPercentage(value: number): string {
  return `${value.toFixed(1)}%`
}

function consistencyErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const status = error.response?.status
    if (status === 401) return 'Your session has expired. Please sign in again.'
    if (status === 403) return 'Your account does not have permission to analyze explanation consistency.'
    if (status === 404) return 'The patient, record, or prediction was not found. Run a prediction for this record first.'
    if (status === 409) return 'The saved prediction no longer matches this record and model. Run a new prediction, then try again.'
    if ((status !== undefined && status >= 500) || status === 503) return 'The consistency analysis is temporarily unavailable. Please try again later.'
    if (!error.response) return 'The consistency service could not be reached. Check your connection and try again.'
  }
  return 'Explanation consistency could not be analyzed. Please try again later.'
}
