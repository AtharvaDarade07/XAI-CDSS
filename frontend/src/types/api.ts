export interface User {
  id: number
  username: string
  email: string
  role: string
  is_active: boolean
}

export interface AuthLoginRequest {
  username: string
  password: string
}

export interface AuthLoginResponse {
  access_token: string
  token_type: 'bearer'
  expires_in: number
}

export interface AuthState {
  user: User | null
  token: string | null
  isAuthenticated: boolean
  isLoading: boolean
  login: (username: string, password: string) => Promise<User>
  logout: () => void
}

export interface Patient {
  id: number
  patient_code: string
  created_at: string
  updated_at: string
}

export interface PatientListResponse {
  items: Patient[]
  skip: number
  limit: number
  total: number
}

export interface PatientCreateRequest {
  patient_code: string
}

export interface PatientRecord {
  id: number
  patient_id: number
  pregnancies: number | null
  glucose: number | null
  blood_pressure: number | null
  skin_thickness: number | null
  insulin: number | null
  bmi: number | null
  diabetes_pedigree_function: number | null
  age: number | null
  created_at: string
}

export type PatientRecordCreateRequest = Omit<PatientRecord, 'id' | 'patient_id' | 'created_at'>

export interface Prediction {
  id: number
  patient_id: number
  patient_record_id: number
  model_version: string
  predicted_class: 0 | 1
  predicted_probability: number
  created_at: string
}

export interface ExplanationFeature {
  feature: string
  feature_value: number | null
  contribution: number
  direction: string
  rank: number
}

export interface ExplanationResponse {
  prediction_id: number
  patient_id: number
  patient_record_id: number
  method: string
  predicted_class: 0 | 1
  predicted_probability: number
  features: ExplanationFeature[]
  created_at: string
  summary: string
  output_space: string
  shap_base_value: number | null
  shap_contribution_sum: number | null
  shap_model_output: number | null
  shap_additivity_error: number | null
  lime_configuration: Record<string, number> | null
}

export interface ExplanationConsistencyResponse {
  prediction_id: number
  patient_id: number
  patient_record_id: number
  top3_overlap_count: number
  top3_overlap_percentage: number
  top5_overlap_count: number
  top5_overlap_percentage: number
  spearman_correlation: number
  spearman_p_value: number
  direction_agreement_count: number
  direction_disagreement_count: number
  direction_agreement_percentage: number
  shared_feature_count: number
  shap_top_features: string[]
  lime_top_features: string[]
  interpretation: string
  created_at: string
}
