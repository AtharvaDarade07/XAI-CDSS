import { apiClient } from './api'
import type { Prediction } from '../types/api'

// The backend predicts only from the saved record; no feature values are sent here.
export async function predictPatientRecord(patientId: number, recordId: number): Promise<Prediction> {
  const response = await apiClient.post<Prediction>(`/patients/${patientId}/records/${recordId}/predict`)
  return response.data
}
