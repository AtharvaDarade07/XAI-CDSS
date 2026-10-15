import { apiClient } from './api'
import type {
  Patient,
  PatientCreateRequest,
  PatientListResponse,
  PatientRecord,
  PatientRecordCreateRequest,
} from '../types/api'

// All patient access goes through the shared client, which attaches the session token.
export async function listPatients(skip = 0, limit = 20): Promise<PatientListResponse> {
  const response = await apiClient.get<PatientListResponse>('/patients', { params: { skip, limit } })
  return response.data
}

export async function getPatient(patientId: number): Promise<Patient> {
  const response = await apiClient.get<Patient>(`/patients/${patientId}`)
  return response.data
}

export async function createPatient(payload: PatientCreateRequest): Promise<Patient> {
  const response = await apiClient.post<Patient>('/patients', payload)
  return response.data
}

export async function listPatientRecords(patientId: number, skip = 0, limit = 50): Promise<PatientRecord[]> {
  const response = await apiClient.get<PatientRecord[]>(`/patients/${patientId}/records`, { params: { skip, limit } })
  return response.data
}

export async function getPatientRecord(patientId: number, recordId: number): Promise<PatientRecord> {
  const response = await apiClient.get<PatientRecord>(`/patients/${patientId}/records/${recordId}`)
  return response.data
}

export async function createPatientRecord(patientId: number, payload: PatientRecordCreateRequest): Promise<PatientRecord> {
  const response = await apiClient.post<PatientRecord>(`/patients/${patientId}/records`, payload)
  return response.data
}
