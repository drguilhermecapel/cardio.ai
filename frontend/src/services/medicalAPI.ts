// Medical APIs Integration for CardioAI Pro
// Integração com APIs médicas reais para dados de pacientes e ECG

import axios, { AxiosInstance } from 'axios'

// Assumindo que a configuração do apiClient já existe
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types
interface Patient {
  id: string
  mrn: string // Medical Record Number
  firstName: string
  lastName: string
  dateOfBirth: string
  gender: 'M' | 'F' | 'O'
  phone?: string
  email?: string
  address?: Address
  emergencyContact?: EmergencyContact
  insurance?: Insurance
  allergies?: Allergy[]
  medications?: Medication[]
  conditions?: MedicalCondition[]
  vitals?: VitalSigns
}

interface Address {
  street: string
  city: string
  state: string
  zipCode: string
  country: string
}

interface EmergencyContact {
  name: string
  relationship: string
  phone: string
}

interface Insurance {
  provider: string
  policyNumber: string
  groupNumber?: string
}

interface Allergy {
  allergen: string
  severity: 'mild' | 'moderate' | 'severe'
  reaction: string
}

interface Medication {
  name: string
  dosage: string
  frequency: string
  startDate: string
  endDate?: string
  prescribedBy: string
}

interface MedicalCondition {
  icd10Code: string
  description: string
  diagnosedDate: string
  status: 'active' | 'resolved' | 'chronic'
}

interface VitalSigns {
  bloodPressure: {
    systolic: number
    diastolic: number
    timestamp: string
  }
  heartRate: {
    bpm: number
    timestamp: string
  }
  temperature: {
    celsius: number
    timestamp: string
  }
  oxygenSaturation: {
    percentage: number
    timestamp: string
  }
  weight: {
    kg: number
    timestamp: string
  }
  height: {
    cm: number
    timestamp: string
  }
}

interface ECGData {
  id: string
  patientId: string
  timestamp: string
  duration: number
  sampleRate: number
  leads: Record<string, number[]>
  annotations?: ECGAnnotation[]
  analysis?: ECGAnalysis
  technician?: string
  device?: ECGDevice
}

interface ECGAnnotation {
  timestamp: number
  type: 'P' | 'QRS' | 'T' | 'artifact' | 'noise'
  confidence: number
  description?: string
}

interface ECGAnalysis {
  heartRate: number
  rhythm: string
  intervals: {
    PR: number
    QRS: number
    QT: number
    QTc: number
  }
  abnormalities: string[]
  interpretation: string
  confidence: number
  aiModel: string
  aiVersion: string
}

interface ECGDevice {
  manufacturer: string
  model: string
  serialNumber: string
  calibrationDate: string
}

// API Configuration
interface APIConfig {
  baseURL: string
  timeout: number
  retries: number
  apiKey?: string
  clientId?: string
  clientSecret?: string
}

// FHIR R4 Integration
class FHIRClient {
  private client: AxiosInstance

  constructor(config: APIConfig) {
    this.client = axios.create({
      baseURL: config.baseURL,
      timeout: config.timeout,
      headers: {
        'Content-Type': 'application/fhir+json',
        'Accept': 'application/fhir+json'
      }
    })

    // Add auth interceptor
    this.client.interceptors.request.use((config) => {
      const token = localStorage.getItem('fhir_token')
      if (token) {
        config.headers.Authorization = `Bearer ${token}`
      }
      return config
    })

    // Add retry interceptor
    this.client.interceptors.response.use(
      (response) => response,
      async (error) => {
        if (error.response?.status === 401) {
          await this.refreshToken()
          return this.client.request(error.config)
        }
        return Promise.reject(error)
      }
    )
  }

  private async refreshToken(): Promise<void> {
    // Implement OAuth2 token refresh
    const refreshToken = localStorage.getItem('fhir_refresh_token')
    if (!refreshToken) throw new Error('No refresh token available')

    const response = await axios.post('/oauth2/token', {
      grant_type: 'refresh_token',
      refresh_token: refreshToken
    })

    localStorage.setItem('fhir_token', response.data.access_token)
    localStorage.setItem('fhir_refresh_token', response.data.refresh_token)
  }

  async searchPatients(params: {
    name?: string
    identifier?: string
    birthdate?: string
    gender?: string
    limit?: number
  }): Promise<Patient[]> {
    const searchParams = new URLSearchParams()
    
    if (params.name) searchParams.append('name', params.name)
    if (params.identifier) searchParams.append('identifier', params.identifier)
    if (params.birthdate) searchParams.append('birthdate', params.birthdate)
    if (params.gender) searchParams.append('gender', params.gender)
    if (params.limit) searchParams.append('_count', params.limit.toString())

    const response = await this.client.get(`/Patient?${searchParams.toString()}`)
    return this.transformFHIRPatients(response.data.entry || [])
  }

  async getPatient(id: string): Promise<Patient> {
    const response = await this.client.get(`/Patient/${id}`)
    return this.transformFHIRPatient(response.data)
  }

  async createPatient(patient: Omit<Patient, 'id'>): Promise<Patient> {
    const fhirPatient = this.transformToFHIRPatient(patient)
    const response = await this.client.post('/Patient', fhirPatient)
    return this.transformFHIRPatient(response.data)
  }

  async updatePatient(id: string, patient: Partial<Patient>): Promise<Patient> {
    const fhirPatient = this.transformToFHIRPatient(patient)
    const response = await this.client.put(`/Patient/${id}`, fhirPatient)
    return this.transformFHIRPatient(response.data)
  }

  async getObservations(patientId: string, category?: string): Promise<VitalSigns[]> {
    const params = new URLSearchParams({
      patient: patientId,
      category: category || 'vital-signs',
      _sort: '-date'
    })

    const response = await this.client.get(`/Observation?${params.toString()}`)
    return this.transformFHIRObservations(response.data.entry || [])
  }

  private transformFHIRPatients(entries: any[]): Patient[] {
    return entries.map(entry => this.transformFHIRPatient(entry.resource))
  }

  private transformFHIRPatient(fhirPatient: any): Patient {
    return {
      id: fhirPatient.id,
      mrn: fhirPatient.identifier?.[0]?.value || '',
      firstName: fhirPatient.name?.[0]?.given?.[0] || '',
      lastName: fhirPatient.name?.[0]?.family || '',
      dateOfBirth: fhirPatient.birthDate || '',
      gender: fhirPatient.gender?.toUpperCase() || 'O',
      phone: fhirPatient.telecom?.find((t: any) => t.system === 'phone')?.value,
      email: fhirPatient.telecom?.find((t: any) => t.system === 'email')?.value,
      address: fhirPatient.address?.[0] ? {
        street: fhirPatient.address[0].line?.join(' ') || '',
        city: fhirPatient.address[0].city || '',
        state: fhirPatient.address[0].state || '',
        zipCode: fhirPatient.address[0].postalCode || '',
        country: fhirPatient.address[0].country || ''
      } : undefined
    }
  }

  private transformToFHIRPatient(patient: Partial<Patient>): any {
    return {
      resourceType: 'Patient',
      identifier: patient.mrn ? [{
        system: 'http://hospital.example.org/mrn',
        value: patient.mrn
      }] : undefined,
      name: [{
        given: [patient.firstName],
        family: patient.lastName
      }],
      birthDate: patient.dateOfBirth,
      gender: patient.gender?.toLowerCase(),
      telecom: [
        ...(patient.phone ? [{ system: 'phone', value: patient.phone }] : []),
        ...(patient.email ? [{ system: 'email', value: patient.email }] : [])
      ],
      address: patient.address ? [{
        line: [patient.address.street],
        city: patient.address.city,
        state: patient.address.state,
        postalCode: patient.address.zipCode,
        country: patient.address.country
      }] : undefined
    }
  }

  private transformFHIRObservations(entries: any[]): VitalSigns[] {
    // Transform FHIR observations to VitalSigns format
    return entries.map(entry => {
      const _obs = entry.resource
      // Implementation depends on specific FHIR observation structure
      return {} as VitalSigns
    })
  }
}

// HL7 Integration
class HL7Client {
  private config: APIConfig

  constructor(config: APIConfig) {
    this.config = config
  }

  async sendADT(patient: Patient, eventType: 'A01' | 'A02' | 'A03' | 'A04' | 'A08'): Promise<boolean> {
    const hl7Message = this.buildADTMessage(patient, eventType)
    
    try {
      const response = await axios.post(`${this.config.baseURL}/hl7/adt`, {
        message: hl7Message
      }, {
        headers: {
          'Content-Type': 'application/x-hl7',
          'Authorization': `Bearer ${this.config.apiKey}`
        }
      })
      
      return response.status === 200
    } catch (error) {
      console.error('HL7 ADT send failed:', error)
      return false
    }
  }

  async sendORU(ecgData: ECGData): Promise<boolean> {
    const hl7Message = this.buildORUMessage(ecgData)
    
    try {
      const response = await axios.post(`${this.config.baseURL}/hl7/oru`, {
        message: hl7Message
      }, {
        headers: {
          'Content-Type': 'application/x-hl7',
          'Authorization': `Bearer ${this.config.apiKey}`
        }
      })
      
      return response.status === 200
    } catch (error) {
      console.error('HL7 ORU send failed:', error)
      return false
    }
  }

  private buildADTMessage(patient: Patient, eventType: string): string {
    const timestamp = new Date().toISOString().replace(/[-:]/g, '').split('.')[0]
    
    return [
      `MSH|^~\\&|CARDIOAI|HOSPITAL|HIS|HOSPITAL|${timestamp}||ADT^${eventType}|${Date.now()}|P|2.5`,
      `EVN|${eventType}|${timestamp}`,
      `PID|1||${patient.mrn}^^^HOSPITAL^MR||${patient.lastName}^${patient.firstName}||${patient.dateOfBirth}|${patient.gender}|||${patient.address?.street}^^${patient.address?.city}^${patient.address?.state}^${patient.address?.zipCode}||${patient.phone}`,
      `PV1|1|I|ICU^101^1|||||||||||||||||||||||||||||||||||${timestamp}`
    ].join('\r')
  }

  private buildORUMessage(ecgData: ECGData): string {
    const timestamp = new Date().toISOString().replace(/[-:]/g, '').split('.')[0]
    
    return [
      `MSH|^~\\&|CARDIOAI|HOSPITAL|HIS|HOSPITAL|${timestamp}||ORU^R01|${Date.now()}|P|2.5`,
      `PID|1||${ecgData.patientId}^^^HOSPITAL^MR`,
      `OBR|1||${ecgData.id}|ECG^Electrocardiogram^LN|||${timestamp}`,
      `OBX|1|ED|ECG^Electrocardiogram^LN||${JSON.stringify(ecgData.leads)}||||||F`
    ].join('\r')
  }
}

// DICOM Integration for ECG
class DICOMClient {
  private config: APIConfig

  constructor(config: APIConfig) {
    this.config = config
  }

  async storeECG(ecgData: ECGData): Promise<string> {
    const dicomData = this.convertToDICOM(ecgData)
    
    try {
      const response = await axios.post(`${this.config.baseURL}/dicom/store`, dicomData, {
        headers: {
          'Content-Type': 'application/dicom',
          'Authorization': `Bearer ${this.config.apiKey}`
        }
      })
      
      return response.data.sopInstanceUID
    } catch (error) {
      console.error('DICOM store failed:', error)
      throw error
    }
  }

  async retrieveECG(sopInstanceUID: string): Promise<ECGData> {
    try {
      const response = await axios.get(`${this.config.baseURL}/dicom/retrieve/${sopInstanceUID}`, {
        headers: {
          'Authorization': `Bearer ${this.config.apiKey}`
        }
      })
      
      return this.convertFromDICOM(response.data)
    } catch (error) {
      console.error('DICOM retrieve failed:', error)
      throw error
    }
  }

  private convertToDICOM(ecgData: ECGData): any {
    // Convert ECG data to DICOM format
    return {
      sopClassUID: '1.2.840.10008.5.1.4.1.1.9.1.1', // 12-lead ECG Waveform Storage
      sopInstanceUID: ecgData.id,
      patientID: ecgData.patientId,
      studyDate: ecgData.timestamp.split('T')[0].replace(/-/g, ''),
      studyTime: ecgData.timestamp.split('T')[1].replace(/:/g, '').split('.')[0],
      waveformData: ecgData.leads,
      sampleRate: ecgData.sampleRate,
      duration: ecgData.duration
    }
  }

  private convertFromDICOM(dicomData: any): ECGData {
    // Convert DICOM data to ECG format
    return {
      id: dicomData.sopInstanceUID,
      patientId: dicomData.patientID,
      timestamp: `${dicomData.studyDate.slice(0,4)}-${dicomData.studyDate.slice(4,6)}-${dicomData.studyDate.slice(6,8)}T${dicomData.studyTime.slice(0,2)}:${dicomData.studyTime.slice(2,4)}:${dicomData.studyTime.slice(4,6)}`,
      duration: dicomData.duration,
      sampleRate: dicomData.sampleRate,
      leads: dicomData.waveformData
    }
  }
}

// Main Medical API Service
export class MedicalAPIService {
  private fhirClient: FHIRClient
  private hl7Client: HL7Client
  private dicomClient: DICOMClient

  constructor() {
    // Initialize clients with configuration
    this.fhirClient = new FHIRClient({
      baseURL: process.env.VITE_FHIR_BASE_URL || 'https://fhir.example.org',
      timeout: 30000,
      retries: 3
    })

    this.hl7Client = new HL7Client({
      baseURL: process.env.VITE_HL7_BASE_URL || 'https://hl7.example.org',
      timeout: 30000,
      retries: 3,
      apiKey: process.env.VITE_HL7_API_KEY
    })

    this.dicomClient = new DICOMClient({
      baseURL: process.env.VITE_DICOM_BASE_URL || 'https://dicom.example.org',
      timeout: 60000,
      retries: 3,
      apiKey: process.env.VITE_DICOM_API_KEY
    })
  }

  // Patient operations
  async searchPatients(query: string): Promise<Patient[]> {
    return this.fhirClient.searchPatients({ name: query, limit: 50 })
  }

  async getPatient(id: string): Promise<Patient> {
    return this.fhirClient.getPatient(id)
  }

  async createPatient(patient: Omit<Patient, 'id'>): Promise<Patient> {
    const createdPatient = await this.fhirClient.createPatient(patient)
    
    // Send ADT message for patient admission
    await this.hl7Client.sendADT(createdPatient, 'A04')
    
    return createdPatient
  }

  async updatePatient(id: string, patient: Partial<Patient>): Promise<Patient> {
    const updatedPatient = await this.fhirClient.updatePatient(id, patient)
    
    // Send ADT message for patient update
    await this.hl7Client.sendADT(updatedPatient, 'A08')
    
    return updatedPatient
  }

  // ECG operations
  async storeECG(ecgData: ECGData): Promise<string> {
    // Store in DICOM
    const sopInstanceUID = await this.dicomClient.storeECG(ecgData)
    
    // Send HL7 ORU message
    await this.hl7Client.sendORU(ecgData)
    
    return sopInstanceUID
  }

  async retrieveECG(id: string): Promise<ECGData> {
    return this.dicomClient.retrieveECG(id)
  }

  // Vital signs
  async getVitalSigns(patientId: string): Promise<VitalSigns[]> {
    return this.fhirClient.getObservations(patientId, 'vital-signs')
  }

  // Drug interaction checking
  async checkDrugInteractions(medications: string[]): Promise<any> {
    try {
      const response = await axios.post('/api/drug-interactions', {
        medications
      }, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('cardioai_token')}`
        }
      })
      
      return response.data
    } catch (error) {
      console.error('Drug interaction check failed:', error)
      throw error
    }
  }

  // Clinical decision support
  async getClinicalGuidelines(condition: string): Promise<any> {
    try {
      const response = await axios.get(`/api/clinical-guidelines/${condition}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('cardioai_token')}`
        }
      })
      
      return response.data
    } catch (error) {
      console.error('Clinical guidelines fetch failed:', error)
      throw error
    }
  }
}

// Singleton instance
export const medicalAPI = new MedicalAPIService()

export default medicalAPI



// --- INÍCIO DA ADIÇÃO PARA ECG-DIGITISER ---

/**
 * Define a estrutura de dados esperada do backend para um ECG digitalizado.
 */
export interface DigitizedECGData {
  signal_data: number[][];
  sampling_rate: number;
  lead_names: string[];
}

/**
 * Envia uma imagem de ECG para o backend para digitalização.
 * @param imageFile O arquivo de imagem (PNG, JPG) a ser processado.
 * @param onUploadProgress Callback opcional para monitorar o progresso do upload.
 * @returns Uma promessa que resolve para os dados do ECG digitalizado.
 */
export const digitizeECGImage = async (
  imageFile: File,
  onUploadProgress?: (progressEvent: any) => void
): Promise<DigitizedECGData> => {
  const formData = new FormData();
  formData.append('file', imageFile);

  try {
    const response = await apiClient.post<DigitizedECGData>('/ecg-image/digitize', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress,
    });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error) && error.response) {
      // Propaga a mensagem de erro detalhada do backend
      throw new Error(error.response.data.detail || 'Falha ao digitalizar a imagem.');
    }
    // Erro genérico para problemas de rede ou outros
    throw new Error('Não foi possível conectar ao servidor. Verifique sua conexão de rede.');
  }
};

// --- FIM DA ADIÇÃO PARA ECG-DIGITISER ---

