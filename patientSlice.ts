import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'

interface Patient {
  id: number
  patientId: string
  firstName: string
  lastName: string
  dateOfBirth: string
  gender: string
  phone?: string
  email?: string
}

interface PatientState {
  patients: Patient[]
  currentPatient: Patient | null
  isLoading: boolean
  error: string | null
}

const initialState: PatientState = {
  patients: [],
  currentPatient: null,
  isLoading: false,
  error: null,
}

export const fetchPatients = createAsyncThunk(
  'patient/fetchPatients',
  async (params: { limit?: number; offset?: number } = {}) => {
    const searchParams = new URLSearchParams()
    if (params.limit) searchParams.append('limit', params.limit.toString())
    if (params.offset) searchParams.append('offset', params.offset.toString())

    const response = await fetch(`/api/v1/patients/?${searchParams}`, {
      headers: {
        Authorization: `Bearer ${localStorage.getItem('token')}`,
      },
      credentials: 'omit',
    })

    if (!response.ok) {
      throw new Error('Failed to fetch patients')
    }

    return response.json()
  }
)

export const createPatient = createAsyncThunk(
  'patient/createPatient',
  async (patientData: Omit<Patient, 'id'>) => {
    const response = await fetch('/api/v1/patients/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${localStorage.getItem('token')}`,
      },
      credentials: 'omit',
      body: JSON.stringify(patientData),
    })

    if (!response.ok) {
      throw new Error('Failed to create patient')
    }

    return response.json()
  }
)

const patientSlice = createSlice({
  name: 'patient',
  initialState,
  reducers: {
    clearError: state => {
      state.error = null
    },
    setCurrentPatient: (state, action) => {
      state.currentPatient = action.payload
    },
  },
  extraReducers: builder => {
    builder
      .addCase(fetchPatients.pending, state => {
        state.isLoading = true
        state.error = null
      })
      .addCase(fetchPatients.fulfilled, (state, action) => {
        state.isLoading = false
        state.patients = action.payload.patients
      })
      .addCase(fetchPatients.rejected, (state, action) => {
        state.isLoading = false
        state.error = action.error.message || 'Failed to fetch patients'
      })
      .addCase(createPatient.pending, state => {
        state.isLoading = true
        state.error = null
      })
      .addCase(createPatient.fulfilled, (state, action) => {
        state.isLoading = false
        state.patients.push(action.payload)
      })
      .addCase(createPatient.rejected, (state, action) => {
        state.isLoading = false
        state.error = action.error.message || 'Failed to create patient'
      })
  },
})

export const { clearError, setCurrentPatient } = patientSlice.actions
export default patientSlice.reducer
