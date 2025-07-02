import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit'

interface ECGAnalysis {
  id: number
  analysisId: string
  patientId: number
  status: string
  diagnosis: string
  clinicalUrgency: string
  confidence: number
  createdAt: string
  updatedAt: string
}

interface ECGState {
  analyses: ECGAnalysis[]
  currentAnalysis: ECGAnalysis | null
  isLoading: boolean
  error: string | null
  uploadProgress: number
}

const initialState: ECGState = {
  analyses: [],
  currentAnalysis: null,
  isLoading: false,
  error: null,
  uploadProgress: 0,
}

export const uploadECG = createAsyncThunk(
  'ecg/upload',
  async (data: { patientId: number; file: File }) => {
    const formData = new FormData()
    formData.append('patient_id', data.patientId.toString())
    formData.append('file', data.file)

    const response = await fetch('/api/v1/ecg/upload', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${localStorage.getItem('token')}`,
      },
      credentials: 'omit',
      body: formData,
    })

    if (!response.ok) {
      throw new Error('Upload failed')
    }

    return response.json()
  }
)

export const fetchAnalyses = createAsyncThunk(
  'ecg/fetchAnalyses',
  async (params: { limit?: number; offset?: number } = {}) => {
    const searchParams = new URLSearchParams()
    if (params.limit) searchParams.append('limit', params.limit.toString())
    if (params.offset) searchParams.append('offset', params.offset.toString())

    const response = await fetch(`/api/v1/ecg/?${searchParams}`, {
      headers: {
        Authorization: `Bearer ${localStorage.getItem('token')}`,
      },
      credentials: 'omit',
    })

    if (!response.ok) {
      throw new Error('Failed to fetch analyses')
    }

    return response.json()
  }
)

export const fetchAnalysis = createAsyncThunk('ecg/fetchAnalysis', async (analysisId: string) => {
  const response = await fetch(`/api/v1/ecg/${analysisId}`, {
    headers: {
      Authorization: `Bearer ${localStorage.getItem('token')}`,
    },
    credentials: 'omit',
  })

  if (!response.ok) {
    throw new Error('Failed to fetch analysis')
  }

  return response.json()
})

const ecgSlice = createSlice({
  name: 'ecg',
  initialState,
  reducers: {
    clearError: state => {
      state.error = null
    },
    setUploadProgress: (state, action: PayloadAction<number>) => {
      state.uploadProgress = action.payload
    },
    clearCurrentAnalysis: state => {
      state.currentAnalysis = null
    },
  },
  extraReducers: builder => {
    builder
      .addCase(uploadECG.pending, state => {
        state.isLoading = true
        state.error = null
        state.uploadProgress = 0
      })
      .addCase(uploadECG.fulfilled, state => {
        state.isLoading = false
        state.uploadProgress = 100
      })
      .addCase(uploadECG.rejected, (state, action) => {
        state.isLoading = false
        state.error = action.error.message || 'Upload failed'
        state.uploadProgress = 0
      })
      .addCase(fetchAnalyses.pending, state => {
        state.isLoading = true
        state.error = null
      })
      .addCase(fetchAnalyses.fulfilled, (state, action) => {
        state.isLoading = false
        state.analyses = action.payload.analyses
      })
      .addCase(fetchAnalyses.rejected, (state, action) => {
        state.isLoading = false
        state.error = action.error.message || 'Failed to fetch analyses'
      })
      .addCase(fetchAnalysis.pending, state => {
        state.isLoading = true
        state.error = null
      })
      .addCase(fetchAnalysis.fulfilled, (state, action) => {
        state.isLoading = false
        state.currentAnalysis = action.payload
      })
      .addCase(fetchAnalysis.rejected, (state, action) => {
        state.isLoading = false
        state.error = action.error.message || 'Failed to fetch analysis'
      })
  },
})

export const { clearError, setUploadProgress, clearCurrentAnalysis } = ecgSlice.actions
export default ecgSlice.reducer
