import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'

interface Validation {
  id: number
  analysisId: number
  validatorId: number
  status: string
  approved: boolean
  clinicalNotes?: string
  createdAt: string
  updatedAt: string
}

interface ValidationState {
  validations: Validation[]
  pendingValidations: Validation[]
  isLoading: boolean
  error: string | null
}

const initialState: ValidationState = {
  validations: [],
  pendingValidations: [],
  isLoading: false,
  error: null,
}

export const fetchMyValidations = createAsyncThunk(
  'validation/fetchMyValidations',
  async (params: { limit?: number; offset?: number } = {}) => {
    const searchParams = new URLSearchParams()
    if (params.limit) searchParams.append('limit', params.limit.toString())
    if (params.offset) searchParams.append('offset', params.offset.toString())

    const response = await fetch(`/api/v1/validations/my-validations?${searchParams}`, {
      headers: {
        Authorization: `Bearer ${localStorage.getItem('token')}`,
      },
      credentials: 'omit',
    })

    if (!response.ok) {
      throw new Error('Failed to fetch validations')
    }

    return response.json()
  }
)

export const submitValidation = createAsyncThunk(
  'validation/submitValidation',
  async (data: { validationId: number; validationData: Record<string, unknown> }) => {
    const response = await fetch(`/api/v1/validations/${data.validationId}/submit`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${localStorage.getItem('token')}`,
      },
      credentials: 'omit',
      body: JSON.stringify(data.validationData),
    })

    if (!response.ok) {
      throw new Error('Failed to submit validation')
    }

    return response.json()
  }
)

const validationSlice = createSlice({
  name: 'validation',
  initialState,
  reducers: {
    clearError: state => {
      state.error = null
    },
  },
  extraReducers: builder => {
    builder
      .addCase(fetchMyValidations.pending, state => {
        state.isLoading = true
        state.error = null
      })
      .addCase(fetchMyValidations.fulfilled, (state, action) => {
        state.isLoading = false
        state.validations = action.payload.validations
      })
      .addCase(fetchMyValidations.rejected, (state, action) => {
        state.isLoading = false
        state.error = action.error.message || 'Failed to fetch validations'
      })
      .addCase(submitValidation.pending, state => {
        state.isLoading = true
        state.error = null
      })
      .addCase(submitValidation.fulfilled, (state, action) => {
        state.isLoading = false
        const index = state.validations.findIndex(v => v.id === action.payload.id)
        if (index !== -1) {
          state.validations[index] = action.payload
        }
      })
      .addCase(submitValidation.rejected, (state, action) => {
        state.isLoading = false
        state.error = action.error.message || 'Failed to submit validation'
      })
  },
})

export const { clearError } = validationSlice.actions
export default validationSlice.reducer
