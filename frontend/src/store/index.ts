import { configureStore } from '@reduxjs/toolkit'

import authReducer from './slices/authSlice'
import ecgReducer from './slices/ecgSlice'
import patientReducer from './slices/patientSlice'
import validationReducer from './slices/validationSlice'
import notificationReducer from './slices/notificationSlice'

export const store = configureStore({
  reducer: {
    auth: authReducer,
    ecg: ecgReducer,
    patient: patientReducer,
    validation: validationReducer,
    notification: notificationReducer,
  },
  middleware: getDefaultMiddleware =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST'],
      },
    }),
})

export type RootState = ReturnType<typeof store.getState>
export type AppDispatch = typeof store.dispatch
