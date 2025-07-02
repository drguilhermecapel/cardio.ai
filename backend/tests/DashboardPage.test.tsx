import React from 'react'
import { render, screen } from '@testing-library/react'
import { Provider } from 'react-redux'
import { BrowserRouter } from 'react-router-dom'
import { ThemeProvider } from '@mui/material/styles'
import { describe, it, expect, vi } from 'vitest'
import { theme } from '../../theme'
import DashboardPage from '../DashboardPage'

vi.mock('../../hooks/redux', () => ({
  useAppDispatch: vi.fn(() => vi.fn()),
  useAppSelector: vi.fn(selector => {
    const mockState = {
      ecg: {
        analyses: [],
        isLoading: false,
        error: null,
        currentAnalysis: null,
        uploadProgress: 0,
      },
      notification: {
        unreadCount: 0,
        notifications: [],
        isLoading: false,
        error: null,
      },
      auth: {
        isAuthenticated: true,
        user: {
          id: 1,
          username: 'testuser',
          email: 'test@example.com',
          firstName: 'Test',
          lastName: 'User',
          role: 'physician',
          isActive: true,
        },
        token: 'mock-token',
        refreshToken: 'mock-refresh-token',
        isLoading: false,
        error: null,
      },
    }
    return selector(mockState)
  }),
}))

import { store } from '../../store'

const renderWithProviders = (component: React.ReactElement): ReturnType<typeof render> => {
  return render(
    <Provider store={store}>
      <BrowserRouter>
        <ThemeProvider theme={theme}>{component}</ThemeProvider>
      </BrowserRouter>
    </Provider>
  )
}

describe('DashboardPage', () => {
  it('renders dashboard page', () => {
    renderWithProviders(<DashboardPage />)
    expect(screen.getByText('Dashboard')).toBeDefined()
  })

  it('displays dashboard metrics', () => {
    renderWithProviders(<DashboardPage />)
    expect(screen.getByText('Total Analyses')).toBeDefined()
    expect(screen.getByText('Pending')).toBeDefined()
  })
})
