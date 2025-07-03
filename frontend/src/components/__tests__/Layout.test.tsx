import React from 'react'
import { render, screen } from '@testing-library/react'
import { Provider } from 'react-redux'
import { BrowserRouter } from 'react-router-dom'
import { store } from '../../store'
import { AuthProvider } from '../../contexts/AuthContext'
import Layout from '../Layout'

jest.mock('../../hooks/useAuth', () => ({
  useAuth: jest.fn(() => ({
    user: { username: 'Test User' },
    isAuthenticated: true,
    login: jest.fn(),
    logout: jest.fn(),
  })),
}))

jest.mock('../../hooks/redux', () => ({
  useAppSelector: jest.fn(() => ({
    auth: {
      isAuthenticated: true,
      user: { firstName: 'Test', lastName: 'User', role: 'physician' },
    },
    ecg: {
      analyses: [],
      isLoading: false,
    },
    notification: {
      unreadCount: 0,
    },
  })),
  useAppDispatch: jest.fn(() => jest.fn()),
}))

const renderWithProviders = (component: React.ReactElement): ReturnType<typeof render> => {
  return render(
    <Provider store={store}>
      <BrowserRouter>
        <AuthProvider>{component}</AuthProvider>
      </BrowserRouter>
    </Provider>
  )
}

describe('Layout', () => {
  it('renders layout component with navigation', () => {
    renderWithProviders(<Layout />)
    expect(screen.getByRole('main')).toBeDefined()
  })

  it('displays navigation elements', () => {
    renderWithProviders(<Layout />)
    expect(screen.getByText('Dashboard')).toBeDefined()
    expect(screen.getByText('SPEI - Sistema EMR')).toBeDefined()
  })
})
