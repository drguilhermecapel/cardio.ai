// Advanced Authentication System for CardioAI Pro
// Implementação de autenticação avançada com JWT, 2FA e biometria

import React, { useReducer, useEffect } from 'react'
import { jwtDecode } from 'jwt-decode'
import { AuthContext, AuthContextType, User, LoginResult } from './AuthContextDefinition'

interface AuthState {
  user: User | null
  token: string | null
  refreshToken: string | null
  isAuthenticated: boolean
  isLoading: boolean
  mfaRequired: boolean
  biometricAvailable: boolean
  sessionExpiry: number | null
}

// Auth Actions
type AuthAction =
  | { type: 'LOGIN_START' }
  | { type: 'LOGIN_SUCCESS'; payload: { user: User; token: string; refreshToken: string } }
  | { type: 'LOGIN_FAILURE'; payload: string }
  | { type: 'MFA_REQUIRED' }
  | { type: 'MFA_SUCCESS'; payload: { user: User; token: string; refreshToken: string } }
  | { type: 'LOGOUT' }
  | { type: 'REFRESH_TOKEN'; payload: { token: string; refreshToken: string } }
  | { type: 'UPDATE_USER'; payload: Partial<User> }
  | { type: 'SET_BIOMETRIC_AVAILABLE'; payload: boolean }
  | { type: 'SESSION_EXPIRED' }

// Auth Reducer
const authReducer = (state: AuthState, action: AuthAction): AuthState => {
  switch (action.type) {
    case 'LOGIN_START':
      return { ...state, isLoading: true, mfaRequired: false }
    
    case 'LOGIN_SUCCESS':
    case 'MFA_SUCCESS':
      const { user, token, refreshToken } = action.payload
      const decoded = jwtDecode(token) as any
      return {
        ...state,
        user,
        token,
        refreshToken,
        isAuthenticated: true,
        isLoading: false,
        mfaRequired: false,
        sessionExpiry: decoded.exp * 1000
      }
    
    case 'LOGIN_FAILURE':
      return {
        ...state,
        isLoading: false,
        mfaRequired: false,
        user: null,
        token: null,
        refreshToken: null,
        isAuthenticated: false
      }
    
    case 'MFA_REQUIRED':
      return { ...state, isLoading: false, mfaRequired: true }
    
    case 'LOGOUT':
    case 'SESSION_EXPIRED':
      return {
        ...state,
        user: null,
        token: null,
        refreshToken: null,
        isAuthenticated: false,
        isLoading: false,
        mfaRequired: false,
        sessionExpiry: null
      }
    
    case 'REFRESH_TOKEN':
      const decodedRefresh = jwtDecode(action.payload.token) as any
      return {
        ...state,
        token: action.payload.token,
        refreshToken: action.payload.refreshToken,
        sessionExpiry: decodedRefresh.exp * 1000
      }
    
    case 'UPDATE_USER':
      return {
        ...state,
        user: state.user ? { ...state.user, ...action.payload } : null
      }
    
    case 'SET_BIOMETRIC_AVAILABLE':
      return { ...state, biometricAvailable: action.payload }
    
    default:
      return state
  }
}

// Initial State
const initialState: AuthState = {
  user: null,
  token: null,
  refreshToken: null,
  isAuthenticated: false,
  isLoading: false,
  mfaRequired: false,
  biometricAvailable: false,
  sessionExpiry: null
}

// Auth Context
const AuthContext = createContext<AuthContextType | undefined>(undefined)

// Auth Provider
export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(authReducer, initialState)

  // Check for stored auth on mount
  useEffect(() => {
    const storedToken = localStorage.getItem('cardioai_token')
    const storedRefreshToken = localStorage.getItem('cardioai_refresh_token')
    const storedUser = localStorage.getItem('cardioai_user')

    if (storedToken && storedRefreshToken && storedUser) {
      try {
        const decoded = jwtDecode(storedToken) as any
        const now = Date.now()
        
        if (decoded.exp * 1000 > now) {
          // Token still valid
          dispatch({
            type: 'LOGIN_SUCCESS',
            payload: {
              user: JSON.parse(storedUser),
              token: storedToken,
              refreshToken: storedRefreshToken
            }
          })
        } else {
          // Token expired, try refresh
          refreshAuth()
        }
      } catch (error) {
        // Invalid token, clear storage
        localStorage.removeItem('cardioai_token')
        localStorage.removeItem('cardioai_refresh_token')
        localStorage.removeItem('cardioai_user')
      }
    }

    // Check biometric availability
    checkBiometricAvailability()
  }, [])

  // Session expiry check
  useEffect(() => {
    if (state.sessionExpiry) {
      const timeUntilExpiry = state.sessionExpiry - Date.now()
      
      if (timeUntilExpiry > 0) {
        const timer = setTimeout(() => {
          dispatch({ type: 'SESSION_EXPIRED' })
          localStorage.clear()
        }, timeUntilExpiry)
        
        return () => clearTimeout(timer)
      }
    }
  }, [state.sessionExpiry])

  // Auto-refresh token
  useEffect(() => {
    if (state.isAuthenticated && state.sessionExpiry) {
      const refreshTime = state.sessionExpiry - Date.now() - 5 * 60 * 1000 // 5 minutes before expiry
      
      if (refreshTime > 0) {
        const timer = setTimeout(() => {
          refreshAuth()
        }, refreshTime)
        
        return () => clearTimeout(timer)
      }
    }
  }, [state.sessionExpiry])

  const checkBiometricAvailability = async () => {
    if ('credentials' in navigator && 'create' in navigator.credentials) {
      try {
        const available = await (navigator.credentials as any).isUserVerifyingPlatformAuthenticatorAvailable()
        dispatch({ type: 'SET_BIOMETRIC_AVAILABLE', payload: available })
      } catch (error) {
        dispatch({ type: 'SET_BIOMETRIC_AVAILABLE', payload: false })
      }
    }
  }

  const login = async (email: string, password: string): Promise<LoginResult> => {
    dispatch({ type: 'LOGIN_START' })

    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      })

      const data = await response.json()

      if (!response.ok) {
        dispatch({ type: 'LOGIN_FAILURE', payload: data.error })
        return { success: false, error: data.error }
      }

      if (data.requiresMFA) {
        dispatch({ type: 'MFA_REQUIRED' })
        return { success: false, requiresMFA: true }
      }

      // Store auth data
      localStorage.setItem('cardioai_token', data.token)
      localStorage.setItem('cardioai_refresh_token', data.refreshToken)
      localStorage.setItem('cardioai_user', JSON.stringify(data.user))

      dispatch({
        type: 'LOGIN_SUCCESS',
        payload: {
          user: data.user,
          token: data.token,
          refreshToken: data.refreshToken
        }
      })

      return { success: true, user: data.user, token: data.token }
    } catch (error) {
      const errorMessage = 'Erro de conexão. Tente novamente.'
      dispatch({ type: 'LOGIN_FAILURE', payload: errorMessage })
      return { success: false, error: errorMessage }
    }
  }

  const loginWithBiometric = async (): Promise<LoginResult> => {
    if (!state.biometricAvailable) {
      return { success: false, error: 'Autenticação biométrica não disponível' }
    }

    try {
      // Get stored credential ID
      const credentialId = localStorage.getItem('cardioai_biometric_id')
      if (!credentialId) {
        return { success: false, error: 'Credencial biométrica não configurada' }
      }

      // Create assertion options
      const assertionOptions = {
        challenge: new Uint8Array(32),
        allowCredentials: [{
          id: new Uint8Array(Buffer.from(credentialId, 'base64')),
          type: 'public-key' as const
        }],
        userVerification: 'required' as const
      }

      // Get assertion
      const assertion = await (navigator.credentials as any).get({
        publicKey: assertionOptions
      })

      if (!assertion) {
        return { success: false, error: 'Autenticação biométrica cancelada' }
      }

      // Verify with server
      const response = await fetch('/api/auth/biometric-login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          credentialId,
          assertion: {
            id: assertion.id,
            rawId: Array.from(new Uint8Array(assertion.rawId)),
            response: {
              authenticatorData: Array.from(new Uint8Array(assertion.response.authenticatorData)),
              clientDataJSON: Array.from(new Uint8Array(assertion.response.clientDataJSON)),
              signature: Array.from(new Uint8Array(assertion.response.signature))
            }
          }
        })
      })

      const data = await response.json()

      if (!response.ok) {
        return { success: false, error: data.error }
      }

      // Store auth data
      localStorage.setItem('cardioai_token', data.token)
      localStorage.setItem('cardioai_refresh_token', data.refreshToken)
      localStorage.setItem('cardioai_user', JSON.stringify(data.user))

      dispatch({
        type: 'LOGIN_SUCCESS',
        payload: {
          user: data.user,
          token: data.token,
          refreshToken: data.refreshToken
        }
      })

      return { success: true, user: data.user, token: data.token }
    } catch (error) {
      return { success: false, error: 'Erro na autenticação biométrica' }
    }
  }

  const logout = () => {
    // Revoke token on server
    if (state.token) {
      fetch('/api/auth/logout', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${state.token}`,
          'Content-Type': 'application/json'
        }
      }).catch(() => {}) // Ignore errors
    }

    // Clear local storage
    localStorage.removeItem('cardioai_token')
    localStorage.removeItem('cardioai_refresh_token')
    localStorage.removeItem('cardioai_user')

    dispatch({ type: 'LOGOUT' })
  }

  const refreshAuth = async (): Promise<boolean> => {
    const refreshToken = localStorage.getItem('cardioai_refresh_token')
    
    if (!refreshToken) {
      return false
    }

    try {
      const response = await fetch('/api/auth/refresh', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refreshToken })
      })

      if (!response.ok) {
        logout()
        return false
      }

      const data = await response.json()

      localStorage.setItem('cardioai_token', data.token)
      localStorage.setItem('cardioai_refresh_token', data.refreshToken)

      dispatch({
        type: 'REFRESH_TOKEN',
        payload: {
          token: data.token,
          refreshToken: data.refreshToken
        }
      })

      return true
    } catch (error) {
      logout()
      return false
    }
  }

  const verifyMFA = async (code: string): Promise<boolean> => {
    try {
      const response = await fetch('/api/auth/verify-mfa', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code })
      })

      const data = await response.json()

      if (!response.ok) {
        return false
      }

      localStorage.setItem('cardioai_token', data.token)
      localStorage.setItem('cardioai_refresh_token', data.refreshToken)
      localStorage.setItem('cardioai_user', JSON.stringify(data.user))

      dispatch({
        type: 'MFA_SUCCESS',
        payload: {
          user: data.user,
          token: data.token,
          refreshToken: data.refreshToken
        }
      })

      return true
    } catch (error) {
      return false
    }
  }

  const enableMFA = async (): Promise<string> => {
    const response = await fetch('/api/auth/enable-mfa', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${state.token}`,
        'Content-Type': 'application/json'
      }
    })

    const data = await response.json()
    
    if (!response.ok) {
      throw new Error(data.error)
    }

    return data.qrCode
  }

  const disableMFA = async (password: string): Promise<boolean> => {
    try {
      const response = await fetch('/api/auth/disable-mfa', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${state.token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ password })
      })

      if (response.ok) {
        dispatch({ type: 'UPDATE_USER', payload: { mfaEnabled: false } })
        return true
      }
      return false
    } catch (error) {
      return false
    }
  }

  const updateProfile = async (data: Partial<User>): Promise<boolean> => {
    try {
      const response = await fetch('/api/auth/profile', {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${state.token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
      })

      if (response.ok) {
        const updatedUser = await response.json()
        localStorage.setItem('cardioai_user', JSON.stringify(updatedUser))
        dispatch({ type: 'UPDATE_USER', payload: updatedUser })
        return true
      }
      return false
    } catch (error) {
      return false
    }
  }

  const changePassword = async (oldPassword: string, newPassword: string): Promise<boolean> => {
    try {
      const response = await fetch('/api/auth/change-password', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${state.token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ oldPassword, newPassword })
      })

      return response.ok
    } catch (error) {
      return false
    }
  }

  const requestPasswordReset = async (email: string): Promise<boolean> => {
    try {
      const response = await fetch('/api/auth/request-reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email })
      })

      return response.ok
    } catch (error) {
      return false
    }
  }

  const resetPassword = async (token: string, newPassword: string): Promise<boolean> => {
    try {
      const response = await fetch('/api/auth/reset-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token, newPassword })
      })

      return response.ok
    } catch (error) {
      return false
    }
  }

  const value: AuthContextType = {
    ...state,
    login,
    loginWithBiometric,
    logout,
    refreshAuth,
    verifyMFA,
    enableMFA,
    disableMFA,
    updateProfile,
    changePassword,
    requestPasswordReset,
    resetPassword
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

// Hook to use auth context
export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

// HOC for protected routes
export const withAuth = <P extends object>(
  Component: React.ComponentType<P>,
  requiredPermissions?: string[]
) => {
  return (props: P) => {
    const { isAuthenticated, user, isLoading } = useAuth()

    if (isLoading) {
      return <div>Carregando...</div>
    }

    if (!isAuthenticated) {
      return <div>Acesso negado. Faça login.</div>
    }

    if (requiredPermissions && user) {
      const hasPermission = requiredPermissions.every(permission =>
        user.permissions.includes(permission)
      )
      
      if (!hasPermission) {
        return <div>Permissão insuficiente.</div>
      }
    }

    return <Component {...props} />
  }
}

export default AuthProvider

