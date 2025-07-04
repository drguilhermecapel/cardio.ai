import { createContext } from 'react'

export interface User {
  id: string
  email: string
  name: string
  role: 'admin' | 'doctor' | 'nurse' | 'technician'
  permissions: string[]
  mfaEnabled: boolean
  lastLogin: string
  profileImage?: string
  specialties?: string[]
  license?: string
}

export interface LoginResult {
  success: boolean
  requiresMFA?: boolean
  error?: string
  user?: User
}

export interface AuthContextType {
  user: User | null
  token: string | null
  refreshToken: string | null
  isAuthenticated: boolean
  isLoading: boolean
  mfaRequired: boolean
  biometricAvailable: boolean
  sessionExpiry: number | null
  login: (email: string, password: string) => Promise<LoginResult>
  loginWithBiometric: () => Promise<LoginResult>
  logout: () => void
  refreshAuth: () => Promise<boolean>
  verifyMFA: (code: string) => Promise<boolean>
  enableMFA: () => Promise<string> // Returns QR code
  disableMFA: (password: string) => Promise<boolean>
  updateProfile: (data: Partial<User>) => Promise<boolean>
  changePassword: (oldPassword: string, newPassword: string) => Promise<boolean>
  requestPasswordReset: (email: string) => Promise<boolean>
  resetPassword: (token: string, newPassword: string) => Promise<boolean>
}

export const AuthContext = createContext<AuthContextType | undefined>(undefined)
