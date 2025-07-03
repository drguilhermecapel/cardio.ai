import authReducer, { login, logout } from '../slices/authSlice'

describe('authSlice', () => {
  const initialState = {
    user: null,
    token: null,
    refreshToken: null,
    isAuthenticated: false,
    isLoading: false,
    error: null,
  }

  it('should return the initial state', () => {
    expect(authReducer(undefined, { type: 'unknown' })).toEqual(
      expect.objectContaining({
        user: null,
        isAuthenticated: false,
        isLoading: false,
        error: null,
      })
    )
  })

  it('should handle login pending', () => {
    const credentials = { username: 'testuser', password: 'testpass' }

    const actual = authReducer(initialState, login.pending('', credentials))

    expect(actual.isLoading).toBe(true)
    expect(actual.error).toBe(null)
  })

  it('should handle logout fulfilled', () => {
    const loggedInState = {
      ...initialState,
      isAuthenticated: true,
      user: {
        id: 1,
        username: 'testuser',
        email: 'test@test.com',
        firstName: 'Test',
        lastName: 'User',
        role: 'physician',
        isActive: true,
      },
      token: 'test-token',
    }

    const actual = authReducer(loggedInState, logout.fulfilled(undefined, '', undefined))

    expect(actual.isAuthenticated).toBe(false)
    expect(actual.user).toBe(null)
    expect(actual.token).toBe(null)
  })
})
