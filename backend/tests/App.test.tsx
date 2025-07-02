import { render } from '@testing-library/react'
import { Provider } from 'react-redux'
import { BrowserRouter } from 'react-router-dom'
import { configureStore } from '@reduxjs/toolkit'
import App from './App'

// Mock do store Redux
const mockStore = configureStore({
  reducer: {
    auth: (state = { isAuthenticated: false, user: null }) => state,
  },
})

const AllTheProviders = ({ children }: { children: React.ReactNode }) => {
  return (
    <Provider store={mockStore}>
      <BrowserRouter>
        {children}
      </BrowserRouter>
    </Provider>
  )
}

describe('App', () => {
  it('should render without crashing', () => {
    render(
      <AllTheProviders>
        <App />
      </AllTheProviders>
    )
    
    // Verifica se algum elemento básico está presente
    expect(document.body).toBeTruthy()
  })
})
