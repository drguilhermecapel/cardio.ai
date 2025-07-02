import React from 'react'
import { Outlet, Link, useNavigate, useLocation } from 'react-router-dom'
import { useAuth } from '../hooks/useAuth'

const Layout: React.FC = (): JSX.Element | null => {
  const { user, logout, isAuthenticated } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()

  React.useEffect(() => {
    if (!isAuthenticated) {
      navigate('/login')
    }
  }, [isAuthenticated, navigate])

  if (!isAuthenticated) {
    return null
  }

  const handleLogout = (): void => {
    logout()
    navigate('/login')
  }

  const isActive = (path: string): boolean => location.pathname === path

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-blue-600 text-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <h1 className="text-xl font-bold">SPEI - Sistema EMR</h1>
            </div>
            <div className="flex items-center space-x-4">
              <span>Bem-vindo, {user?.username}</span>
              <button
                onClick={handleLogout}
                className="bg-blue-700 hover:bg-blue-800 px-3 py-1 rounded"
              >
                Sair
              </button>
            </div>
          </div>
        </div>
      </nav>

      <div className="flex">
        <aside className="w-64 bg-white shadow-md min-h-screen">
          <nav className="mt-8">
            <div className="px-4 space-y-2">
              <Link
                to="/dashboard"
                className={`block px-4 py-2 rounded-md ${
                  isActive('/dashboard')
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                Dashboard
              </Link>
              <Link
                to="/patients"
                className={`block px-4 py-2 rounded-md ${
                  isActive('/patients')
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                Pacientes
              </Link>
              <Link
                to="/medical-records"
                className={`block px-4 py-2 rounded-md ${
                  isActive('/medical-records')
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                Prontuários
              </Link>
              <Link
                to="/ai-diagnostics"
                className={`block px-4 py-2 rounded-md ${
                  isActive('/ai-diagnostics')
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                IA Diagnóstica
              </Link>
              <Link
                to="/telemedicine"
                className={`block px-4 py-2 rounded-md ${
                  isActive('/telemedicine')
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                Telemedicina
              </Link>
            </div>
          </nav>
        </aside>

        <main className="flex-1 p-8">
          <Outlet />
        </main>
      </div>
    </div>
  )
}

export default Layout
