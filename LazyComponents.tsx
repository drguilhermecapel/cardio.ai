// Lazy Loading Components for CardioAI Pro
// Componentes com carregamento sob demanda para otimização

import React, { Suspense, lazy } from 'react'
import { CircularProgress } from './components/ui/BasicComponents'

// Loading fallback component
const LoadingFallback: React.FC<{ message?: string }> = ({ message = 'Carregando...' }) => (
  <div className="flex items-center justify-center min-h-[200px] bg-gray-50 rounded-lg">
    <div className="text-center">
      <CircularProgress size="large" />
      <p className="mt-4 text-gray-600">{message}</p>
    </div>
  </div>
)

// Lazy loaded pages
export const DashboardPage = lazy(() => 
  import('./pages/DashboardPage').then(module => ({ default: module.default }))
)

export const PatientsPage = lazy(() => 
  import('./pages/PatientsPage').then(module => ({ default: module.default }))
)

export const ECGAnalysisPage = lazy(() => 
  import('./pages/ECGAnalysisPage').then(module => ({ default: module.default }))
)

export const ValidationsPage = lazy(() => 
  import('./pages/ValidationsPage').then(module => ({ default: module.default }))
)

export const NotificationsPage = lazy(() => 
  import('./pages/NotificationsPage').then(module => ({ default: module.default }))
)

export const ProfilePage = lazy(() => 
  import('./pages/ProfilePage').then(module => ({ default: module.default }))
)

export const AdminDashboard = lazy(() => 
  import('./pages/AdminDashboard').then(module => ({ default: module.default }))
)

// Lazy loaded components
export const ModernECGVisualization = lazy(() => 
  import('./components/medical/ModernECGVisualization').then(module => ({ default: module.default }))
)

export const AdvancedECGUpload = lazy(() => 
  import('./components/upload/AdvancedECGUpload').then(module => ({ default: module.default }))
)

export const VisualAIAnalysis = lazy(() => 
  import('./components/analysis/VisualAIAnalysis').then(module => ({ default: module.default }))
)

export const ModernAnalysisModal = lazy(() => 
  import('./components/analysis/ModernAnalysisModal').then(module => ({ default: module.default }))
)

export const MedicalAnimations = lazy(() => 
  import('./components/animations/MedicalAnimations').then(module => ({ default: module.default }))
)

// HOC for lazy loading with custom loading message
export const withLazyLoading = <P extends object>(
  Component: React.LazyExoticComponent<React.ComponentType<P>>,
  loadingMessage?: string
) => {
  return (props: P) => (
    <Suspense fallback={<LoadingFallback message={loadingMessage} />}>
      <Component {...props} />
    </Suspense>
  )
}

// Preload function for critical components
export const preloadCriticalComponents = () => {
  // Preload dashboard and ECG components as they're most commonly used
  import('./pages/DashboardPage')
  import('./components/medical/ModernECGVisualization')
  import('./components/upload/AdvancedECGUpload')
}

// Route-based code splitting wrapper
export const LazyRoute: React.FC<{
  component: React.LazyExoticComponent<React.ComponentType<any>>
  loadingMessage?: string
  children?: React.ReactNode
}> = ({ component: Component, loadingMessage, children, ...props }) => (
  <Suspense fallback={<LoadingFallback message={loadingMessage} />}>
    <Component {...props}>
      {children}
    </Component>
  </Suspense>
)

export default {
  DashboardPage,
  PatientsPage,
  ECGAnalysisPage,
  ValidationsPage,
  NotificationsPage,
  ProfilePage,
  AdminDashboard,
  ModernECGVisualization,
  AdvancedECGUpload,
  VisualAIAnalysis,
  ModernAnalysisModal,
  MedicalAnimations,
  withLazyLoading,
  preloadCriticalComponents,
  LazyRoute
}

