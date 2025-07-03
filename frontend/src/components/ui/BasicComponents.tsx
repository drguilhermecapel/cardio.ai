// Modern UI Components for CardioAI Pro
// Fusão entre cardiologia e inteligência artificial

import React from 'react'

// Basic Box component with medical styling
export const Box: React.FC<{
  children: React.ReactNode
  className?: string
}> = ({ children, className = '' }) => {
  return (
    <div className={`${className}`}>
      {children}
    </div>
  )
}

// Modern Card component
export const Card: React.FC<{
  children: React.ReactNode
  className?: string
  variant?: 'default' | 'medical' | 'ai' | 'critical'
}> = ({ children, className = '', variant = 'default' }) => {
  const variantClasses = {
    default: 'card-medical',
    medical: 'card-medical shadow-medical',
    ai: 'card-medical shadow-ai bg-gradient-to-br from-purple-50 to-blue-50',
    critical: 'card-medical shadow-critical bg-gradient-to-br from-red-50 to-orange-50'
  }
  
  return (
    <div className={`${variantClasses[variant]} ${className}`}>
      {children}
    </div>
  )
}

// Modern CardContent component
export const CardContent: React.FC<{
  children: React.ReactNode
  className?: string
}> = ({ children, className = '' }) => {
  return (
    <div className={`p-6 ${className}`}>
      {children}
    </div>
  )
}

// Modern Typography component
export const Typography: React.FC<{
  children: React.ReactNode
  variant?: 'h1' | 'h2' | 'h3' | 'h4' | 'h5' | 'h6' | 'body1' | 'body2' | 'caption'
  className?: string
  color?: 'primary' | 'secondary' | 'critical' | 'warning' | 'ai' | 'default'
}> = ({ children, variant = 'body1', className = '', color = 'default' }) => {
  const variantClasses = {
    h1: 'text-4xl font-bold text-medical-title',
    h2: 'text-3xl font-bold text-medical-title',
    h3: 'text-2xl font-semibold text-medical-subtitle',
    h4: 'text-xl font-semibold text-medical-subtitle',
    h5: 'text-lg font-medium text-medical-subtitle',
    h6: 'text-base font-medium text-medical-subtitle',
    body1: 'text-medical-body',
    body2: 'text-sm text-medical-body',
    caption: 'text-medical-caption'
  }
  
  const colorClasses = {
    primary: 'text-blue-600',
    secondary: 'text-green-600',
    critical: 'text-red-600',
    warning: 'text-yellow-600',
    ai: 'text-purple-600',
    default: ''
  }
  
  const Component = variant.startsWith('h') ? variant as keyof JSX.IntrinsicElements : 'p'
  
  return (
    <Component className={`${variantClasses[variant]} ${colorClasses[color]} ${className}`}>
      {children}
    </Component>
  )
}

// Modern Button component
export const Button: React.FC<{
  children: React.ReactNode
  variant?: 'contained' | 'outlined' | 'text'
  color?: 'primary' | 'secondary' | 'critical' | 'warning' | 'ai'
  size?: 'small' | 'medium' | 'large'
  disabled?: boolean
  onClick?: () => void
  className?: string
  type?: 'button' | 'submit' | 'reset'
}> = ({ 
  children, 
  variant = 'contained', 
  color = 'primary', 
  size = 'medium',
  disabled = false,
  onClick,
  className = '',
  type = 'button'
}) => {
  const baseClasses = 'btn-medical inline-flex items-center justify-center'
  
  const variantClasses = {
    contained: {
      primary: 'btn-primary',
      secondary: 'btn-secondary',
      critical: 'btn-critical',
      warning: 'bg-yellow-500 text-white hover:bg-yellow-600 focus:ring-yellow-500',
      ai: 'btn-ai'
    },
    outlined: {
      primary: 'border-2 border-blue-500 text-blue-500 hover:bg-blue-50 focus:ring-blue-500',
      secondary: 'border-2 border-green-500 text-green-500 hover:bg-green-50 focus:ring-green-500',
      critical: 'border-2 border-red-500 text-red-500 hover:bg-red-50 focus:ring-red-500',
      warning: 'border-2 border-yellow-500 text-yellow-500 hover:bg-yellow-50 focus:ring-yellow-500',
      ai: 'border-2 border-purple-500 text-purple-500 hover:bg-purple-50 focus:ring-purple-500'
    },
    text: {
      primary: 'text-blue-500 hover:bg-blue-50 focus:ring-blue-500',
      secondary: 'text-green-500 hover:bg-green-50 focus:ring-green-500',
      critical: 'text-red-500 hover:bg-red-50 focus:ring-red-500',
      warning: 'text-yellow-500 hover:bg-yellow-50 focus:ring-yellow-500',
      ai: 'text-purple-500 hover:bg-purple-50 focus:ring-purple-500'
    }
  }
  
  const sizeClasses = {
    small: 'px-3 py-1.5 text-sm',
    medium: 'px-4 py-2',
    large: 'px-6 py-3 text-lg'
  }
  
  const disabledClasses = disabled ? 'opacity-50 cursor-not-allowed' : ''
  
  return (
    <button
      type={type}
      onClick={disabled ? undefined : onClick}
      disabled={disabled}
      className={`
        ${baseClasses}
        ${variantClasses[variant][color]}
        ${sizeClasses[size]}
        ${disabledClasses}
        ${className}
      `}
    >
      {children}
    </button>
  )
}

// Modern TextField component
export const TextField: React.FC<{
  label?: string
  value?: string
  onChange?: (event: React.ChangeEvent<HTMLInputElement>) => void
  placeholder?: string
  type?: string
  fullWidth?: boolean
  variant?: 'outlined' | 'filled' | 'standard'
  className?: string
  disabled?: boolean
  error?: boolean
  helperText?: string
}> = ({ 
  label, 
  value, 
  onChange, 
  placeholder, 
  type = 'text', 
  fullWidth = false, 
  variant = 'outlined', 
  className = '', 
  disabled = false,
  error = false,
  helperText
}) => {
  const inputClasses = `
    input-medical
    ${fullWidth ? 'w-full' : ''}
    ${error ? 'border-red-500 focus:ring-red-500 focus:border-red-500' : ''}
    ${disabled ? 'opacity-50 cursor-not-allowed bg-gray-100' : ''}
  `
  
  return (
    <div className={`${fullWidth ? 'w-full' : ''} ${className}`}>
      {label && (
        <label className="form-label">
          {label}
        </label>
      )}
      <input
        type={type}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        disabled={disabled}
        className={inputClasses}
      />
      {helperText && (
        <p className={`text-sm mt-1 ${error ? 'text-red-600' : 'text-gray-500'}`}>
          {helperText}
        </p>
      )}
    </div>
  )
}

// Modern Grid component
export const Grid: React.FC<{
  children: React.ReactNode
  container?: boolean
  item?: boolean
  xs?: number
  sm?: number
  md?: number
  lg?: number
  xl?: number
  spacing?: number
  className?: string
}> = ({ 
  children, 
  container = false, 
  item = false, 
  xs, 
  sm, 
  md, 
  lg, 
  xl, 
  spacing = 0,
  className = '' 
}) => {
  let classes = ''
  
  if (container) {
    classes += 'grid '
    if (spacing > 0) {
      classes += `gap-${spacing} `
    }
  }
  
  if (item) {
    if (xs) classes += `col-span-${xs} `
    if (sm) classes += `sm:col-span-${sm} `
    if (md) classes += `md:col-span-${md} `
    if (lg) classes += `lg:col-span-${lg} `
    if (xl) classes += `xl:col-span-${xl} `
  }
  
  return (
    <div className={`${classes} ${className}`}>
      {children}
    </div>
  )
}

// Modern Alert component
export const Alert: React.FC<{
  children: React.ReactNode
  severity?: 'success' | 'error' | 'warning' | 'info'
  className?: string
}> = ({ children, severity = 'info', className = '' }) => {
  const severityClasses = {
    success: 'success-message',
    error: 'error-message',
    warning: 'warning-message',
    info: 'info-message'
  }
  
  return (
    <div className={`${severityClasses[severity]} ${className}`}>
      {children}
    </div>
  )
}

// Modern CircularProgress component
export const CircularProgress: React.FC<{
  size?: number
  className?: string
  color?: 'primary' | 'secondary' | 'critical' | 'ai'
}> = ({ size = 40, className = '', color = 'primary' }) => {
  const colorClasses = {
    primary: 'border-blue-500',
    secondary: 'border-green-500',
    critical: 'border-red-500',
    ai: 'border-purple-500'
  }
  
  return (
    <div 
      className={`loading-medical-spinner ${colorClasses[color]} ${className}`}
      style={{ width: size, height: size }}
    >
      <span className="sr-only">Loading...</span>
    </div>
  )
}

// Medical Badge component
export const Badge: React.FC<{
  children: React.ReactNode
  variant?: 'success' | 'warning' | 'critical' | 'info' | 'ai'
  className?: string
}> = ({ children, variant = 'info', className = '' }) => {
  const variantClasses = {
    success: 'badge-success',
    warning: 'badge-warning',
    critical: 'badge-critical',
    info: 'badge-info',
    ai: 'badge-ai'
  }
  
  return (
    <span className={`${variantClasses[variant]} ${className}`}>
      {children}
    </span>
  )
}

// Status Indicator component
export const StatusIndicator: React.FC<{
  status: 'online' | 'offline' | 'critical' | 'warning'
  label?: string
  className?: string
}> = ({ status, label, className = '' }) => {
  const statusClasses = {
    online: 'status-online',
    offline: 'status-offline',
    critical: 'status-critical',
    warning: 'status-warning'
  }
  
  return (
    <div className={`flex items-center space-x-2 ${className}`}>
      <div className={statusClasses[status]}></div>
      {label && <span className="text-sm text-gray-600">{label}</span>}
    </div>
  )
}

// Heartbeat Indicator component
export const HeartbeatIndicator: React.FC<{
  active?: boolean
  className?: string
}> = ({ active = true, className = '' }) => {
  return (
    <div className={`heartbeat-indicator ${active ? 'animate-heartbeat' : ''} ${className}`}>
    </div>
  )
}

// AI Glow component
export const AIGlow: React.FC<{
  children: React.ReactNode
  active?: boolean
  className?: string
}> = ({ children, active = true, className = '' }) => {
  return (
    <div className={`${active ? 'ai-glow animate-glow-pulse' : ''} rounded-lg p-4 ${className}`}>
      {children}
    </div>
  )
}

