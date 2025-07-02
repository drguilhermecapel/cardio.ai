import React from 'react'
import { LucideIcon } from 'lucide-react'

interface MedicalCardProps {
  title: string
  value: string | number
  unit?: string
  icon: LucideIcon
  trend?: 'up' | 'down' | 'stable'
  trendValue?: string
  severity?: 'normal' | 'warning' | 'critical'
  className?: string
  children?: React.ReactNode
}

const MedicalCard: React.FC<MedicalCardProps> = ({
  title,
  value,
  unit,
  icon: Icon,
  trend,
  trendValue,
  severity = 'normal',
  className = '',
  children,
}) => {
  const getSeverityColors = (): {
    bg: string
    border: string
    glow: string
    icon: string
    accent: string
  } => {
    switch (severity) {
      case 'warning':
        return {
          bg: 'bg-yellow-500/10',
          border: 'border-yellow-500/20',
          glow: 'shadow-yellow-500/25',
          icon: 'text-yellow-400',
          accent: 'from-yellow-500 to-orange-500',
        }
      case 'critical':
        return {
          bg: 'bg-red-500/10',
          border: 'border-red-500/20',
          glow: 'shadow-red-500/25',
          icon: 'text-red-400',
          accent: 'from-red-500 to-pink-500',
        }
      default:
        return {
          bg: 'bg-cyan-500/10',
          border: 'border-cyan-500/20',
          glow: 'shadow-cyan-500/25',
          icon: 'text-cyan-400',
          accent: 'from-cyan-500 to-blue-500',
        }
    }
  }

  const getTrendIcon = (): string => {
    if (trend === 'up') return '↗'
    if (trend === 'down') return '↘'
    return '→'
  }

  const getTrendColor = (): string => {
    if (trend === 'up') return 'text-green-400'
    if (trend === 'down') return 'text-red-400'
    return 'text-gray-400'
  }

  const colors = getSeverityColors()

  return (
    <div className={`relative group ${className}`}>
      <div
        className={`
        ${colors.bg} ${colors.border} backdrop-blur-xl rounded-2xl border 
        shadow-2xl ${colors.glow} p-6 transition-all duration-300 
        hover:scale-105 hover:shadow-2xl relative overflow-hidden
      `}
      >
        {/* Glassmorphism overlay */}
        <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent rounded-2xl"></div>

        {/* Animated background gradient */}
        <div
          className={`
          absolute inset-0 bg-gradient-to-br ${colors.accent} opacity-0 
          group-hover:opacity-10 transition-opacity duration-500 rounded-2xl
        `}
        ></div>

        <div className="relative z-10">
          {/* Header */}
          <div className="flex items-center justify-between mb-4">
            <div
              className={`
              p-3 rounded-xl bg-gradient-to-br ${colors.accent} 
              shadow-lg ${colors.glow}
            `}
            >
              <Icon className={`w-6 h-6 text-white`} />
            </div>

            {trend && trendValue && (
              <div
                className={`
                flex items-center space-x-1 px-2 py-1 rounded-lg 
                bg-gray-800/50 ${getTrendColor()}
              `}
              >
                <span className="text-sm font-medium">{getTrendIcon()}</span>
                <span className="text-sm font-semibold">{trendValue}</span>
              </div>
            )}
          </div>

          {/* Content */}
          <div className="space-y-2">
            <h3 className="text-sm font-medium text-gray-300 uppercase tracking-wide">{title}</h3>
            <div className="flex items-baseline space-x-2">
              <span className="text-3xl font-bold text-white">{value}</span>
              {unit && <span className="text-lg text-gray-400 font-medium">{unit}</span>}
            </div>
          </div>

          {/* Additional content */}
          {children && <div className="mt-4 pt-4 border-t border-gray-700/50">{children}</div>}
        </div>

        {/* Pulse animation for critical severity */}
        {severity === 'critical' && (
          <div className="absolute inset-0 rounded-2xl bg-red-500/20 animate-pulse"></div>
        )}
      </div>
    </div>
  )
}

export default MedicalCard
