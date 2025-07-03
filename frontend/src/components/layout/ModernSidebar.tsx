// Modern Sidebar Component with Animated Icons
// CardioAI Pro - Fusão entre cardiologia e inteligência artificial

import React, { useState } from 'react'
import { Typography, Button, Badge, StatusIndicator } from '../ui/BasicComponents'

interface SidebarProps {
  isCollapsed?: boolean
  onToggle?: () => void
  currentPath?: string
}

interface NavItem {
  id: string
  label: string
  icon: string
  path: string
  badge?: number
  color?: 'primary' | 'secondary' | 'ai' | 'critical'
}

const navigationItems: NavItem[] = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    icon: '📊',
    path: '/dashboard',
    color: 'primary'
  },
  {
    id: 'patients',
    label: 'Pacientes',
    icon: '👥',
    path: '/patients',
    color: 'secondary'
  },
  {
    id: 'ecg-analysis',
    label: 'Análise ECG',
    icon: '💓',
    path: '/ecg-analysis',
    badge: 5,
    color: 'critical'
  },
  {
    id: 'ai-insights',
    label: 'IA Insights',
    icon: '🧠',
    path: '/ai-insights',
    badge: 2,
    color: 'ai'
  },
  {
    id: 'reports',
    label: 'Relatórios',
    icon: '📋',
    path: '/reports',
    color: 'primary'
  },
  {
    id: 'validations',
    label: 'Validações',
    icon: '✅',
    path: '/validations',
    color: 'secondary'
  },
  {
    id: 'notifications',
    label: 'Notificações',
    icon: '🔔',
    path: '/notifications',
    badge: 3,
    color: 'primary'
  },
  {
    id: 'profile',
    label: 'Perfil',
    icon: '👤',
    path: '/profile',
    color: 'primary'
  }
]

export const ModernSidebar: React.FC<SidebarProps> = ({
  isCollapsed = false,
  onToggle,
  currentPath = '/dashboard'
}) => {
  const [hoveredItem, setHoveredItem] = useState<string | null>(null)

  return (
    <aside className={`relative transition-all duration-300 ${isCollapsed ? 'w-16' : 'w-64'}`}>
      {/* Glassmorphism Background */}
      <div className="absolute inset-0 bg-white/80 backdrop-blur-lg border-r border-white/20 shadow-lg"></div>
      
      {/* Sidebar Content */}
      <div className="relative h-full flex flex-col">
        
        {/* Header */}
        <div className="p-4 border-b border-white/20">
          <div className="flex items-center justify-between">
            {!isCollapsed && (
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-gradient-ai rounded-lg flex items-center justify-center shadow-ai">
                  <span className="text-white text-sm">💜</span>
                </div>
                <div>
                  <Typography variant="h6" className="font-bold text-gray-900">
                    CardioAI
                  </Typography>
                  <Typography variant="caption" className="text-gray-500">
                    v2.0 Pro
                  </Typography>
                </div>
              </div>
            )}
            
            {/* Toggle Button */}
            <Button
              variant="text"
              color="primary"
              className="p-2 rounded-lg hover:bg-white/50"
              onClick={onToggle}
            >
              <svg 
                className={`w-5 h-5 transition-transform duration-200 ${isCollapsed ? 'rotate-180' : ''}`} 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
              </svg>
            </Button>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-2">
          {navigationItems.map((item) => {
            const isActive = currentPath === item.path
            const isHovered = hoveredItem === item.id
            
            return (
              <div
                key={item.id}
                className="relative"
                onMouseEnter={() => setHoveredItem(item.id)}
                onMouseLeave={() => setHoveredItem(null)}
              >
                <Button
                  variant="text"
                  color={item.color}
                  className={`
                    w-full justify-start p-3 rounded-xl transition-all duration-200
                    ${isActive 
                      ? 'bg-gradient-primary text-white shadow-md' 
                      : 'hover:bg-white/50 text-gray-700'
                    }
                    ${isHovered ? 'transform scale-105' : ''}
                  `}
                >
                  {/* Icon */}
                  <div className={`
                    flex items-center justify-center w-8 h-8 rounded-lg
                    ${isActive ? 'bg-white/20' : 'bg-transparent'}
                    ${isHovered ? 'animate-pulse' : ''}
                  `}>
                    <span className="text-lg">{item.icon}</span>
                  </div>
                  
                  {/* Label and Badge */}
                  {!isCollapsed && (
                    <div className="flex items-center justify-between flex-1 ml-3">
                      <Typography 
                        variant="body2" 
                        className={`font-medium ${isActive ? 'text-white' : 'text-gray-700'}`}
                      >
                        {item.label}
                      </Typography>
                      
                      {item.badge && (
                        <Badge 
                          variant={item.color === 'critical' ? 'critical' : 'info'} 
                          className="ml-2"
                        >
                          {item.badge}
                        </Badge>
                      )}
                    </div>
                  )}
                </Button>

                {/* Tooltip for Collapsed State */}
                {isCollapsed && isHovered && (
                  <div className="absolute left-full top-0 ml-2 px-3 py-2 bg-gray-900 text-white text-sm rounded-lg shadow-lg z-50 animate-fade-in">
                    {item.label}
                    {item.badge && (
                      <Badge variant="critical" className="ml-2">
                        {item.badge}
                      </Badge>
                    )}
                    {/* Arrow */}
                    <div className="absolute left-0 top-1/2 transform -translate-y-1/2 -translate-x-1 w-2 h-2 bg-gray-900 rotate-45"></div>
                  </div>
                )}
              </div>
            )
          })}
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-white/20">
          {/* System Status */}
          <div className={`
            flex items-center space-x-3 p-3 rounded-xl bg-white/50 backdrop-blur-sm
            ${isCollapsed ? 'justify-center' : ''}
          `}>
            <StatusIndicator status="online" />
            {!isCollapsed && (
              <div>
                <Typography variant="body2" className="font-medium text-gray-900">
                  Sistema Online
                </Typography>
                <Typography variant="caption" className="text-gray-500">
                  Todos os serviços ativos
                </Typography>
              </div>
            )}
          </div>

          {/* AI Status */}
          {!isCollapsed && (
            <div className="mt-3 p-3 rounded-xl bg-gradient-to-r from-purple-50 to-blue-50 border border-purple-200">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-purple-500 rounded-full animate-pulse"></div>
                <Typography variant="body2" className="font-medium text-purple-700">
                  IA Ativa
                </Typography>
              </div>
              <Typography variant="caption" className="text-purple-600">
                Análise em tempo real
              </Typography>
            </div>
          )}
        </div>
      </div>
    </aside>
  )
}

export default ModernSidebar

