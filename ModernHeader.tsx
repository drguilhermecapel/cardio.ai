// Modern Header Component with Glassmorphism
// CardioAI Pro - Fusão entre cardiologia e inteligência artificial

import React, { useState } from 'react'
import { Typography, Button, Badge, StatusIndicator, HeartbeatIndicator } from '../ui/BasicComponents'

interface HeaderProps {
  currentUser?: {
    name: string
    role: string
    avatar?: string
  }
  notifications?: number
  systemStatus?: 'online' | 'offline' | 'critical' | 'warning'
}

export const ModernHeader: React.FC<HeaderProps> = ({
  currentUser = { name: 'Dr. Silva', role: 'Cardiologista' },
  notifications = 3,
  systemStatus = 'online'
}) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false)

  return (
    <header className="relative">
      {/* Glassmorphism Background */}
      <div className="absolute inset-0 bg-white/80 backdrop-blur-lg border-b border-white/20 shadow-lg"></div>
      
      {/* Header Content */}
      <div className="relative container-medical">
        <div className="flex items-center justify-between h-16 px-6">
          
          {/* Logo and Brand */}
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-3">
              {/* Logo with AI Glow */}
              <div className="relative">
                <div className="w-10 h-10 bg-gradient-ai rounded-lg flex items-center justify-center shadow-ai animate-glow-pulse">
                  <HeartbeatIndicator className="w-6 h-6" />
                </div>
              </div>
              
              {/* Brand Text */}
              <div>
                <Typography variant="h5" className="font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  CardioAI Pro
                </Typography>
                <Typography variant="caption" className="text-gray-500">
                  Inteligência Artificial Médica
                </Typography>
              </div>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="hidden md:flex items-center space-x-1">
            <Button variant="text" color="primary" className="nav-link">
              Dashboard
            </Button>
            <Button variant="text" color="primary" className="nav-link">
              Pacientes
            </Button>
            <Button variant="text" color="primary" className="nav-link">
              ECG Analysis
            </Button>
            <Button variant="text" color="primary" className="nav-link">
              Relatórios
            </Button>
            <Button variant="text" color="ai" className="nav-link">
              IA Insights
            </Button>
          </nav>

          {/* Right Side Actions */}
          <div className="flex items-center space-x-4">
            
            {/* System Status */}
            <div className="hidden sm:flex items-center space-x-2 px-3 py-1 rounded-full bg-white/50 backdrop-blur-sm">
              <StatusIndicator status={systemStatus} />
              <Typography variant="caption" className="text-gray-600">
                Sistema {systemStatus === 'online' ? 'Online' : 'Offline'}
              </Typography>
            </div>

            {/* Notifications */}
            <div className="relative">
              <Button variant="text" color="primary" className="p-2 rounded-full hover:bg-white/50">
                <div className="relative">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-5 5v-5zM4 19h10a2 2 0 002-2V7a2 2 0 00-2-2H4a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                  {notifications > 0 && (
                    <Badge variant="critical" className="absolute -top-1 -right-1 min-w-[1.25rem] h-5 flex items-center justify-center text-xs">
                      {notifications}
                    </Badge>
                  )}
                </div>
              </Button>
            </div>

            {/* User Profile */}
            <div className="flex items-center space-x-3 px-3 py-2 rounded-full bg-white/50 backdrop-blur-sm hover:bg-white/70 transition-all duration-200 cursor-pointer">
              {/* Avatar */}
              <div className="w-8 h-8 bg-gradient-primary rounded-full flex items-center justify-center text-white font-medium text-sm">
                {currentUser.name.split(' ').map(n => n[0]).join('')}
              </div>
              
              {/* User Info */}
              <div className="hidden sm:block">
                <Typography variant="body2" className="font-medium text-gray-900">
                  {currentUser.name}
                </Typography>
                <Typography variant="caption" className="text-gray-500">
                  {currentUser.role}
                </Typography>
              </div>
              
              {/* Dropdown Arrow */}
              <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </div>

            {/* Mobile Menu Button */}
            <Button 
              variant="text" 
              color="primary" 
              className="md:hidden p-2"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </Button>
          </div>
        </div>

        {/* Mobile Menu */}
        {isMenuOpen && (
          <div className="md:hidden absolute top-full left-0 right-0 bg-white/95 backdrop-blur-lg border-b border-white/20 shadow-lg animate-slide-up">
            <nav className="px-6 py-4 space-y-2">
              <Button variant="text" color="primary" className="w-full justify-start nav-link">
                Dashboard
              </Button>
              <Button variant="text" color="primary" className="w-full justify-start nav-link">
                Pacientes
              </Button>
              <Button variant="text" color="primary" className="w-full justify-start nav-link">
                ECG Analysis
              </Button>
              <Button variant="text" color="primary" className="w-full justify-start nav-link">
                Relatórios
              </Button>
              <Button variant="text" color="ai" className="w-full justify-start nav-link">
                IA Insights
              </Button>
            </nav>
          </div>
        )}
      </div>
    </header>
  )
}

export default ModernHeader

