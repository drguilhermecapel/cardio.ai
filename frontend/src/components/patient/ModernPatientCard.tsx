// Modern Patient Card Component with Glassmorphism
// CardioAI Pro - Fusão entre cardiologia e inteligência artificial

import React, { useState } from 'react'
import { Typography, Button, Badge, HeartbeatIndicator, Card, CardContent } from '../ui/BasicComponents'

interface PatientData {
  id: string
  name: string
  age: number
  gender: 'M' | 'F'
  condition: string
  status: 'stable' | 'critical' | 'monitoring' | 'discharged'
  lastECG: string
  heartRate: number
  bloodPressure: {
    systolic: number
    diastolic: number
  }
  riskLevel: 'low' | 'medium' | 'high' | 'critical'
  aiInsights?: string[]
  avatar?: string
}

interface PatientCardProps {
  patient: PatientData
  onViewDetails?: (patientId: string) => void
  onViewECG?: (patientId: string) => void
  className?: string
}

export const ModernPatientCard: React.FC<PatientCardProps> = ({
  patient,
  onViewDetails,
  onViewECG,
  className = ''
}) => {
  const [isHovered, setIsHovered] = useState(false)

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'stable': return 'success'
      case 'critical': return 'critical'
      case 'monitoring': return 'warning'
      case 'discharged': return 'info'
      default: return 'info'
    }
  }

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return 'success'
      case 'medium': return 'warning'
      case 'high': return 'warning'
      case 'critical': return 'critical'
      default: return 'info'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'stable': return '✅'
      case 'critical': return '🚨'
      case 'monitoring': return '👁️'
      case 'discharged': return '🏠'
      default: return '📋'
    }
  }

  return (
    <Card
      variant={patient.status === 'critical' ? 'critical' : 'medical'}
      className={`
        patient-card transition-all duration-300 cursor-pointer
        ${isHovered ? 'transform -translate-y-2 shadow-xl' : ''}
        ${patient.status === 'critical' ? 'ring-2 ring-red-200 animate-pulse' : ''}
        ${className}
      `}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <CardContent className="p-6">
        
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center space-x-3">
            {/* Avatar */}
            <div className="relative">
              <div className="w-12 h-12 bg-gradient-primary rounded-full flex items-center justify-center text-white font-bold text-lg shadow-md">
                {patient.name.split(' ').map(n => n[0]).join('')}
              </div>
              {patient.status === 'critical' && (
                <div className="absolute -top-1 -right-1">
                  <HeartbeatIndicator className="w-4 h-4" />
                </div>
              )}
            </div>
            
            {/* Patient Info */}
            <div>
              <Typography variant="h6" className="font-bold text-gray-900">
                {patient.name}
              </Typography>
              <Typography variant="body2" className="text-gray-600">
                {patient.age} anos • {patient.gender === 'M' ? 'Masculino' : 'Feminino'}
              </Typography>
            </div>
          </div>

          {/* Status Badge */}
          <div className="flex items-center space-x-2">
            <Badge variant={getStatusColor(patient.status)} className="flex items-center space-x-1">
              <span>{getStatusIcon(patient.status)}</span>
              <span className="capitalize">{patient.status}</span>
            </Badge>
          </div>
        </div>

        {/* Condition */}
        <div className="mb-4">
          <Typography variant="body2" className="text-gray-700 font-medium">
            {patient.condition}
          </Typography>
        </div>

        {/* Vital Signs */}
        <div className="grid grid-cols-2 gap-4 mb-4">
          {/* Heart Rate */}
          <div className="bg-gradient-to-br from-red-50 to-pink-50 rounded-lg p-3 border border-red-100">
            <div className="flex items-center space-x-2 mb-1">
              <HeartbeatIndicator className="w-4 h-4" />
              <Typography variant="caption" className="text-red-600 font-medium">
                Freq. Cardíaca
              </Typography>
            </div>
            <Typography variant="h6" className="font-bold text-red-700">
              {patient.heartRate} bpm
            </Typography>
          </div>

          {/* Blood Pressure */}
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg p-3 border border-blue-100">
            <div className="flex items-center space-x-2 mb-1">
              <span className="text-blue-500">🩺</span>
              <Typography variant="caption" className="text-blue-600 font-medium">
                Pressão Arterial
              </Typography>
            </div>
            <Typography variant="h6" className="font-bold text-blue-700">
              {patient.bloodPressure.systolic}/{patient.bloodPressure.diastolic}
            </Typography>
          </div>
        </div>

        {/* Risk Level */}
        <div className="mb-4">
          <div className="flex items-center justify-between">
            <Typography variant="body2" className="text-gray-600">
              Nível de Risco:
            </Typography>
            <Badge variant={getRiskColor(patient.riskLevel)} className="capitalize">
              {patient.riskLevel === 'low' ? 'Baixo' : 
               patient.riskLevel === 'medium' ? 'Médio' : 
               patient.riskLevel === 'high' ? 'Alto' : 'Crítico'}
            </Badge>
          </div>
        </div>

        {/* AI Insights */}
        {patient.aiInsights && patient.aiInsights.length > 0 && (
          <div className="mb-4 p-3 bg-gradient-to-br from-purple-50 to-blue-50 rounded-lg border border-purple-200">
            <div className="flex items-center space-x-2 mb-2">
              <span className="text-purple-500">🧠</span>
              <Typography variant="caption" className="text-purple-600 font-medium">
                IA Insights
              </Typography>
            </div>
            <Typography variant="body2" className="text-purple-700">
              {patient.aiInsights[0]}
            </Typography>
            {patient.aiInsights.length > 1 && (
              <Typography variant="caption" className="text-purple-500">
                +{patient.aiInsights.length - 1} insights adicionais
              </Typography>
            )}
          </div>
        )}

        {/* Last ECG */}
        <div className="mb-4">
          <Typography variant="caption" className="text-gray-500">
            Último ECG: {patient.lastECG}
          </Typography>
        </div>

        {/* Actions */}
        <div className="flex space-x-2">
          <Button
            variant="contained"
            color="primary"
            size="small"
            className="flex-1"
            onClick={() => onViewDetails?.(patient.id)}
          >
            Ver Detalhes
          </Button>
          <Button
            variant="outlined"
            color="secondary"
            size="small"
            className="flex-1"
            onClick={() => onViewECG?.(patient.id)}
          >
            📈 ECG
          </Button>
        </div>

        {/* Hover Effect Overlay */}
        {isHovered && (
          <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-purple-500/5 rounded-xl pointer-events-none animate-fade-in"></div>
        )}
      </CardContent>
    </Card>
  )
}

// Patient Grid Component
interface PatientGridProps {
  patients: PatientData[]
  onViewDetails?: (patientId: string) => void
  onViewECG?: (patientId: string) => void
  className?: string
}

export const ModernPatientGrid: React.FC<PatientGridProps> = ({
  patients,
  onViewDetails,
  onViewECG,
  className = ''
}) => {
  return (
    <div className={`grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 ${className}`}>
      {patients.map((patient) => (
        <ModernPatientCard
          key={patient.id}
          patient={patient}
          onViewDetails={onViewDetails}
          onViewECG={onViewECG}
        />
      ))}
    </div>
  )
}

export default ModernPatientCard

