// Admin Dashboard for CardioAI Pro
// Dashboard de administração com métricas avançadas e controle do sistema

import React, { useState, useEffect } from 'react'
import { 
  Card, 
  Button, 
  Typography, 
  Badge,
  StatusIndicator,
  CircularProgress
} from '../components/ui/BasicComponents'
import { useAuth } from '../contexts/AuthContext'
import { useNotifications } from '../contexts/NotificationContext'

// Types
interface SystemMetrics {
  users: {
    total: number
    active: number
    newToday: number
    byRole: Record<string, number>
  }
  ecg: {
    totalAnalyses: number
    todayAnalyses: number
    avgProcessingTime: number
    successRate: number
  }
  ai: {
    modelVersion: string
    accuracy: number
    confidence: number
    processingQueue: number
  }
  system: {
    uptime: number
    cpuUsage: number
    memoryUsage: number
    diskUsage: number
    apiLatency: number
  }
  security: {
    loginAttempts: number
    failedLogins: number
    suspiciousActivity: number
    lastSecurityScan: string
  }
}

interface UserManagement {
  id: string
  name: string
  email: string
  role: string
  status: 'active' | 'inactive' | 'suspended'
  lastLogin: string
  permissions: string[]
}

interface AuditLog {
  id: string
  timestamp: string
  user: string
  action: string
  resource: string
  details: string
  ipAddress: string
  userAgent: string
}

const AdminDashboard: React.FC = () => {
  const { user } = useAuth()
  const { notifications } = useNotifications()
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null)
  const [users, setUsers] = useState<UserManagement[]>([])
  const [auditLogs, setAuditLogs] = useState<AuditLog[]>([])
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState<'overview' | 'users' | 'security' | 'logs'>('overview')

  // Check admin permissions
  if (!user?.permissions.includes('admin')) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <Card variant="medical" className="p-8 text-center">
          <Typography variant="h4" className="text-red-600 mb-4">
            Acesso Negado
          </Typography>
          <Typography variant="body1" className="text-gray-600">
            Você não tem permissões de administrador para acessar esta página.
          </Typography>
        </Card>
      </div>
    )
  }

  useEffect(() => {
    loadDashboardData()
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(loadDashboardData, 30000)
    return () => clearInterval(interval)
  }, [])

  const loadDashboardData = async () => {
    try {
      setLoading(true)
      
      // Load system metrics
      const metricsResponse = await fetch('/api/admin/metrics', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('cardioai_token')}`
        }
      })
      
      if (metricsResponse.ok) {
        const metricsData = await metricsResponse.json()
        setMetrics(metricsData)
      }

      // Load users
      const usersResponse = await fetch('/api/admin/users', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('cardioai_token')}`
        }
      })
      
      if (usersResponse.ok) {
        const usersData = await usersResponse.json()
        setUsers(usersData)
      }

      // Load audit logs
      const logsResponse = await fetch('/api/admin/audit-logs?limit=50', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('cardioai_token')}`
        }
      })
      
      if (logsResponse.ok) {
        const logsData = await logsResponse.json()
        setAuditLogs(logsData)
      }
      
    } catch (error) {
      console.error('Error loading dashboard data:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleUserAction = async (userId: string, action: 'activate' | 'suspend' | 'delete') => {
    try {
      const response = await fetch(`/api/admin/users/${userId}/${action}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('cardioai_token')}`
        }
      })

      if (response.ok) {
        loadDashboardData() // Refresh data
      }
    } catch (error) {
      console.error(`Error ${action} user:`, error)
    }
  }

  const exportAuditLogs = async () => {
    try {
      const response = await fetch('/api/admin/audit-logs/export', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('cardioai_token')}`
        }
      })

      if (response.ok) {
        const blob = await response.blob()
        const url = URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.href = url
        link.download = `audit-logs-${new Date().toISOString().split('T')[0]}.csv`
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
        URL.revokeObjectURL(url)
      }
    } catch (error) {
      console.error('Error exporting audit logs:', error)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <CircularProgress size="large" />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      {/* Header */}
      <div className="mb-8">
        <Typography variant="h2" className="font-bold text-gray-900 mb-2">
          Dashboard de Administração
        </Typography>
        <Typography variant="body1" className="text-gray-600">
          Monitoramento e controle do sistema CardioAI Pro
        </Typography>
      </div>

      {/* Navigation Tabs */}
      <div className="mb-6">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            {[
              { id: 'overview', label: 'Visão Geral' },
              { id: 'users', label: 'Usuários' },
              { id: 'security', label: 'Segurança' },
              { id: 'logs', label: 'Logs de Auditoria' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && metrics && (
        <div className="space-y-6">
          {/* System Status Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {/* Users */}
            <Card variant="medical">
              <div className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <Typography variant="caption" className="text-gray-600">
                      Usuários Ativos
                    </Typography>
                    <Typography variant="h3" className="font-bold text-blue-600">
                      {metrics.users.active}
                    </Typography>
                    <Typography variant="caption" className="text-green-600">
                      +{metrics.users.newToday} hoje
                    </Typography>
                  </div>
                  <StatusIndicator status="online" />
                </div>
              </div>
            </Card>

            {/* ECG Analyses */}
            <Card variant="medical">
              <div className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <Typography variant="caption" className="text-gray-600">
                      Análises ECG Hoje
                    </Typography>
                    <Typography variant="h3" className="font-bold text-green-600">
                      {metrics.ecg.todayAnalyses}
                    </Typography>
                    <Typography variant="caption" className="text-gray-600">
                      {metrics.ecg.successRate}% sucesso
                    </Typography>
                  </div>
                  <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
                    <span className="text-green-600 text-xl">📊</span>
                  </div>
                </div>
              </div>
            </Card>

            {/* AI Performance */}
            <Card variant="ai">
              <div className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <Typography variant="caption" className="text-gray-600">
                      Precisão da IA
                    </Typography>
                    <Typography variant="h3" className="font-bold text-purple-600">
                      {(metrics.ai.accuracy * 100).toFixed(1)}%
                    </Typography>
                    <Typography variant="caption" className="text-gray-600">
                      v{metrics.ai.modelVersion}
                    </Typography>
                  </div>
                  <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
                    <span className="text-purple-600 text-xl">🧠</span>
                  </div>
                </div>
              </div>
            </Card>

            {/* System Health */}
            <Card variant="medical">
              <div className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <Typography variant="caption" className="text-gray-600">
                      Saúde do Sistema
                    </Typography>
                    <Typography variant="h3" className="font-bold text-green-600">
                      {metrics.system.uptime}h
                    </Typography>
                    <Typography variant="caption" className="text-gray-600">
                      CPU: {metrics.system.cpuUsage}%
                    </Typography>
                  </div>
                  <StatusIndicator status="online" />
                </div>
              </div>
            </Card>
          </div>

          {/* Detailed Metrics */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* System Resources */}
            <Card variant="medical">
              <div className="p-6">
                <Typography variant="h5" className="font-bold mb-4">
                  Recursos do Sistema
                </Typography>
                
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm text-gray-600">CPU</span>
                      <span className="text-sm text-gray-900">{metrics.system.cpuUsage}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-blue-600 h-2 rounded-full" 
                        style={{ width: `${metrics.system.cpuUsage}%` }}
                      ></div>
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm text-gray-600">Memória</span>
                      <span className="text-sm text-gray-900">{metrics.system.memoryUsage}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-green-600 h-2 rounded-full" 
                        style={{ width: `${metrics.system.memoryUsage}%` }}
                      ></div>
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm text-gray-600">Disco</span>
                      <span className="text-sm text-gray-900">{metrics.system.diskUsage}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-yellow-600 h-2 rounded-full" 
                        style={{ width: `${metrics.system.diskUsage}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>
            </Card>

            {/* Security Overview */}
            <Card variant="medical">
              <div className="p-6">
                <Typography variant="h5" className="font-bold mb-4">
                  Segurança
                </Typography>
                
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Tentativas de Login</span>
                    <Badge variant="info">{metrics.security.loginAttempts}</Badge>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-gray-600">Logins Falhados</span>
                    <Badge variant={metrics.security.failedLogins > 10 ? "warning" : "success"}>
                      {metrics.security.failedLogins}
                    </Badge>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-gray-600">Atividade Suspeita</span>
                    <Badge variant={metrics.security.suspiciousActivity > 0 ? "critical" : "success"}>
                      {metrics.security.suspiciousActivity}
                    </Badge>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-gray-600">Último Scan</span>
                    <span className="text-sm text-gray-900">
                      {new Date(metrics.security.lastSecurityScan).toLocaleDateString()}
                    </span>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        </div>
      )}

      {/* Users Tab */}
      {activeTab === 'users' && (
        <div className="space-y-6">
          <div className="flex justify-between items-center">
            <Typography variant="h4" className="font-bold">
              Gerenciamento de Usuários
            </Typography>
            <Button variant="contained" color="primary">
              Adicionar Usuário
            </Button>
          </div>

          <Card variant="medical">
            <div className="p-6">
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Usuário
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Função
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Status
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Último Login
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Ações
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {users.map((user) => (
                      <tr key={user.id}>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div>
                            <div className="text-sm font-medium text-gray-900">{user.name}</div>
                            <div className="text-sm text-gray-500">{user.email}</div>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <Badge variant="info">{user.role}</Badge>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <Badge 
                            variant={
                              user.status === 'active' ? 'success' : 
                              user.status === 'suspended' ? 'warning' : 'critical'
                            }
                          >
                            {user.status}
                          </Badge>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {new Date(user.lastLogin).toLocaleDateString()}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium space-x-2">
                          {user.status === 'active' ? (
                            <Button 
                              variant="outlined" 
                              color="warning" 
                              size="small"
                              onClick={() => handleUserAction(user.id, 'suspend')}
                            >
                              Suspender
                            </Button>
                          ) : (
                            <Button 
                              variant="outlined" 
                              color="success" 
                              size="small"
                              onClick={() => handleUserAction(user.id, 'activate')}
                            >
                              Ativar
                            </Button>
                          )}
                          <Button 
                            variant="outlined" 
                            color="critical" 
                            size="small"
                            onClick={() => handleUserAction(user.id, 'delete')}
                          >
                            Excluir
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Security Tab */}
      {activeTab === 'security' && metrics && (
        <div className="space-y-6">
          <Typography variant="h4" className="font-bold">
            Monitoramento de Segurança
          </Typography>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card variant="critical">
              <div className="p-6 text-center">
                <Typography variant="h3" className="font-bold text-red-600 mb-2">
                  {metrics.security.failedLogins}
                </Typography>
                <Typography variant="body2" className="text-gray-600">
                  Tentativas de Login Falhadas (24h)
                </Typography>
              </div>
            </Card>

            <Card variant="medical">
              <div className="p-6 text-center">
                <Typography variant="h3" className="font-bold text-yellow-600 mb-2">
                  {metrics.security.suspiciousActivity}
                </Typography>
                <Typography variant="body2" className="text-gray-600">
                  Atividades Suspeitas Detectadas
                </Typography>
              </div>
            </Card>

            <Card variant="medical">
              <div className="p-6 text-center">
                <Typography variant="h3" className="font-bold text-green-600 mb-2">
                  99.9%
                </Typography>
                <Typography variant="body2" className="text-gray-600">
                  Uptime de Segurança
                </Typography>
              </div>
            </Card>
          </div>

          <Card variant="medical">
            <div className="p-6">
              <Typography variant="h5" className="font-bold mb-4">
                Ações de Segurança
              </Typography>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Button variant="contained" color="primary" className="w-full">
                  Executar Scan de Segurança
                </Button>
                <Button variant="outlined" color="warning" className="w-full">
                  Forçar Logout de Todos os Usuários
                </Button>
                <Button variant="outlined" color="primary" className="w-full">
                  Backup de Segurança
                </Button>
                <Button variant="outlined" color="critical" className="w-full">
                  Modo de Emergência
                </Button>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Audit Logs Tab */}
      {activeTab === 'logs' && (
        <div className="space-y-6">
          <div className="flex justify-between items-center">
            <Typography variant="h4" className="font-bold">
              Logs de Auditoria
            </Typography>
            <Button variant="outlined" color="primary" onClick={exportAuditLogs}>
              Exportar Logs
            </Button>
          </div>

          <Card variant="medical">
            <div className="p-6">
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Timestamp
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Usuário
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Ação
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Recurso
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        IP
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {auditLogs.map((log) => (
                      <tr key={log.id}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {new Date(log.timestamp).toLocaleString()}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {log.user}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <Badge variant="info">{log.action}</Badge>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {log.resource}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {log.ipAddress}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}

export default AdminDashboard

