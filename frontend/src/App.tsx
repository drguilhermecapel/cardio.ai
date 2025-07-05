import { useState, useEffect } from 'react'
import { BrowserRouter as Router } from 'react-router-dom'
import {
  Calendar,
  FileText,
  Users,
  BrainCircuit,
  Video,
  Bell,
  Search,
  Menu,
  X,
  Sun,
  Moon,
  Wifi,
  WifiOff,
  Shield,
  ChevronRight,
  LogOut,
  User,
  Eye,
  Microscope,
  TrendingUp,
  TrendingDown,
  Activity,
  Pill,
  ScanLine,
  Sparkles,
  Heart,
  Zap,
  Stethoscope,
} from 'lucide-react'
import { AuthProvider } from './contexts/AuthContext'
import { useAuth } from './hooks/useAuth'
import LoginPage from './pages/LoginPage'
import MedicalCard from './components/ui/MedicalCard'
import ECGVisualization from './components/ui/ECGVisualization'
import AIInsightPanel from './components/ui/AIInsightPanel'
import ECGImageAnalysisPage from './pages/ECGImageAnalysisPage'

const CardioApp = (): JSX.Element => {
  const [activeModule, setActiveModule] = useState('dashboard')
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [theme, setTheme] = useState('dark')
  const [isOnline] = useState(true)
  const [showNotifications, setShowNotifications] = useState(false)
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false)
  const [notifications] = useState([
    { id: 1, type: 'alert', message: 'Paciente João Silva requer atenção imediata', time: '2 min' },
    { id: 2, type: 'info', message: 'Nova atualização do sistema disponível', time: '1h' },
    { id: 3, type: 'success', message: 'Backup realizado com sucesso', time: '3h' },
  ])

  const [userProfile] = useState({
    name: 'Dr. Admin',
    specialty: 'Clínico Geral',
    photo:
      'https://images.unsplash.com/photo-1612349317150-e413f6a5b16d?w=100&h=100&fit=crop&crop=face',
    performanceScore: 98,
    aiLevel: 'Expert',
  })

  const modules = [
    { id: 'dashboard', name: 'Dashboard', icon: Activity, color: 'from-blue-500 to-cyan-500' },
    { id: 'patients', name: 'Pacientes', icon: Users, color: 'from-green-500 to-emerald-500' },
    {
      id: 'medical-records',
      name: 'Prontuários',
      icon: FileText,
      color: 'from-purple-500 to-pink-500',
    },
    {
      id: 'ai-diagnostics',
      name: 'IA Diagnóstica',
      icon: BrainCircuit,
      color: 'from-orange-500 to-red-500',
    },
    {
      id: 'ecg-image-analysis',
      name: 'Análise de Imagem',
      icon: ScanLine,
      color: 'from-teal-500 to-blue-500',
    },
    {
      id: 'telemedicine',
      name: 'Telemedicina',
      icon: Video,
      color: 'from-indigo-500 to-purple-500',
    },
  ]

  const NotificationPanel = (): JSX.Element => (
    <div className="absolute top-12 right-0 w-80 bg-gray-900/95 backdrop-blur-xl rounded-xl border border-gray-700 shadow-2xl z-50">
      <div className="p-4 border-b border-gray-700">
        <h3 className="text-lg font-semibold text-white">Notificações</h3>
      </div>
      <div className="max-h-96 overflow-y-auto">
        {notifications.map(notification => (
          <div
            key={notification.id}
            className="p-4 border-b border-gray-800 hover:bg-gray-800/50 transition-colors"
          >
            <div className="flex items-start space-x-3">
              <div
                className={`w-2 h-2 rounded-full mt-2 ${
                  notification.type === 'alert'
                    ? 'bg-red-500'
                    : notification.type === 'info'
                      ? 'bg-blue-500'
                      : 'bg-green-500'
                }`}
              />
              <div className="flex-1">
                <p className="text-sm text-white">{notification.message}</p>
                <p className="text-xs text-gray-400 mt-1">{notification.time} atrás</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )

  const AIAssistant = (): JSX.Element | null => {
    const [messages] = useState([{ id: 1, type: 'ai', content: 'Olá! Como posso ajudá-lo hoje?' }])

    if (!aiAssistantOpen) return null

    return (
      <div className="fixed bottom-20 right-6 w-96 h-96 bg-gray-900/95 backdrop-blur-xl rounded-2xl border border-gray-700 shadow-2xl z-50">
        <div className="p-4 border-b border-gray-700 flex items-center justify-between">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-purple-400" />
            Assistente IA
          </h3>
          <button
            onClick={() => setAiAssistantOpen(false)}
            className="p-1 rounded-lg hover:bg-gray-700 transition-colors"
          >
            <X className="w-4 h-4 text-gray-400" />
          </button>
        </div>
        <div className="flex-1 p-4 h-80 overflow-y-auto">
          {messages.map(message => (
            <div
              key={message.id}
              className={`mb-3 ${message.type === 'user' ? 'text-right' : 'text-left'}`}
            >
              <div
                className={`inline-block p-3 rounded-xl max-w-xs ${
                  message.type === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-200'
                }`}
              >
                <p className="text-sm">{message.content}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  const Header = (): JSX.Element => {
    return (
      <header className="fixed top-0 left-0 right-0 h-16 bg-gray-900/80 backdrop-blur-xl border-b border-gray-800/50 z-40 shadow-lg">
        <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/10 via-blue-500/5 to-purple-500/10"></div>
        <div className="relative flex items-center justify-between h-full px-6">
          <div className="flex items-center space-x-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 rounded-xl hover:bg-gray-800 transition-all group"
            >
              <Menu className="w-5 h-5 text-gray-400 group-hover:text-white" />
            </button>

            <div className="flex items-center space-x-3">
              <div className="relative">
                <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg">
                  <BrainCircuit className="w-6 h-6 text-white" />
                </div>
                <div className="absolute -top-1 -right-1 w-4 h-4 bg-green-400 rounded-full animate-pulse" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-white">SPEI</h1>
                <p className="text-xs text-gray-400">Sistema Inteligente</p>
              </div>
            </div>
          </div>

          <div className="flex-1 max-w-md mx-8">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="Buscar pacientes, prontuários..."
                className="w-full pl-10 pr-4 py-2 bg-gray-800/50 border border-gray-700 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
              />
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <button
                onClick={() => {}}
                className="p-2 rounded-xl hover:bg-gray-800 transition-all group"
              >
                <ScanLine className="w-5 h-5 text-gray-400 group-hover:text-white" />
              </button>
              <button
                onClick={() => setAiAssistantOpen(!aiAssistantOpen)}
                className="p-2 rounded-xl hover:bg-gray-800 transition-all group bg-gradient-to-r from-purple-600/10 to-pink-600/10"
              >
                <Sparkles className="w-5 h-5 text-purple-400 group-hover:text-purple-300" />
              </button>
            </div>

            <div
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-xl text-sm ${
                isOnline
                  ? 'bg-green-500/10 text-green-400 border border-green-500/20'
                  : 'bg-red-500/10 text-red-400 border border-red-500/20'
              }`}
            >
              <div className="relative">
                {isOnline ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
                <span
                  className={`absolute -top-1 -right-1 w-2 h-2 rounded-full ${
                    isOnline ? 'bg-green-400' : 'bg-red-400'
                  } animate-pulse`}
                />
              </div>
              <span className="hidden sm:inline">{isOnline ? 'Online' : 'Offline'}</span>
            </div>

            <div className="relative">
              <button
                onClick={() => setShowNotifications(!showNotifications)}
                className="p-2 rounded-xl hover:bg-gray-800 transition-all relative group"
              >
                <Bell className="w-5 h-5 text-gray-400 group-hover:text-white" />
                {notifications.length > 0 && (
                  <span className="absolute -top-1 -right-1 flex items-center justify-center w-5 h-5 bg-gradient-to-r from-red-500 to-pink-500 text-white text-xs rounded-full font-semibold">
                    {notifications.length}
                  </span>
                )}
              </button>
              {showNotifications && <NotificationPanel />}
            </div>

            <button
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              className="p-2 rounded-xl hover:bg-gray-800 transition-all group"
            >
              {theme === 'dark' ? (
                <Sun className="w-5 h-5 text-gray-400 group-hover:text-yellow-400" />
              ) : (
                <Moon className="w-5 h-5 text-gray-400 group-hover:text-white" />
              )}
            </button>

            <div className="flex items-center space-x-3 pl-4 border-l border-gray-800">
              <div className="relative">
                <img
                  src={userProfile.photo}
                  alt={userProfile.name}
                  className="w-10 h-10 rounded-xl ring-2 ring-cyan-500/20"
                />
                <div className="absolute -bottom-1 -right-1 px-1.5 py-0.5 bg-gradient-to-r from-green-500 to-emerald-500 text-white text-xs rounded-md font-semibold">
                  {userProfile.performanceScore}
                </div>
              </div>
              <div>
                <p className="text-sm font-medium text-white">{userProfile.name}</p>
                <p className="text-xs text-gray-400 flex items-center gap-1">
                  <Shield className="w-3 h-3 text-cyan-400" />
                  {userProfile.specialty} • {userProfile.aiLevel}
                </p>
              </div>
            </div>
          </div>
        </div>
      </header>
    )
  }

  const Sidebar = (): JSX.Element => (
    <aside
      className={`fixed left-0 top-16 h-full bg-gray-900/80 backdrop-blur-xl text-white transition-all duration-300 z-30 border-r border-gray-800/50 ${
        sidebarOpen ? 'w-64' : 'w-20'
      }`}
    >
      <div className="absolute inset-0 bg-gradient-to-b from-blue-500/5 to-purple-500/5"></div>
      <nav className="relative mt-8 px-4">
        {modules.map(module => {
          const Icon = module.icon
          const isActive = activeModule === module.id

          return (
            <button
              key={module.id}
              onClick={() => setActiveModule(module.id)}
              className={`w-full mb-2 group relative overflow-hidden rounded-xl transition-all duration-300 ${
                isActive ? 'bg-gradient-to-r ' + module.color : 'hover:bg-gray-800'
              }`}
            >
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent translate-x-[-200%] group-hover:translate-x-[200%] transition-transform duration-700" />

              <div
                className={`relative flex items-center space-x-3 px-4 py-3 ${
                  isActive ? 'text-white' : 'text-gray-400 group-hover:text-white'
                }`}
              >
                <Icon className={`w-5 h-5 flex-shrink-0 ${isActive ? 'animate-pulse' : ''}`} />
                {sidebarOpen && (
                  <>
                    <span className="font-medium">{module.name}</span>
                    {isActive && <ChevronRight className="w-4 h-4 ml-auto" />}
                  </>
                )}
              </div>

              {isActive && (
                <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-white rounded-r-full shadow-lg shadow-white/50" />
              )}
            </button>
          )
        })}
      </nav>

      <div className="absolute bottom-0 w-full p-4">
        <button className="w-full group relative overflow-hidden rounded-xl hover:bg-red-600/20 transition-all duration-300">
          <div className="relative flex items-center space-x-3 px-4 py-3 text-gray-400 group-hover:text-red-400">
            <LogOut className="w-5 h-5 flex-shrink-0" />
            {sidebarOpen && <span className="font-medium">Sair</span>}
          </div>
        </button>
      </div>
    </aside>
  )

  const DashboardModule = (): JSX.Element => {
    const stats = [
      {
        title: 'Pacientes Hoje',
        value: '24',
        change: '+12%',
        icon: Users,
        color: 'from-blue-500 to-cyan-500',
        bgGlow: 'bg-blue-500/20',
        trend: 'up',
      },
      {
        title: 'Consultas Realizadas',
        value: '18',
        change: '+5%',
        icon: Calendar,
        color: 'from-green-500 to-emerald-500',
        bgGlow: 'bg-green-500/20',
        trend: 'up',
      },
      {
        title: 'Exames Pendentes',
        value: '7',
        change: '-3%',
        icon: Microscope,
        color: 'from-yellow-500 to-orange-500',
        bgGlow: 'bg-yellow-500/20',
        trend: 'down',
      },
      {
        title: 'IA Insights',
        value: '12',
        change: '+28%',
        icon: BrainCircuit,
        color: 'from-purple-500 to-pink-500',
        bgGlow: 'bg-purple-500/20',
        trend: 'up',
      },
    ]

    const recentPatients = [
      {
        id: 1,
        name: 'Maria Silva',
        age: 45,
        lastVisit: '10:30',
        status: 'Em consulta',
        priority: 'high',
        aiRisk: 85,
        conditions: ['Hipertensão', 'Diabetes'],
      },
      {
        id: 2,
        name: 'João Santos',
        age: 62,
        lastVisit: '09:45',
        status: 'Aguardando',
        priority: 'medium',
        aiRisk: 60,
        conditions: ['Cardiopatia'],
      },
      {
        id: 3,
        name: 'Ana Costa',
        age: 28,
        lastVisit: '09:00',
        status: 'Finalizado',
        priority: 'low',
        aiRisk: 15,
        conditions: [],
      },
    ]

    const ActivityChart = (): JSX.Element => {
      const [data, setData] = useState(Array.from({ length: 20 }, () => Math.random() * 100))

      useEffect(() => {
        const interval = setInterval(() => {
          setData(prev => [...prev.slice(1), Math.random() * 100])
        }, 1000)
        return () => clearInterval(interval)
      }, [])

      return (
        <div className="h-32 flex items-end space-x-1">
          {data.map((value, idx) => (
            <div
              key={idx}
              className="flex-1 bg-gradient-to-t from-cyan-500 to-blue-500 rounded-t transition-all duration-300"
              style={{ height: `${value}%` }}
            />
          ))}
        </div>
      )
    }

    return (
      <div className="space-y-6">
        <div className="relative overflow-hidden bg-gradient-to-r from-gray-900 via-gray-800 to-gray-900 rounded-2xl p-6 border border-gray-700">
          <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/10 via-transparent to-purple-500/10" />
          <div className="relative z-10">
            <h1 className="text-3xl font-bold text-white mb-2">
              Bem-vindo de volta, {userProfile.name}
            </h1>
            <p className="text-gray-400">
              Sua IA está monitorando {stats[0].value} pacientes em tempo real
            </p>
          </div>
          <div className="absolute -right-10 -top-10 w-40 h-40 bg-gradient-to-br from-cyan-500/20 to-purple-500/20 rounded-full blur-3xl" />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {stats.map((stat, idx) => {
            const Icon = stat.icon
            return (
              <div key={idx} className="relative group">
                <div
                  className={`absolute inset-0 ${stat.bgGlow} rounded-2xl blur-xl group-hover:blur-2xl transition-all duration-300`}
                />

                <div className="relative bg-gray-900/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-700 hover:border-gray-600 transition-all duration-300">
                  <div className="flex items-center justify-between mb-4">
                    <div className={`p-3 rounded-xl bg-gradient-to-br ${stat.color} shadow-lg`}>
                      <Icon className="w-6 h-6 text-white" />
                    </div>
                    <div
                      className={`flex items-center space-x-1 text-sm font-medium ${
                        stat.trend === 'up' ? 'text-green-400' : 'text-red-400'
                      }`}
                    >
                      {stat.trend === 'up' ? (
                        <TrendingUp className="w-4 h-4" />
                      ) : (
                        <TrendingDown className="w-4 h-4" />
                      )}
                      <span>{stat.change}</span>
                    </div>
                  </div>
                  <h3 className="text-3xl font-bold text-white mb-1">{stat.value}</h3>
                  <p className="text-sm text-gray-400">{stat.title}</p>

                  <div className="mt-4 h-8 flex items-end space-x-0.5">
                    {Array.from({ length: 8 }, (_, i) => (
                      <div
                        key={i}
                        className={`flex-1 bg-gradient-to-t ${stat.color} rounded-sm opacity-40`}
                        style={{ height: `${Math.random() * 100}%` }}
                      />
                    ))}
                  </div>
                </div>
              </div>
            )
          })}
        </div>

        <div className="bg-gray-900/80 backdrop-blur-xl rounded-2xl border border-gray-700">
          <div className="p-6 border-b border-gray-700">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                <Users className="w-5 h-5 text-cyan-400" />
                Pacientes em Monitoramento
              </h2>
              <button className="text-sm text-cyan-400 hover:text-cyan-300 transition-colors">
                Ver todos →
              </button>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-800/50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                    Paciente
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                    Risco IA
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                    Última Visita
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                    Ações
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-700">
                {recentPatients.map(patient => (
                  <tr key={patient.id} className="hover:bg-gray-800/30 transition-colors">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="relative">
                          <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-gray-700 to-gray-800 flex items-center justify-center">
                            <User className="w-5 h-5 text-gray-400" />
                          </div>
                          <div
                            className={`absolute -bottom-1 -right-1 w-3 h-3 rounded-full ${
                              patient.priority === 'high'
                                ? 'bg-red-500'
                                : patient.priority === 'medium'
                                  ? 'bg-yellow-500'
                                  : 'bg-green-500'
                            } ring-2 ring-gray-900`}
                          />
                        </div>
                        <div className="ml-4">
                          <div className="text-sm font-medium text-white">{patient.name}</div>
                          <div className="text-xs text-gray-400">
                            {patient.age} anos • {patient.conditions.join(', ') || 'Sem condições'}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center space-x-2">
                        <div className="w-24 bg-gray-700 rounded-full h-2 overflow-hidden">
                          <div
                            className={`h-full rounded-full bg-gradient-to-r ${
                              patient.aiRisk > 70
                                ? 'from-red-500 to-orange-500'
                                : patient.aiRisk > 40
                                  ? 'from-yellow-500 to-orange-500'
                                  : 'from-green-500 to-emerald-500'
                            }`}
                            style={{ width: `${patient.aiRisk}%` }}
                          />
                        </div>
                        <span
                          className={`text-xs font-medium ${
                            patient.aiRisk > 70
                              ? 'text-red-400'
                              : patient.aiRisk > 40
                                ? 'text-yellow-400'
                                : 'text-green-400'
                          }`}
                        >
                          {patient.aiRisk}%
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-400">
                      {patient.lastVisit}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span
                        className={`px-3 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                          patient.status === 'Em consulta'
                            ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                            : patient.status === 'Aguardando'
                              ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
                              : 'bg-gray-500/20 text-gray-400 border border-gray-500/30'
                        }`}
                      >
                        {patient.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      <div className="flex items-center space-x-2">
                        <button className="p-1.5 rounded-lg hover:bg-gray-700 transition-colors group">
                          <Eye className="w-4 h-4 text-gray-400 group-hover:text-cyan-400" />
                        </button>
                        <button className="p-1.5 rounded-lg hover:bg-gray-700 transition-colors group">
                          <FileText className="w-4 h-4 text-gray-400 group-hover:text-cyan-400" />
                        </button>
                        <button className="p-1.5 rounded-lg hover:bg-gray-700 transition-colors group">
                          <BrainCircuit className="w-4 h-4 text-gray-400 group-hover:text-purple-400" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <AIInsightPanel />
          <ECGVisualization heartRate={72} rhythm="Sinusal Normal" isRealTime={true} height={300} />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
          <MedicalCard
            title="Frequência Cardíaca"
            value={72}
            unit="BPM"
            icon={Heart}
            trend="stable"
            trendValue="Normal"
            severity="normal"
          />
          <MedicalCard
            title="Pressão Arterial"
            value="120/80"
            unit="mmHg"
            icon={Activity}
            trend="up"
            trendValue="+2%"
            severity="normal"
          />
          <MedicalCard
            title="Saturação O2"
            value={98}
            unit="%"
            icon={Stethoscope}
            trend="stable"
            trendValue="Estável"
            severity="normal"
          />
        </div>

        <div className="bg-gray-900/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white flex items-center gap-2">
              <Zap className="w-5 h-5 text-cyan-400" />
              Monitoramento em Tempo Real
            </h3>
            <div className="flex items-center space-x-2">
              <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              <span className="text-xs text-gray-400">Live</span>
            </div>
          </div>

          <ActivityChart />

          <div className="grid grid-cols-3 gap-4 mt-6">
            <div className="text-center">
              <p className="text-2xl font-bold text-cyan-400">98.2%</p>
              <p className="text-xs text-gray-400">Uptime Sistema</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-green-400">1.2s</p>
              <p className="text-xs text-gray-400">Resp. Média IA</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-purple-400">847</p>
              <p className="text-xs text-gray-400">Análises/hora</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { icon: Calendar, label: 'Nova Consulta', color: 'from-blue-500 to-cyan-500' },
            { icon: Pill, label: 'Prescrição Rápida', color: 'from-green-500 to-emerald-500' },
            { icon: Microscope, label: 'Solicitar Exame', color: 'from-purple-500 to-pink-500' },
            { icon: BrainCircuit, label: 'Análise IA', color: 'from-orange-500 to-red-500' },
          ].map((action, idx) => {
            const Icon = action.icon
            return (
              <button
                key={idx}
                className="group relative overflow-hidden rounded-xl bg-gray-900/80 backdrop-blur-xl border border-gray-700 p-4 hover:border-gray-600 transition-all duration-300"
              >
                <div className="absolute inset-0 bg-gradient-to-r opacity-0 group-hover:opacity-10 transition-opacity duration-300" />
                <div className="relative flex flex-col items-center space-y-2">
                  <div className={`p-3 rounded-xl bg-gradient-to-br ${action.color}`}>
                    <Icon className="w-6 h-6 text-white" />
                  </div>
                  <span className="text-sm text-gray-300">{action.label}</span>
                </div>
              </button>
            )
          })}
        </div>
      </div>
    )
  }

  const renderModule = (): JSX.Element => {
    switch (activeModule) {
      case 'dashboard':
        return <DashboardModule />
      case 'ecg-image-analysis':
        return <ECGImageAnalysisPage />
      default:
        return (
          <div className="flex items-center justify-center h-[calc(100vh-200px)]">
            <div className="text-center">
              <div className="relative">
                <BrainCircuit className="w-20 h-20 text-gray-700 mx-auto mb-4" />
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-24 h-24 bg-purple-500/20 rounded-full animate-ping" />
                </div>
              </div>
              <p className="text-gray-400 text-lg">Módulo em desenvolvimento</p>
              <p className="text-gray-500 text-sm mt-2">Em breve novas funcionalidades</p>
            </div>
          </div>
        )
    }
  }

  return (
    <div className="min-h-screen bg-gray-950">
      <div className="fixed inset-0 bg-gradient-to-br from-gray-900 via-gray-950 to-black" />

      <div className="relative z-10">
        <Header />
        <Sidebar />

        <main
          className={`transition-all duration-300 ${sidebarOpen ? 'ml-64' : 'ml-20'} mt-16 p-6`}
        >
          {renderModule()}
        </main>

        <AIAssistant />

        <button
          onClick={() => setAiAssistantOpen(!aiAssistantOpen)}
          className="fixed bottom-6 right-6 w-14 h-14 bg-gradient-to-r from-purple-600 to-pink-600 rounded-full shadow-lg shadow-purple-500/25 flex items-center justify-center hover:scale-110 transition-transform group"
        >
          <Sparkles className="w-6 h-6 text-white" />
          <div className="absolute -top-2 -right-2 w-4 h-4 bg-green-400 rounded-full animate-pulse" />
        </button>
      </div>
    </div>
  )
}

function App(): JSX.Element {
  const { isAuthenticated } = useAuth()

  if (!isAuthenticated) {
    return <LoginPage />
  }

  return <CardioApp />
}

function AppWithAuth(): JSX.Element {
  return (
    <AuthProvider>
      <Router>
        <App />
      </Router>
    </AuthProvider>
  )
}

export default AppWithAuth
