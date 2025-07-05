// frontend/src/components/test/SystemStatusMonitor.tsx

import React, { useState, useEffect } from 'react';
import { 
  CheckCircleIcon, 
  XCircleIcon, 
  ClockIcon,
  ExclamationTriangleIcon,
  ServerIcon,
  CpuChipIcon,
  SignalIcon,
  CloudIcon
} from '@heroicons/react/24/solid';

interface SystemStatus {
  component: string;
  status: 'online' | 'offline' | 'warning' | 'checking';
  message: string;
  lastCheck: Date;
  responseTime?: number;
}

interface SystemHealth {
  overall: 'healthy' | 'degraded' | 'critical';
  components: SystemStatus[];
  uptime: string;
  version: string;
}

const SystemStatusMonitor: React.FC = () => {
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [isChecking, setIsChecking] = useState<boolean>(false);
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true);

  const checkSystemHealth = async () => {
    setIsChecking(true);
    
    try {
      // Simular verificações de sistema
      const components: SystemStatus[] = [
        {
          component: 'Backend API',
          status: 'checking',
          message: 'Verificando conectividade...',
          lastCheck: new Date()
        },
        {
          component: 'ECG-Digitiser Service',
          status: 'checking',
          message: 'Testando digitalização...',
          lastCheck: new Date()
        },
        {
          component: 'AI Analysis Engine',
          status: 'checking',
          message: 'Verificando modelo IA...',
          lastCheck: new Date()
        },
        {
          component: 'Database Connection',
          status: 'checking',
          message: 'Testando conexão BD...',
          lastCheck: new Date()
        },
        {
          component: 'File Upload Service',
          status: 'checking',
          message: 'Verificando upload...',
          lastCheck: new Date()
        }
      ];

      setSystemHealth({
        overall: 'healthy',
        components,
        uptime: '99.9%',
        version: '2.1.0'
      });

      // Simular verificações sequenciais
      for (let i = 0; i < components.length; i++) {
        await new Promise(resolve => setTimeout(resolve, 800));
        
        const updatedComponents = [...components];
        const responseTime = Math.random() * 200 + 50; // 50-250ms
        
        // Simular diferentes status
        const statusOptions: SystemStatus['status'][] = ['online', 'online', 'online', 'warning', 'online'];
        const randomStatus = statusOptions[Math.floor(Math.random() * statusOptions.length)];
        
        updatedComponents[i] = {
          ...updatedComponents[i],
          status: randomStatus,
          message: getStatusMessage(updatedComponents[i].component, randomStatus),
          responseTime,
          lastCheck: new Date()
        };

        setSystemHealth(prev => prev ? {
          ...prev,
          components: updatedComponents,
          overall: calculateOverallHealth(updatedComponents)
        } : null);
      }

    } catch (error) {
      console.error('Erro ao verificar status do sistema:', error);
    } finally {
      setIsChecking(false);
    }
  };

  const getStatusMessage = (component: string, status: SystemStatus['status']): string => {
    const messages = {
      online: {
        'Backend API': 'API respondendo normalmente',
        'ECG-Digitiser Service': 'Serviço de digitalização ativo',
        'AI Analysis Engine': 'Modelo IA carregado e funcional',
        'Database Connection': 'Conexão com BD estabelecida',
        'File Upload Service': 'Serviço de upload operacional'
      },
      warning: {
        'Backend API': 'API com latência elevada',
        'ECG-Digitiser Service': 'Serviço com performance reduzida',
        'AI Analysis Engine': 'Modelo usando fallback',
        'Database Connection': 'Conexão instável',
        'File Upload Service': 'Upload com limitações'
      },
      offline: {
        'Backend API': 'API não responsiva',
        'ECG-Digitiser Service': 'Serviço indisponível',
        'AI Analysis Engine': 'Modelo não carregado',
        'Database Connection': 'Falha na conexão',
        'File Upload Service': 'Serviço offline'
      }
    };

    return messages[status]?.[component] || 'Status desconhecido';
  };

  const calculateOverallHealth = (components: SystemStatus[]): SystemHealth['overall'] => {
    const offlineCount = components.filter(c => c.status === 'offline').length;
    const warningCount = components.filter(c => c.status === 'warning').length;

    if (offlineCount > 0) return 'critical';
    if (warningCount > 1) return 'degraded';
    return 'healthy';
  };

  const getStatusIcon = (status: SystemStatus['status']) => {
    switch (status) {
      case 'online':
        return <CheckCircleIcon className="h-5 w-5 text-green-500" />;
      case 'warning':
        return <ExclamationTriangleIcon className="h-5 w-5 text-yellow-500" />;
      case 'offline':
        return <XCircleIcon className="h-5 w-5 text-red-500" />;
      case 'checking':
        return <ClockIcon className="h-5 w-5 text-blue-500 animate-spin" />;
      default:
        return <ClockIcon className="h-5 w-5 text-gray-400" />;
    }
  };

  const getOverallStatusColor = (status: SystemHealth['overall']) => {
    switch (status) {
      case 'healthy': return 'text-green-600 bg-green-100';
      case 'degraded': return 'text-yellow-600 bg-yellow-100';
      case 'critical': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getComponentIcon = (component: string) => {
    switch (component) {
      case 'Backend API': return <ServerIcon className="h-5 w-5" />;
      case 'ECG-Digitiser Service': return <SignalIcon className="h-5 w-5" />;
      case 'AI Analysis Engine': return <CpuChipIcon className="h-5 w-5" />;
      case 'Database Connection': return <CloudIcon className="h-5 w-5" />;
      case 'File Upload Service': return <ServerIcon className="h-5 w-5" />;
      default: return <ServerIcon className="h-5 w-5" />;
    }
  };

  useEffect(() => {
    checkSystemHealth();
  }, []);

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(checkSystemHealth, 30000); // Refresh a cada 30s
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  return (
    <div className="bg-white p-6 rounded-xl shadow-lg">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-bold text-gray-800 flex items-center">
          <ServerIcon className="h-6 w-6 mr-2" />
          Status do Sistema
        </h3>
        <div className="flex items-center space-x-4">
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="mr-2"
            />
            <span className="text-sm text-gray-600">Auto-refresh</span>
          </label>
          <button
            onClick={checkSystemHealth}
            disabled={isChecking}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 text-sm"
          >
            {isChecking ? 'Verificando...' : 'Atualizar'}
          </button>
        </div>
      </div>

      {systemHealth && (
        <>
          {/* Overall Status */}
          <div className={`p-4 rounded-lg mb-6 ${getOverallStatusColor(systemHealth.overall)}`}>
            <div className="flex items-center justify-between">
              <div>
                <h4 className="font-semibold text-lg capitalize">{systemHealth.overall}</h4>
                <p className="text-sm opacity-80">
                  Sistema {systemHealth.overall === 'healthy' ? 'funcionando normalmente' : 
                          systemHealth.overall === 'degraded' ? 'com problemas menores' : 
                          'com problemas críticos'}
                </p>
              </div>
              <div className="text-right">
                <p className="text-sm font-medium">Uptime: {systemHealth.uptime}</p>
                <p className="text-xs opacity-80">Versão: {systemHealth.version}</p>
              </div>
            </div>
          </div>

          {/* Component Status */}
          <div className="space-y-3">
            {systemHealth.components.map((component, index) => (
              <div key={index} className="flex items-center justify-between p-4 border rounded-lg hover:bg-gray-50 transition-colors">
                <div className="flex items-center space-x-3">
                  {getComponentIcon(component.component)}
                  <div>
                    <h5 className="font-medium text-gray-800">{component.component}</h5>
                    <p className="text-sm text-gray-600">{component.message}</p>
                  </div>
                </div>
                <div className="flex items-center space-x-4">
                  {component.responseTime && (
                    <span className="text-xs text-gray-500">
                      {component.responseTime.toFixed(0)}ms
                    </span>
                  )}
                  <span className="text-xs text-gray-500">
                    {component.lastCheck.toLocaleTimeString()}
                  </span>
                  {getStatusIcon(component.status)}
                </div>
              </div>
            ))}
          </div>

          {/* Quick Stats */}
          <div className="mt-6 grid grid-cols-3 gap-4">
            <div className="text-center p-3 bg-green-50 rounded-lg">
              <p className="text-lg font-bold text-green-600">
                {systemHealth.components.filter(c => c.status === 'online').length}
              </p>
              <p className="text-xs text-gray-600">Online</p>
            </div>
            <div className="text-center p-3 bg-yellow-50 rounded-lg">
              <p className="text-lg font-bold text-yellow-600">
                {systemHealth.components.filter(c => c.status === 'warning').length}
              </p>
              <p className="text-xs text-gray-600">Avisos</p>
            </div>
            <div className="text-center p-3 bg-red-50 rounded-lg">
              <p className="text-lg font-bold text-red-600">
                {systemHealth.components.filter(c => c.status === 'offline').length}
              </p>
              <p className="text-xs text-gray-600">Offline</p>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default SystemStatusMonitor;

