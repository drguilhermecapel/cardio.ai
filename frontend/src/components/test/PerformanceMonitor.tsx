// frontend/src/components/test/PerformanceMonitor.tsx

import React, { useState, useEffect, useRef } from 'react';
import { 
  ChartBarIcon,
  ClockIcon,
  CpuChipIcon,
  SignalIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon
} from '@heroicons/react/24/solid';

interface PerformanceMetric {
  timestamp: Date;
  uploadTime: number;
  digitizationTime: number;
  analysisTime: number;
  totalTime: number;
  memoryUsage: number;
  cpuUsage: number;
}

interface PerformanceStats {
  current: PerformanceMetric;
  average: {
    uploadTime: number;
    digitizationTime: number;
    analysisTime: number;
    totalTime: number;
  };
  trend: {
    direction: 'up' | 'down' | 'stable';
    percentage: number;
  };
}

const PerformanceMonitor: React.FC = () => {
  const [metrics, setMetrics] = useState<PerformanceMetric[]>([]);
  const [stats, setStats] = useState<PerformanceStats | null>(null);
  const [isMonitoring, setIsMonitoring] = useState<boolean>(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const generateMockMetric = (): PerformanceMetric => {
    const baseUpload = 200 + Math.random() * 100;
    const baseDigitization = 1500 + Math.random() * 500;
    const baseAnalysis = 2000 + Math.random() * 800;
    
    return {
      timestamp: new Date(),
      uploadTime: baseUpload,
      digitizationTime: baseDigitization,
      analysisTime: baseAnalysis,
      totalTime: baseUpload + baseDigitization + baseAnalysis,
      memoryUsage: 60 + Math.random() * 30, // 60-90%
      cpuUsage: 30 + Math.random() * 40 // 30-70%
    };
  };

  const calculateStats = (metricsArray: PerformanceMetric[]): PerformanceStats => {
    if (metricsArray.length === 0) {
      const mockMetric = generateMockMetric();
      return {
        current: mockMetric,
        average: {
          uploadTime: mockMetric.uploadTime,
          digitizationTime: mockMetric.digitizationTime,
          analysisTime: mockMetric.analysisTime,
          totalTime: mockMetric.totalTime
        },
        trend: { direction: 'stable', percentage: 0 }
      };
    }

    const current = metricsArray[metricsArray.length - 1];
    const average = {
      uploadTime: metricsArray.reduce((sum, m) => sum + m.uploadTime, 0) / metricsArray.length,
      digitizationTime: metricsArray.reduce((sum, m) => sum + m.digitizationTime, 0) / metricsArray.length,
      analysisTime: metricsArray.reduce((sum, m) => sum + m.analysisTime, 0) / metricsArray.length,
      totalTime: metricsArray.reduce((sum, m) => sum + m.totalTime, 0) / metricsArray.length
    };

    // Calcular tendência baseada nos últimos 5 pontos
    const recentMetrics = metricsArray.slice(-5);
    const oldAvg = recentMetrics.slice(0, Math.floor(recentMetrics.length / 2))
      .reduce((sum, m) => sum + m.totalTime, 0) / Math.floor(recentMetrics.length / 2) || current.totalTime;
    const newAvg = recentMetrics.slice(Math.floor(recentMetrics.length / 2))
      .reduce((sum, m) => sum + m.totalTime, 0) / Math.ceil(recentMetrics.length / 2) || current.totalTime;
    
    const trendPercentage = ((newAvg - oldAvg) / oldAvg) * 100;
    const trend = {
      direction: Math.abs(trendPercentage) < 5 ? 'stable' : trendPercentage > 0 ? 'up' : 'down',
      percentage: Math.abs(trendPercentage)
    } as const;

    return { current, average, trend };
  };

  const startMonitoring = () => {
    setIsMonitoring(true);
    intervalRef.current = setInterval(() => {
      const newMetric = generateMockMetric();
      setMetrics(prev => {
        const updated = [...prev, newMetric].slice(-20); // Manter apenas os últimos 20 pontos
        return updated;
      });
    }, 3000); // Atualizar a cada 3 segundos
  };

  const stopMonitoring = () => {
    setIsMonitoring(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const clearMetrics = () => {
    setMetrics([]);
    setStats(null);
  };

  useEffect(() => {
    const newStats = calculateStats(metrics);
    setStats(newStats);
  }, [metrics]);

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  const formatTime = (ms: number) => {
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  const getTrendIcon = (direction: 'up' | 'down' | 'stable') => {
    switch (direction) {
      case 'up':
        return <ArrowTrendingUpIcon className="h-4 w-4 text-red-500" />;
      case 'down':
        return <ArrowTrendingDownIcon className="h-4 w-4 text-green-500" />;
      default:
        return <div className="h-4 w-4 bg-gray-400 rounded-full"></div>;
    }
  };

  const getTrendColor = (direction: 'up' | 'down' | 'stable') => {
    switch (direction) {
      case 'up': return 'text-red-600';
      case 'down': return 'text-green-600';
      default: return 'text-gray-600';
    }
  };

  return (
    <div className="bg-white p-6 rounded-xl shadow-lg">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-bold text-gray-800 flex items-center">
          <ChartBarIcon className="h-6 w-6 mr-2" />
          Monitor de Performance
        </h3>
        <div className="flex items-center space-x-2">
          <button
            onClick={clearMetrics}
            className="px-3 py-1 bg-gray-200 text-gray-700 rounded text-sm hover:bg-gray-300"
          >
            Limpar
          </button>
          <button
            onClick={isMonitoring ? stopMonitoring : startMonitoring}
            className={`px-4 py-2 rounded-lg text-sm font-medium ${
              isMonitoring 
                ? 'bg-red-600 text-white hover:bg-red-700' 
                : 'bg-green-600 text-white hover:bg-green-700'
            }`}
          >
            {isMonitoring ? 'Parar Monitor' : 'Iniciar Monitor'}
          </button>
        </div>
      </div>

      {stats && (
        <>
          {/* Current Performance */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-blue-50 p-4 rounded-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Upload</p>
                  <p className="text-lg font-bold text-blue-600">
                    {formatTime(stats.current.uploadTime)}
                  </p>
                </div>
                <ClockIcon className="h-8 w-8 text-blue-400" />
              </div>
            </div>

            <div className="bg-green-50 p-4 rounded-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Digitalização</p>
                  <p className="text-lg font-bold text-green-600">
                    {formatTime(stats.current.digitizationTime)}
                  </p>
                </div>
                <SignalIcon className="h-8 w-8 text-green-400" />
              </div>
            </div>

            <div className="bg-purple-50 p-4 rounded-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Análise IA</p>
                  <p className="text-lg font-bold text-purple-600">
                    {formatTime(stats.current.analysisTime)}
                  </p>
                </div>
                <CpuChipIcon className="h-8 w-8 text-purple-400" />
              </div>
            </div>

            <div className="bg-orange-50 p-4 rounded-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Total</p>
                  <p className="text-lg font-bold text-orange-600">
                    {formatTime(stats.current.totalTime)}
                  </p>
                </div>
                <div className="flex items-center">
                  {getTrendIcon(stats.trend.direction)}
                </div>
              </div>
            </div>
          </div>

          {/* Averages and Trends */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div className="border rounded-lg p-4">
              <h4 className="font-semibold text-gray-800 mb-3">Médias Históricas</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Upload:</span>
                  <span className="text-sm font-medium">{formatTime(stats.average.uploadTime)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Digitalização:</span>
                  <span className="text-sm font-medium">{formatTime(stats.average.digitizationTime)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Análise IA:</span>
                  <span className="text-sm font-medium">{formatTime(stats.average.analysisTime)}</span>
                </div>
                <div className="flex justify-between border-t pt-2">
                  <span className="text-sm font-semibold text-gray-800">Total:</span>
                  <span className="text-sm font-bold">{formatTime(stats.average.totalTime)}</span>
                </div>
              </div>
            </div>

            <div className="border rounded-lg p-4">
              <h4 className="font-semibold text-gray-800 mb-3">Tendência de Performance</h4>
              <div className="flex items-center justify-center space-x-3">
                {getTrendIcon(stats.trend.direction)}
                <div className="text-center">
                  <p className={`text-lg font-bold ${getTrendColor(stats.trend.direction)}`}>
                    {stats.trend.direction === 'stable' ? 'Estável' : 
                     stats.trend.direction === 'up' ? 'Degradando' : 'Melhorando'}
                  </p>
                  <p className="text-sm text-gray-600">
                    {stats.trend.percentage.toFixed(1)}% vs média anterior
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* System Resources */}
          <div className="border rounded-lg p-4">
            <h4 className="font-semibold text-gray-800 mb-3">Recursos do Sistema</h4>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-gray-600">CPU</span>
                  <span className="text-sm font-medium">{stats.current.cpuUsage.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${stats.current.cpuUsage}%` }}
                  ></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-gray-600">Memória</span>
                  <span className="text-sm font-medium">{stats.current.memoryUsage.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-green-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${stats.current.memoryUsage}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>

          {/* Status Indicator */}
          {isMonitoring && (
            <div className="mt-4 flex items-center justify-center">
              <div className="flex items-center space-x-2 text-green-600">
                <div className="h-2 w-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-sm">Monitoramento ativo - {metrics.length} amostras coletadas</span>
              </div>
            </div>
          )}
        </>
      )}

      {!stats && (
        <div className="text-center py-8 text-gray-500">
          <ChartBarIcon className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>Inicie o monitoramento para ver métricas de performance</p>
        </div>
      )}
    </div>
  );
};

export default PerformanceMonitor;

