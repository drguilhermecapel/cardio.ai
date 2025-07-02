// frontend/jest.setup.js
// Configurações globais para todos os testes

// Polyfills e configurações globais
import '@testing-library/jest-dom';
import 'jest-canvas-mock';

// Mock do ResizeObserver (necessário para gráficos)
global.ResizeObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// Mock do IntersectionObserver
global.IntersectionObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// Mock do window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(),
    removeListener: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// Mock do navigator
Object.defineProperty(window, 'navigator', {
  value: {
    userAgent: 'jest',
    clipboard: {
      writeText: jest.fn(),
      readText: jest.fn(),
    },
  },
  writable: true,
});

// Mock do localStorage
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
  length: 0,
  key: jest.fn(),
};
global.localStorage = localStorageMock;

// Mock do sessionStorage
global.sessionStorage = localStorageMock;

// Mock do URL.createObjectURL
global.URL.createObjectURL = jest.fn(() => 'mock-url');
global.URL.revokeObjectURL = jest.fn();

// Mock do WebSocket (para real-time ECG)
global.WebSocket = jest.fn().mockImplementation(() => ({
  send: jest.fn(),
  close: jest.fn(),
  addEventListener: jest.fn(),
  removeEventListener: jest.fn(),
}));

// Mock do Worker (para processamento pesado)
global.Worker = jest.fn().mockImplementation(() => ({
  postMessage: jest.fn(),
  terminate: jest.fn(),
  addEventListener: jest.fn(),
  removeEventListener: jest.fn(),
}));

// Mock do fetch
global.fetch = jest.fn();

// Configurações de console para testes
const originalError = console.error;
const originalWarn = console.warn;

beforeAll(() => {
  // Suprimir warnings específicos do React
  console.error = (...args) => {
    if (
      typeof args[0] === 'string' &&
      (args[0].includes('ReactDOMTestUtils') ||
       args[0].includes('act()') ||
       args[0].includes('Not implemented'))
    ) {
      return;
    }
    originalError.call(console, ...args);
  };
  
  console.warn = (...args) => {
    if (
      typeof args[0] === 'string' &&
      args[0].includes('componentWillReceiveProps')
    ) {
      return;
    }
    originalWarn.call(console, ...args);
  };
});

afterAll(() => {
  console.error = originalError;
  console.warn = originalWarn;
});

// Limpar todos os mocks após cada teste
afterEach(() => {
  jest.clearAllMocks();
});

// Configurações adicionais para componentes médicos
// jest.mock('@/services/ecg/analysis', () => ({
//   analyzeECG: jest.fn(),
//   detectArrhythmia: jest.fn(),
//   calculateHeartRate: jest.fn(),
// }));

// jest.mock('@/services/diagnosis/engine', () => ({
//   generateDiagnosis: jest.fn(),
//   assessSeverity: jest.fn(),
//   getRecommendations: jest.fn(),
// }));

// Mock de bibliotecas de gráficos
jest.mock('recharts', () => ({
  ResponsiveContainer: ({ children }) => children,
  LineChart: ({ children }) => children,
  Line: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  Legend: () => null,
}));

// Mock do Chart.js para ECG
jest.mock('chart.js', () => ({
  Chart: jest.fn(),
  registerables: [],
}));

// Timeout padrão para testes assíncronos
jest.setTimeout(10000);

// Configuração para testes de componentes médicos críticos
beforeEach(() => {
  // Reset de dados médicos mock
  global.mockECGData = {
    signal: Array(1000).fill(0).map(() => Math.random()),
    samplingRate: 360,
    duration: 2.78,
    leadConfiguration: 'II',
  };
  
  global.mockPatientData = {
    id: 'TEST_PATIENT_001',
    name: 'Test Patient',
    age: 45,
    sex: 'M',
    medicalHistory: [],
  };
});

// Utilitários globais para testes
global.waitForAsync = (fn, timeout = 5000) => {
  return new Promise((resolve, reject) => {
    const startTime = Date.now();
    
    const checkCondition = async () => {
      try {
        const result = await fn();
        if (result) {
          resolve(result);
        } else if (Date.now() - startTime > timeout) {
          reject(new Error('Timeout waiting for condition'));
        } else {
          setTimeout(checkCondition, 100);
        }
      } catch (error) {
        reject(error);
      }
    };
    
    checkCondition();
  });
};

// Mock de APIs médicas
global.mockMedicalAPIs = () => {
  window.fetch = jest.fn().mockImplementation((url) => {
    if (url.includes('/api/ecg/analyze')) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          diagnosis: 'normal_sinus_rhythm',
          confidence: 0.95,
          heartRate: 72,
          findings: [],
        }),
      });
    }
    
    if (url.includes('/api/patient')) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(global.mockPatientData),
      });
    }
    
    return Promise.reject(new Error('Unknown API endpoint'));
  });
};
