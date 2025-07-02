// frontend/coverage.setup.js
// Configuração completa de cobertura para React/TypeScript

module.exports = {
  // Padrões de arquivos para coletar cobertura
  collectCoverageFrom: [
    'src/**/*.{js,jsx,ts,tsx}',
    '!src/**/*.d.ts',
    '!src/index.tsx',
    '!src/serviceWorker.ts',
    '!src/setupTests.ts',
    '!src/vite-env.d.ts',
    '!src/**/*.stories.{js,jsx,ts,tsx}',
    '!src/**/__tests__/**',
    '!src/**/*.test.{js,jsx,ts,tsx}',
    '!src/**/*.spec.{js,jsx,ts,tsx}'
  ],

  // Diretório de saída para relatórios
  coverageDirectory: 'coverage',

  // Limiares de cobertura (FDA/ANVISA requerem >80%)
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    },
    // Componentes críticos médicos requerem 100%
    './src/components/medical/': {
      branches: 100,
      functions: 100,
      lines: 100,
      statements: 100
    },
    './src/utils/ecg/': {
      branches: 100,
      functions: 100,
      lines: 100,
      statements: 100
    },
    './src/services/diagnosis/': {
      branches: 95,
      functions: 95,
      lines: 95,
      statements: 95
    }
  },

  // Formatos de relatório
  coverageReporters: [
    'json',
    'lcov',
    'text',
    'text-summary',
    'html',
    'cobertura' // Para CI/CD
  ],

  // Caminhos para ignorar
  coveragePathIgnorePatterns: [
    '/node_modules/',
    '/build/',
    '/dist/',
    '/coverage/',
    '/.next/',
    '/public/'
  ],

  // Transformações
  transform: {
    '^.+\\.(js|jsx|ts|tsx)$': ['@swc/jest', {
      jsc: {
        transform: {
          react: {
            runtime: 'automatic'
          }
        }
      }
    }]
  },

  // Configuração de ambiente de teste
  testEnvironment: 'jsdom',
  
  // Setup files
  setupFilesAfterEnv: [
    '<rootDir>/src/setupTests.ts',
    '<rootDir>/coverage.setup.js'
  ],

  // Module name mapper para imports absolutos
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '^@components/(.*)$': '<rootDir>/src/components/$1',
    '^@utils/(.*)$': '<rootDir>/src/utils/$1',
    '^@services/(.*)$': '<rootDir>/src/services/$1',
    '\\.(css|less|scss|sass)$': 'identity-obj-proxy',
    '\\.(jpg|jpeg|png|gif|svg)$': '<rootDir>/__mocks__/fileMock.js'
  }
};
