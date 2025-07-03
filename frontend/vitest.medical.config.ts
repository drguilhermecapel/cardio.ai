/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { configDefaults } from 'vitest/config'

// Configuração específica para compliance médico FDA/ANVISA
export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/setupTests.ts',
    
    // Cobertura obrigatória para software médico
    coverage: {
      provider: 'istanbul',
      enabled: true,
      reporter: ['text', 'json', 'html', 'lcov', 'cobertura'],
      
      // FDA/ANVISA requerem mínimo 80% geral
      thresholds: {
        lines: 80,
        functions: 80,
        branches: 80,
        statements: 80,
        
        // Componentes críticos requerem 100%
        perFile: true,
        100: [
          'src/components/medical/**',
          'src/utils/ecg/**',
          'src/services/diagnosis/**',
          'src/hooks/useECGAnalysis.ts',
          'src/hooks/usePatientMonitor.ts'
        ],
        
        // Componentes de segurança requerem 95%
        95: [
          'src/services/auth/**',
          'src/services/encryption/**',
          'src/utils/validation/**'
        ]
      },
      
      // Arquivos incluídos na cobertura
      include: [
        'src/**/*.{js,jsx,ts,tsx}'
      ],
      
      // Exclusões
      exclude: [
        ...configDefaults.coverage.exclude,
        'src/main.tsx',
        'src/vite-env.d.ts',
        '**/*.d.ts',
        '**/*.test.{js,jsx,ts,tsx}',
        '**/*.spec.{js,jsx,ts,tsx}',
        '**/test-utils/**',
        '**/__mocks__/**'
      ],
      
      // Relatórios detalhados
      reportsDirectory: './coverage',
      skipFull: false,
      all: true,
      clean: true,
      
      // Linhas a ignorar
      ignoreClassMethods: ['constructor'],
      watermarks: {
        statements: [80, 95],
        functions: [80, 95],
        branches: [80, 95],
        lines: [80, 95]
      }
    },
    
    // Configurações de teste
    testTimeout: 30000,
    hookTimeout: 30000,
    teardownTimeout: 10000,
    
    // Relatórios para CI/CD
    reporters: ['default', 'junit', 'json', 'html'],
    outputFile: {
      junit: './test-results/junit.xml',
      json: './test-results/results.json',
      html: './test-results/index.html'
    },
    
    // Pool de threads otimizado
    pool: 'threads',
    poolOptions: {
      threads: {
        singleThread: false,
        isolate: true
      }
    },
    
    // Validações médicas customizadas
    includeSource: ['src/**/*.{js,ts,jsx,tsx}'],
    
    // Benchmarks de performance para componentes críticos
    benchmark: {
      include: ['**/*.bench.{js,ts}'],
      reporters: ['default', 'json'],
      outputFile: './benchmark/results.json'
    }
  },
  
  // Aliases
  resolve: {
    alias: {
      '@': '/src',
      '@components': '/src/components',
      '@utils': '/src/utils',
      '@services': '/src/services',
      '@hooks': '/src/hooks'
    }
  }
})
