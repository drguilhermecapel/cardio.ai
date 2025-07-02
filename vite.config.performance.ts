// Performance Optimization Configuration for CardioAI Pro
// Configurações de otimização de performance e lazy loading

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

export default defineConfig({
  plugins: [react()],
  
  // Build optimizations
  build: {
    target: 'es2020',
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true,
        pure_funcs: ['console.log', 'console.info', 'console.debug']
      }
    },
    rollupOptions: {
      output: {
        manualChunks: {
          // Vendor chunks
          'react-vendor': ['react', 'react-dom'],
          'router-vendor': ['react-router-dom'],
          'redux-vendor': ['@reduxjs/toolkit', 'react-redux'],
          'ui-vendor': ['@radix-ui/react-dialog', '@radix-ui/react-dropdown-menu', '@radix-ui/react-select'],
          'chart-vendor': ['chart.js', 'react-chartjs-2', 'recharts'],
          'medical-vendor': ['plotly.js', 'react-plotly.js'],
          'utils-vendor': ['axios', 'date-fns', 'clsx', 'tailwind-merge'],
          'pdf-vendor': ['jspdf', 'jspdf-autotable'],
          
          // Feature chunks
          'auth': ['./src/contexts/AuthContext.tsx'],
          'notifications': ['./src/contexts/NotificationContext.tsx'],
          'medical-api': ['./src/services/medicalAPI.ts'],
          'backup': ['./src/services/backupSystem.ts'],
          'pdf-generator': ['./src/utils/pdfGenerator.ts'],
          
          // Page chunks
          'admin': ['./src/pages/AdminDashboard.tsx'],
          'dashboard': ['./src/pages/DashboardPage.tsx'],
          'patients': ['./src/pages/PatientsPage.tsx'],
          'ecg-analysis': ['./src/pages/ECGAnalysisPage.tsx']
        }
      }
    },
    chunkSizeWarningLimit: 1000,
    sourcemap: false
  },
  
  // Development optimizations
  server: {
    hmr: {
      overlay: false
    }
  },
  
  // Resolve optimizations
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      '@components': resolve(__dirname, 'src/components'),
      '@pages': resolve(__dirname, 'src/pages'),
      '@utils': resolve(__dirname, 'src/utils'),
      '@services': resolve(__dirname, 'src/services'),
      '@contexts': resolve(__dirname, 'src/contexts')
    }
  },
  
  // Dependency pre-bundling
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      'react-router-dom',
      '@reduxjs/toolkit',
      'react-redux',
      'axios',
      'date-fns',
      'chart.js',
      'react-chartjs-2'
    ],
    exclude: [
      'plotly.js',
      'jspdf'
    ]
  }
})

