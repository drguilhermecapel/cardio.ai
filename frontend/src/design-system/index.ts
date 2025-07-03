// Design System Moderno para CardioAI Pro
// Fusão entre cardiologia e inteligência artificial

export const designSystem = {
  // Paleta de cores médica futurista
  colors: {
    // Cores primárias - baseadas em tons médicos e tecnológicos
    primary: {
      50: '#f0f9ff',   // Azul muito claro (confiança médica)
      100: '#e0f2fe',  // Azul claro
      200: '#bae6fd',  // Azul suave
      300: '#7dd3fc',  // Azul médio
      400: '#38bdf8',  // Azul vibrante
      500: '#0ea5e9',  // Azul principal (confiança e tecnologia)
      600: '#0284c7',  // Azul escuro
      700: '#0369a1',  // Azul profundo
      800: '#075985',  // Azul muito escuro
      900: '#0c4a6e',  // Azul quase preto
    },
    
    // Cores secundárias - tons de verde médico (saúde e vida)
    secondary: {
      50: '#f0fdf4',   // Verde muito claro
      100: '#dcfce7',  // Verde claro
      200: '#bbf7d0',  // Verde suave
      300: '#86efac',  // Verde médio
      400: '#4ade80',  // Verde vibrante
      500: '#22c55e',  // Verde principal (saúde e vitalidade)
      600: '#16a34a',  // Verde escuro
      700: '#15803d',  // Verde profundo
      800: '#166534',  // Verde muito escuro
      900: '#14532d',  // Verde quase preto
    },
    
    // Cores de alerta médico
    medical: {
      // Vermelho para emergências e alertas críticos
      critical: {
        50: '#fef2f2',
        100: '#fee2e2',
        200: '#fecaca',
        300: '#fca5a5',
        400: '#f87171',
        500: '#ef4444',  // Vermelho principal
        600: '#dc2626',
        700: '#b91c1c',
        800: '#991b1b',
        900: '#7f1d1d',
      },
      
      // Amarelo para avisos e atenção
      warning: {
        50: '#fffbeb',
        100: '#fef3c7',
        200: '#fde68a',
        300: '#fcd34d',
        400: '#fbbf24',
        500: '#f59e0b',  // Amarelo principal
        600: '#d97706',
        700: '#b45309',
        800: '#92400e',
        900: '#78350f',
      },
      
      // Roxo para IA e tecnologia avançada
      ai: {
        50: '#faf5ff',
        100: '#f3e8ff',
        200: '#e9d5ff',
        300: '#d8b4fe',
        400: '#c084fc',
        500: '#a855f7',  // Roxo principal (IA)
        600: '#9333ea',
        700: '#7c3aed',
        800: '#6b21a8',
        900: '#581c87',
      }
    },
    
    // Tons neutros modernos
    neutral: {
      50: '#fafafa',   // Branco quase puro
      100: '#f5f5f5',  // Cinza muito claro
      200: '#e5e5e5',  // Cinza claro
      300: '#d4d4d4',  // Cinza suave
      400: '#a3a3a3',  // Cinza médio
      500: '#737373',  // Cinza principal
      600: '#525252',  // Cinza escuro
      700: '#404040',  // Cinza profundo
      800: '#262626',  // Cinza muito escuro
      900: '#171717',  // Quase preto
    },
    
    // Gradientes futuristas
    gradients: {
      primary: 'linear-gradient(135deg, #0ea5e9 0%, #22c55e 100%)',
      medical: 'linear-gradient(135deg, #ef4444 0%, #f59e0b 100%)',
      ai: 'linear-gradient(135deg, #a855f7 0%, #0ea5e9 100%)',
      dark: 'linear-gradient(135deg, #1f2937 0%, #111827 100%)',
    }
  },
  
  // Tipografia médica moderna
  typography: {
    fontFamily: {
      sans: ['Inter', 'system-ui', 'sans-serif'],
      mono: ['JetBrains Mono', 'Consolas', 'monospace'],
      medical: ['Source Sans Pro', 'Inter', 'sans-serif'],
    },
    
    fontSize: {
      xs: ['0.75rem', { lineHeight: '1rem' }],
      sm: ['0.875rem', { lineHeight: '1.25rem' }],
      base: ['1rem', { lineHeight: '1.5rem' }],
      lg: ['1.125rem', { lineHeight: '1.75rem' }],
      xl: ['1.25rem', { lineHeight: '1.75rem' }],
      '2xl': ['1.5rem', { lineHeight: '2rem' }],
      '3xl': ['1.875rem', { lineHeight: '2.25rem' }],
      '4xl': ['2.25rem', { lineHeight: '2.5rem' }],
      '5xl': ['3rem', { lineHeight: '1' }],
      '6xl': ['3.75rem', { lineHeight: '1' }],
    },
    
    fontWeight: {
      thin: '100',
      light: '300',
      normal: '400',
      medium: '500',
      semibold: '600',
      bold: '700',
      extrabold: '800',
    }
  },
  
  // Espaçamento e dimensões
  spacing: {
    px: '1px',
    0: '0',
    0.5: '0.125rem',
    1: '0.25rem',
    1.5: '0.375rem',
    2: '0.5rem',
    2.5: '0.625rem',
    3: '0.75rem',
    3.5: '0.875rem',
    4: '1rem',
    5: '1.25rem',
    6: '1.5rem',
    7: '1.75rem',
    8: '2rem',
    9: '2.25rem',
    10: '2.5rem',
    11: '2.75rem',
    12: '3rem',
    14: '3.5rem',
    16: '4rem',
    20: '5rem',
    24: '6rem',
    28: '7rem',
    32: '8rem',
    36: '9rem',
    40: '10rem',
    44: '11rem',
    48: '12rem',
    52: '13rem',
    56: '14rem',
    60: '15rem',
    64: '16rem',
    72: '18rem',
    80: '20rem',
    96: '24rem',
  },
  
  // Bordas e raios
  borderRadius: {
    none: '0',
    sm: '0.125rem',
    DEFAULT: '0.25rem',
    md: '0.375rem',
    lg: '0.5rem',
    xl: '0.75rem',
    '2xl': '1rem',
    '3xl': '1.5rem',
    full: '9999px',
  },
  
  // Sombras médicas modernas
  boxShadow: {
    sm: '0 1px 2px 0 rgb(0 0 0 / 0.05)',
    DEFAULT: '0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)',
    md: '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)',
    lg: '0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)',
    xl: '0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1)',
    '2xl': '0 25px 50px -12px rgb(0 0 0 / 0.25)',
    inner: 'inset 0 2px 4px 0 rgb(0 0 0 / 0.05)',
    
    // Sombras especiais para elementos médicos
    medical: '0 4px 20px -2px rgb(34 197 94 / 0.2)',
    critical: '0 4px 20px -2px rgb(239 68 68 / 0.2)',
    ai: '0 4px 20px -2px rgb(168 85 247 / 0.2)',
    glow: '0 0 20px rgb(14 165 233 / 0.3)',
  },
  
  // Animações e transições
  animation: {
    // Animações básicas
    none: 'none',
    spin: 'spin 1s linear infinite',
    ping: 'ping 1s cubic-bezier(0, 0, 0.2, 1) infinite',
    pulse: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
    bounce: 'bounce 1s infinite',
    
    // Animações médicas específicas
    heartbeat: 'heartbeat 1.5s ease-in-out infinite',
    ecgPulse: 'ecgPulse 2s linear infinite',
    breathe: 'breathe 3s ease-in-out infinite',
    fadeIn: 'fadeIn 0.5s ease-out',
    slideUp: 'slideUp 0.3s ease-out',
    scaleIn: 'scaleIn 0.2s ease-out',
  },
  
  // Transições suaves
  transition: {
    none: 'none',
    all: 'all 150ms cubic-bezier(0.4, 0, 0.2, 1)',
    DEFAULT: 'all 150ms cubic-bezier(0.4, 0, 0.2, 1)',
    colors: 'color 150ms cubic-bezier(0.4, 0, 0.2, 1), background-color 150ms cubic-bezier(0.4, 0, 0.2, 1), border-color 150ms cubic-bezier(0.4, 0, 0.2, 1)',
    opacity: 'opacity 150ms cubic-bezier(0.4, 0, 0.2, 1)',
    shadow: 'box-shadow 150ms cubic-bezier(0.4, 0, 0.2, 1)',
    transform: 'transform 150ms cubic-bezier(0.4, 0, 0.2, 1)',
    
    // Transições médicas específicas
    medical: 'all 200ms cubic-bezier(0.4, 0, 0.2, 1)',
    smooth: 'all 300ms cubic-bezier(0.25, 0.46, 0.45, 0.94)',
  },
  
  // Breakpoints responsivos
  screens: {
    sm: '640px',
    md: '768px',
    lg: '1024px',
    xl: '1280px',
    '2xl': '1536px',
  },
  
  // Z-index para camadas
  zIndex: {
    0: '0',
    10: '10',
    20: '20',
    30: '30',
    40: '40',
    50: '50',
    auto: 'auto',
    
    // Z-index específicos para elementos médicos
    tooltip: '1000',
    modal: '1050',
    popover: '1060',
    notification: '1070',
    emergency: '9999',
  }
}

// Keyframes para animações customizadas
export const keyframes = `
  @keyframes heartbeat {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
  }
  
  @keyframes ecgPulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }
  
  @keyframes breathe {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.02); opacity: 0.9; }
  }
  
  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }
  
  @keyframes slideUp {
    from { transform: translateY(20px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
  }
  
  @keyframes scaleIn {
    from { transform: scale(0.95); opacity: 0; }
    to { transform: scale(1); opacity: 1; }
  }
`

export default designSystem

