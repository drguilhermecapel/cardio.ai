// Theme configuration for CardioAI Pro
// Using CSS custom properties for modern theming

export const theme = {
  colors: {
    // Cores primárias - Fusão Cardiologia + IA
    primary: {
      blue: '#0066FF',
      blueLight: '#00AAFF',
      blueLighter: '#66CCFF',
    },
    cardiac: {
      red: '#FF3366',
      redLight: '#FF6B9D',
      redLighter: '#FFB3C6',
    },
    tech: {
      cyan: '#00FFFF',
      cyanLight: '#66FFFF',
      cyanLighter: '#B3FFFF',
    },
    // Cores secundárias
    neutral: {
      dark: '#1A1A1A',
      medium: '#2D2D2D',
      light: '#404040',
      white: '#FFFFFF',
      offWhite: '#F8F9FA',
      lighter: '#E9ECEF',
    },
    status: {
      success: '#00FF88',
      successLight: '#66FFAA',
      warning: '#FFD700',
      warningLight: '#FFED4E',
      error: '#FF3366',
      errorLight: '#FF6B9D',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    sizes: {
      xs: '0.75rem',
      sm: '0.875rem',
      base: '1rem',
      lg: '1.125rem',
      xl: '1.25rem',
      '2xl': '1.5rem',
      '3xl': '1.875rem',
      '4xl': '2.25rem',
      '5xl': '3rem',
    },
    weights: {
      light: 300,
      normal: 400,
      medium: 500,
      semibold: 600,
      bold: 700,
    },
  },
  spacing: {
    xs: '0.25rem',
    sm: '0.5rem',
    md: '1rem',
    lg: '1.5rem',
    xl: '2rem',
    '2xl': '3rem',
    '3xl': '4rem',
  },
  borderRadius: {
    sm: '0.25rem',
    md: '0.5rem',
    lg: '0.75rem',
    xl: '1rem',
    '2xl': '1.5rem',
    full: '9999px',
  },
  shadows: {
    sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
    xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1)',
    glow: '0 0 20px rgba(0, 102, 255, 0.3)',
    cardiacGlow: '0 0 20px rgba(255, 51, 102, 0.3)',
  },
  animations: {
    duration: {
      fast: '150ms',
      normal: '300ms',
      slow: '500ms',
    },
    easing: {
      easeInOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
      easeOut: 'cubic-bezier(0, 0, 0.2, 1)',
      easeIn: 'cubic-bezier(0.4, 0, 1, 1)',
    },
  },
  breakpoints: {
    sm: '640px',
    md: '768px',
    lg: '1024px',
    xl: '1280px',
    '2xl': '1536px',
  },
}

// CSS Custom Properties for runtime theme switching
export const cssVariables = {
  '--color-primary': theme.colors.primary.blue,
  '--color-primary-light': theme.colors.primary.blueLight,
  '--color-cardiac': theme.colors.cardiac.red,
  '--color-cardiac-light': theme.colors.cardiac.redLight,
  '--color-tech': theme.colors.tech.cyan,
  '--color-tech-light': theme.colors.tech.cyanLight,
  '--color-neutral-dark': theme.colors.neutral.dark,
  '--color-neutral-medium': theme.colors.neutral.medium,
  '--color-neutral-light': theme.colors.neutral.light,
  '--color-white': theme.colors.neutral.white,
  '--color-success': theme.colors.status.success,
  '--color-warning': theme.colors.status.warning,
  '--color-error': theme.colors.status.error,
  '--font-family': theme.typography.fontFamily,
  '--border-radius': theme.borderRadius.md,
  '--shadow-glow': theme.shadows.glow,
  '--shadow-cardiac-glow': theme.shadows.cardiacGlow,
  '--animation-duration': theme.animations.duration.normal,
  '--animation-easing': theme.animations.easing.easeInOut,
}

// Apply CSS variables to document root
export const applyTheme = () => {
  if (typeof document !== 'undefined') {
    const root = document.documentElement
    Object.entries(cssVariables).forEach(([property, value]) => {
      root.style.setProperty(property, value)
    })
  }
}

export default theme
