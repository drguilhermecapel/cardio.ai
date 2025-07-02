import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs))
}

export const medicalColors = {
  primary: {
    50: '#eef2ff',
    100: '#e0e7ff',
    200: '#c7d2fe',
    300: '#a5b4fc',
    400: '#818cf8',
    500: '#6366f1', // Main primary color
    600: '#4f46e5',
    700: '#4338ca',
    800: '#3730a3',
    900: '#312e81',
    950: '#1e1b4b',
  },
  medical: {
    50: '#f0f9ff',
    100: '#e0f2fe',
    200: '#bae6fd',
    300: '#7dd3fc',
    400: '#38bdf8',
    500: '#0ea5e9', // Main medical color
    600: '#0284c7',
    700: '#0369a1',
    800: '#075985',
    900: '#0c4a6e',
    950: '#082f49',
  },
  critical: {
    50: '#fef2f2',
    100: '#fee2e2',
    200: '#fecaca',
    300: '#fca5a5',
    400: '#f87171',
    500: '#ef4444', // Main critical color
    600: '#dc2626',
    700: '#b91c1c',
    800: '#991b1b',
    900: '#7f1d1d',
    950: '#450a0a',
  },
  warning: {
    50: '#fffbeb',
    100: '#fef3c7',
    200: '#fde68a',
    300: '#fcd34d',
    400: '#fbbf24',
    500: '#f59e0b', // Main warning color
    600: '#d97706',
    700: '#b45309',
    800: '#92400e',
    900: '#78350f',
    950: '#451a03',
  },
  success: {
    50: '#ecfdf5',
    100: '#d1fae5',
    200: '#a7f3d0',
    300: '#6ee7b7',
    400: '#34d399',
    500: '#10b981', // Main success color
    600: '#059669',
    700: '#047857',
    800: '#065f46',
    900: '#064e3b',
    950: '#022c22',
  },
  info: {
    50: '#eff6ff',
    100: '#dbeafe',
    200: '#bfdbfe',
    300: '#93c5fd',
    400: '#60a5fa',
    500: '#3b82f6', // Main info color
    600: '#2563eb',
    700: '#1d4ed8',
    800: '#1e40af',
    900: '#1e3a8a',
    950: '#172554',
  },
  neutral: {
    50: '#f8fafc',
    100: '#f1f5f9',
    200: '#e2e8f0',
    300: '#cbd5e1',
    400: '#94a3b8',
    500: '#64748b',
    600: '#475569',
    700: '#334155',
    800: '#1e293b',
    900: '#0f172a',
    950: '#020617',
  },
}

export const medicalGradients = {
  primary: 'from-cyan-500 to-blue-600',
  secondary: 'from-purple-500 to-pink-500',
  warning: 'from-yellow-500 to-orange-500',
  success: 'from-green-500 to-emerald-500',
  critical: 'from-red-500 to-pink-600',
  info: 'from-blue-400 to-indigo-500',
  holographic: 'from-cyan-500/20 via-blue-500/20 to-purple-500/20',
  glassmorphism: 'from-white/10 via-white/5 to-transparent',
  medicalPrimary: 'from-cyan-400 via-blue-500 to-indigo-600',
  medicalSecondary: 'from-teal-400 via-cyan-500 to-blue-600',
  heartbeat: 'from-red-400 via-pink-500 to-red-600',
  ecgWave: 'from-green-400 via-emerald-500 to-teal-600',
}

export const medicalShadows = {
  sm: 'shadow-sm shadow-cyan-500/10',
  DEFAULT: 'shadow shadow-cyan-500/20',
  md: 'shadow-md shadow-cyan-500/20',
  lg: 'shadow-lg shadow-cyan-500/25',
  xl: 'shadow-xl shadow-cyan-500/30',
  '2xl': 'shadow-2xl shadow-cyan-500/40',
  inner: 'shadow-inner shadow-cyan-500/10',
  glow: {
    cyan: 'shadow-lg shadow-cyan-500/50',
    blue: 'shadow-lg shadow-blue-500/50',
    purple: 'shadow-lg shadow-purple-500/50',
    green: 'shadow-lg shadow-green-500/50',
    red: 'shadow-lg shadow-red-500/50',
    yellow: 'shadow-lg shadow-yellow-500/50',
  },
  medical: {
    normal: 'shadow-lg shadow-cyan-500/30',
    warning: 'shadow-lg shadow-yellow-500/40',
    critical: 'shadow-lg shadow-red-500/50',
    success: 'shadow-lg shadow-green-500/40',
  },
}

export const medicalAnimations = {
  pulse: 'animate-pulse',
  bounce: 'animate-bounce',
  spin: 'animate-spin',
  ping: 'animate-ping',

  heartbeat: 'animate-[heartbeat_1.5s_ease-in-out_infinite]',
  ecgLine: 'animate-[ecgLine_2s_linear_infinite]',
  breathe: 'animate-[breathe_2s_ease-in-out_infinite]',
  medicalPulse: 'animate-[medicalPulse_1.5s_ease-in-out_infinite]',

  fadeIn: 'animate-[fadeIn_0.5s_ease-in-out]',
  slideIn: 'animate-[slideIn_0.3s_ease-in-out]',
  slideUp: 'animate-[slideUp_0.4s_ease-out]',
  scaleIn: 'animate-[scaleIn_0.2s_ease-out]',

  holographicScan: 'animate-[holographicScan_3s_linear_infinite]',
  dataFlow: 'animate-[dataFlow_4s_linear_infinite]',
  shimmer: 'animate-[shimmer_1.5s_infinite]',
  float: 'animate-[float_3s_ease-in-out_infinite]',

  gradientX: 'animate-[gradientX_15s_ease_infinite]',
  gradientY: 'animate-[gradientY_15s_ease_infinite]',
  gradientXY: 'animate-[gradientXY_15s_ease_infinite]',
}

export const medicalBreakpoints = {
  mobile: '640px',
  tablet: '768px',
  laptop: '1024px',
  desktop: '1280px',
  medicalDisplay: '1536px', // Large medical displays
  wallMount: '1920px', // Wall-mounted medical displays
  ultrawide: '2560px', // Ultra-wide medical workstations
}

export const glassmorphism = {
  light: 'bg-white/10 backdrop-blur-sm border border-white/20',
  DEFAULT: 'bg-white/20 backdrop-blur-md border border-white/30',
  dark: 'bg-black/30 backdrop-blur-lg border border-white/10',
  medical: 'bg-cyan-500/10 backdrop-blur-xl border border-cyan-500/20',
  critical: 'bg-red-500/10 backdrop-blur-xl border border-red-500/20',
  warning: 'bg-yellow-500/10 backdrop-blur-xl border border-yellow-500/20',
  success: 'bg-green-500/10 backdrop-blur-xl border border-green-500/20',
  info: 'bg-blue-500/10 backdrop-blur-xl border border-blue-500/20',
}

export const medicalVariants = {
  card: {
    normal: 'bg-gray-900/80 border-gray-700/50 shadow-lg shadow-cyan-500/20',
    warning: 'bg-yellow-500/10 border-yellow-500/20 shadow-lg shadow-yellow-500/25',
    critical: 'bg-red-500/10 border-red-500/20 shadow-lg shadow-red-500/25',
    success: 'bg-green-500/10 border-green-500/20 shadow-lg shadow-green-500/25',
    info: 'bg-blue-500/10 border-blue-500/20 shadow-lg shadow-blue-500/25',
  },
  button: {
    primary: 'bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700',
    secondary:
      'bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600',
    warning:
      'bg-gradient-to-r from-yellow-500 to-orange-500 hover:from-yellow-600 hover:to-orange-600',
    critical: 'bg-gradient-to-r from-red-500 to-pink-600 hover:from-red-600 hover:to-pink-700',
    success:
      'bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600',
    ghost: 'bg-transparent border border-gray-600 hover:bg-gray-800/50',
  },
  input: {
    normal: 'bg-gray-900/50 border-gray-600 focus:border-cyan-500 focus:ring-cyan-500/20',
    error: 'bg-red-900/20 border-red-500 focus:border-red-400 focus:ring-red-500/20',
    success: 'bg-green-900/20 border-green-500 focus:border-green-400 focus:ring-green-500/20',
  },
}

export const medicalTypography = {
  heading: {
    h1: 'text-4xl font-bold tracking-tight text-white',
    h2: 'text-3xl font-semibold tracking-tight text-white',
    h3: 'text-2xl font-semibold text-white',
    h4: 'text-xl font-semibold text-white',
    h5: 'text-lg font-medium text-white',
    h6: 'text-base font-medium text-white',
  },
  body: {
    large: 'text-lg text-gray-200',
    DEFAULT: 'text-base text-gray-300',
    small: 'text-sm text-gray-400',
    xs: 'text-xs text-gray-500',
  },
  medical: {
    value: 'text-2xl font-bold text-white',
    unit: 'text-lg font-medium text-gray-400',
    label: 'text-sm font-medium text-gray-300 uppercase tracking-wide',
    status: 'text-sm font-semibold',
  },
}

export const medicalSpacing = {
  xs: '0.25rem', // 4px
  sm: '0.5rem', // 8px
  md: '1rem', // 16px
  lg: '1.5rem', // 24px
  xl: '2rem', // 32px
  '2xl': '3rem', // 48px
  '3xl': '4rem', // 64px
  '4xl': '6rem', // 96px
}

export const medicalBorderRadius = {
  none: '0',
  sm: '0.125rem', // 2px
  DEFAULT: '0.25rem', // 4px
  md: '0.375rem', // 6px
  lg: '0.5rem', // 8px
  xl: '0.75rem', // 12px
  '2xl': '1rem', // 16px
  '3xl': '1.5rem', // 24px
  full: '9999px',
}

export const medicalTheme = {
  colors: medicalColors,
  gradients: medicalGradients,
  shadows: medicalShadows,
  animations: medicalAnimations,
  breakpoints: medicalBreakpoints,
  glass: glassmorphism,
  variants: medicalVariants,
  typography: medicalTypography,
  spacing: medicalSpacing,
  borderRadius: medicalBorderRadius,
}

export default medicalTheme
