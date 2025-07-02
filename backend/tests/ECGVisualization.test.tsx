import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import ECGVisualization from '../ECGVisualization'

const mockContext = {
  fillStyle: '',
  strokeStyle: '',
  lineWidth: 0,
  shadowColor: '',
  shadowBlur: 0,
  fillRect: vi.fn(),
  beginPath: vi.fn(),
  moveTo: vi.fn(),
  lineTo: vi.fn(),
  stroke: vi.fn(),
  fill: vi.fn(),
  arc: vi.fn(),
  scale: vi.fn(),
}

Object.defineProperty(HTMLCanvasElement.prototype, 'getContext', {
  value: vi.fn(() => mockContext),
})

Object.defineProperty(HTMLCanvasElement.prototype, 'offsetWidth', {
  value: 800,
})

Object.defineProperty(HTMLCanvasElement.prototype, 'offsetHeight', {
  value: 200,
})

global.requestAnimationFrame = vi.fn(cb => {
  setTimeout(cb, 16)
  return 1
})

global.cancelAnimationFrame = vi.fn()

describe('ECGVisualization', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  const sampleECGData = [
    { time: 0, amplitude: 0 },
    { time: 0.1, amplitude: 0.5 },
    { time: 0.2, amplitude: -0.3 },
    { time: 0.3, amplitude: 1.0 },
    { time: 0.4, amplitude: 0.2 },
  ]

  it('renders ECG visualization component', () => {
    render(<ECGVisualization />)

    expect(screen.getByText('ECG Monitor')).toBeInTheDocument()
    expect(screen.getByText('Eletrocardiograma em tempo real')).toBeInTheDocument()
  })

  it('displays heart rate correctly', () => {
    render(<ECGVisualization heartRate={85} />)

    expect(screen.getByText('85 BPM')).toBeInTheDocument()
  })

  it('displays rhythm information', () => {
    render(<ECGVisualization rhythm="Arritmia Sinusal" />)

    expect(screen.getByText('Arritmia Sinusal')).toBeInTheDocument()
  })

  it('shows correct heart rate color for normal range', () => {
    render(<ECGVisualization heartRate={75} />)

    const heartRateElement = screen.getByText('75 BPM')
    expect(heartRateElement).toHaveClass('text-green-400')
  })

  it('shows correct heart rate color for bradycardia', () => {
    render(<ECGVisualization heartRate={55} />)

    const heartRateElement = screen.getByText('55 BPM')
    expect(heartRateElement).toHaveClass('text-blue-400')
  })

  it('shows correct heart rate color for tachycardia', () => {
    render(<ECGVisualization heartRate={110} />)

    const heartRateElement = screen.getByText('110 BPM')
    expect(heartRateElement).toHaveClass('text-red-400')
  })

  it('shows correct rhythm color for normal rhythm', () => {
    render(<ECGVisualization rhythm="Sinusal Normal" />)

    const rhythmElement = screen.getByText('Sinusal Normal')
    expect(rhythmElement).toHaveClass('text-green-400')
  })

  it('shows correct rhythm color for arrhythmia', () => {
    render(<ECGVisualization rhythm="Arritmia Ventricular" />)

    const rhythmElement = screen.getByText('Arritmia Ventricular')
    expect(rhythmElement).toHaveClass('text-red-400')
  })

  it('toggles real-time monitoring when button is clicked', async () => {
    render(<ECGVisualization isRealTime={true} />)

    const toggleButton = screen.getByText('Iniciar')
    fireEvent.click(toggleButton)

    await waitFor(() => {
      expect(screen.getByText('Pausar')).toBeInTheDocument()
    })
  })

  it('displays real-time indicator when active', () => {
    render(<ECGVisualization isRealTime={true} />)

    expect(screen.getByText('PAUSADO')).toBeInTheDocument()
  })

  it('renders canvas element for ECG waveform', () => {
    const { container } = render(<ECGVisualization />)

    const canvas = container.querySelector('canvas')
    expect(canvas).toBeInTheDocument()
  })

  it('applies custom height to canvas', () => {
    const { container } = render(<ECGVisualization height={300} />)

    const canvas = container.querySelector('canvas')
    expect(canvas).toHaveStyle({ height: '300px' })
  })

  it('displays technical parameters in footer', () => {
    render(<ECGVisualization />)

    expect(screen.getByText('Velocidade: 25mm/s')).toBeInTheDocument()
    expect(screen.getByText('Amplitude: 10mm/mV')).toBeInTheDocument()
    expect(screen.getByText('Qualidade do sinal: Boa')).toBeInTheDocument()
  })

  it('renders with custom ECG data', () => {
    render(<ECGVisualization data={sampleECGData} />)

    expect(screen.getByText('ECG Monitor')).toBeInTheDocument()
  })

  it('applies custom className', () => {
    const { container } = render(<ECGVisualization className="custom-ecg-class" />)

    const ecgContainer = container.firstChild
    expect(ecgContainer).toHaveClass('custom-ecg-class')
  })

  it('has proper medical design with glassmorphism', () => {
    const { container } = render(<ECGVisualization />)

    const ecgContainer = container.firstChild
    expect(ecgContainer).toHaveClass('backdrop-blur-xl', 'rounded-2xl')
  })

  it('generates sample ECG data when no data provided', () => {
    render(<ECGVisualization />)

    expect(mockContext.beginPath).toHaveBeenCalled()
    expect(mockContext.stroke).toHaveBeenCalled()
  })

  it('draws ECG grid pattern', () => {
    render(<ECGVisualization />)

    expect(mockContext.fillRect).toHaveBeenCalled()
    expect(mockContext.moveTo).toHaveBeenCalled()
    expect(mockContext.lineTo).toHaveBeenCalled()
  })

  it('handles real-time animation properly', async () => {
    render(<ECGVisualization isRealTime={true} />)

    const toggleButton = screen.getByText('Iniciar')
    fireEvent.click(toggleButton)

    await waitFor(() => {
      expect(global.requestAnimationFrame).toHaveBeenCalled()
    })
  })

  it('cleans up animation on unmount', () => {
    const { unmount } = render(<ECGVisualization isRealTime={true} />)

    unmount()

    expect(global.cancelAnimationFrame).toHaveBeenCalled()
  })

  it('supports high DPI displays with device pixel ratio', () => {
    render(<ECGVisualization />)

    expect(mockContext.scale).toHaveBeenCalled()
  })
})
