import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { Heart } from 'lucide-react'
import MedicalCard from '../MedicalCard'

describe('MedicalCard', () => {
  const defaultProps = {
    title: 'Heart Rate',
    value: '72',
    unit: 'BPM',
    icon: Heart,
  }

  it('renders medical card with basic props', () => {
    render(<MedicalCard {...defaultProps} />)

    expect(screen.getByText('Heart Rate')).toBeInTheDocument()
    expect(screen.getByText('72')).toBeInTheDocument()
    expect(screen.getByText('BPM')).toBeInTheDocument()
  })

  it('displays trend information when provided', () => {
    render(<MedicalCard {...defaultProps} trend="up" trendValue="+5%" />)

    expect(screen.getByText('↗')).toBeInTheDocument()
    expect(screen.getByText('+5%')).toBeInTheDocument()
  })

  it('applies correct severity colors for normal state', () => {
    const { container } = render(<MedicalCard {...defaultProps} severity="normal" />)

    const cardElement = container.querySelector('.bg-cyan-500\\/10')
    expect(cardElement).toBeInTheDocument()
  })

  it('applies correct severity colors for warning state', () => {
    const { container } = render(<MedicalCard {...defaultProps} severity="warning" />)

    const cardElement = container.querySelector('.bg-yellow-500\\/10')
    expect(cardElement).toBeInTheDocument()
  })

  it('applies correct severity colors for critical state', () => {
    const { container } = render(<MedicalCard {...defaultProps} severity="critical" />)

    const cardElement = container.querySelector('.bg-red-500\\/10')
    expect(cardElement).toBeInTheDocument()
  })

  it('shows pulse animation for critical severity', () => {
    const { container } = render(<MedicalCard {...defaultProps} severity="critical" />)

    const pulseElement = container.querySelector('.animate-pulse')
    expect(pulseElement).toBeInTheDocument()
  })

  it('renders children content when provided', () => {
    render(
      <MedicalCard {...defaultProps}>
        <div data-testid="child-content">Additional medical info</div>
      </MedicalCard>
    )

    expect(screen.getByTestId('child-content')).toBeInTheDocument()
    expect(screen.getByText('Additional medical info')).toBeInTheDocument()
  })

  it('applies custom className when provided', () => {
    const { container } = render(<MedicalCard {...defaultProps} className="custom-medical-card" />)

    const cardElement = container.querySelector('.custom-medical-card')
    expect(cardElement).toBeInTheDocument()
  })

  it('displays different trend icons correctly', () => {
    const { rerender } = render(<MedicalCard {...defaultProps} trend="up" trendValue="+5%" />)
    expect(screen.getByText('↗')).toBeInTheDocument()

    rerender(<MedicalCard {...defaultProps} trend="down" trendValue="-3%" />)
    expect(screen.getByText('↘')).toBeInTheDocument()

    rerender(<MedicalCard {...defaultProps} trend="stable" trendValue="0%" />)
    expect(screen.getByText('→')).toBeInTheDocument()
  })

  it('handles numeric values correctly', () => {
    render(<MedicalCard {...defaultProps} value={120} />)
    expect(screen.getByText('120')).toBeInTheDocument()
  })

  it('renders without unit when not provided', () => {
    const propsWithoutUnit = { ...defaultProps, unit: undefined }

    render(<MedicalCard {...propsWithoutUnit} />)
    expect(screen.getByText('72')).toBeInTheDocument()
    expect(screen.queryByText('BPM')).not.toBeInTheDocument()
  })

  it('has proper medical accessibility attributes', () => {
    render(<MedicalCard {...defaultProps} />)

    const titleElement = screen.getByText('Heart Rate')
    expect(titleElement).toHaveClass('uppercase', 'tracking-wide')

    const valueElement = screen.getByText('72')
    expect(valueElement).toHaveClass('text-3xl', 'font-bold')
  })

  it('supports glassmorphism design with backdrop blur', () => {
    const { container } = render(<MedicalCard {...defaultProps} />)

    const cardElement = container.querySelector('.backdrop-blur-xl')
    expect(cardElement).toBeInTheDocument()
  })

  it('has hover effects for interactive medical interface', () => {
    const { container } = render(<MedicalCard {...defaultProps} />)

    const cardElement = container.querySelector('.hover\\:scale-105')
    expect(cardElement).toBeInTheDocument()
  })
})
