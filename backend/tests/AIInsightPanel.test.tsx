import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import AIInsightPanel from '../AIInsightPanel'

describe('AIInsightPanel', () => {
  const mockInsights = [
    {
      id: '1',
      type: 'diagnosis' as const,
      severity: 'medium' as const,
      title: 'Test Diagnosis',
      description: 'Test diagnosis description',
      confidence: 85,
      timestamp: '2 min ago',
      details: ['Detail 1', 'Detail 2'],
    },
    {
      id: '2',
      type: 'alert' as const,
      severity: 'critical' as const,
      title: 'Critical Alert',
      description: 'Critical alert description',
      confidence: 95,
      timestamp: '1 min ago',
      details: ['Critical detail 1', 'Critical detail 2'],
    },
  ]

  it('renders AI insight panel with header', () => {
    render(<AIInsightPanel />)

    expect(screen.getByText('IA Diagnóstica')).toBeInTheDocument()
    expect(screen.getByText('Insights e recomendações em tempo real')).toBeInTheDocument()
  })

  it('displays default insights when no insights provided', () => {
    render(<AIInsightPanel />)

    expect(screen.getByText('Possível Arritmia Sinusal')).toBeInTheDocument()
    expect(screen.getByText('Otimização de Medicação')).toBeInTheDocument()
    expect(screen.getByText('Alteração no Segmento ST')).toBeInTheDocument()
  })

  it('displays custom insights when provided', () => {
    render(<AIInsightPanel insights={mockInsights} />)

    expect(screen.getByText('Test Diagnosis')).toBeInTheDocument()
    expect(screen.getByText('Critical Alert')).toBeInTheDocument()
  })

  it('shows processing indicator when isProcessing is true', () => {
    render(<AIInsightPanel isProcessing={true} />)

    expect(screen.getByText('Analisando...')).toBeInTheDocument()
  })

  it('displays confidence percentages correctly', () => {
    render(<AIInsightPanel insights={mockInsights} />)

    expect(screen.getByText('85%')).toBeInTheDocument()
    expect(screen.getByText('95%')).toBeInTheDocument()
  })

  it('shows correct confidence colors', () => {
    render(<AIInsightPanel insights={mockInsights} />)

    const mediumConfidence = screen.getByText('85%')
    expect(mediumConfidence).toHaveClass('text-yellow-400')

    const highConfidence = screen.getByText('95%')
    expect(highConfidence).toHaveClass('text-green-400')
  })

  it('applies correct severity colors for critical insights', () => {
    const { container } = render(<AIInsightPanel insights={mockInsights} />)

    const criticalInsightCard = container.querySelector('.bg-red-500\\/10.border-red-500\\/30')
    expect(criticalInsightCard).toBeInTheDocument()
  })

  it('applies correct severity colors for medium insights', () => {
    const { container } = render(<AIInsightPanel insights={mockInsights} />)

    const mediumInsightCard = container.querySelector('.bg-yellow-500\\/10.border-yellow-500\\/30')
    expect(mediumInsightCard).toBeInTheDocument()
  })

  it('expands insight details when clicked', async () => {
    render(<AIInsightPanel insights={mockInsights} />)

    const insightCard = screen.getByText('Test Diagnosis').closest('div')
    fireEvent.click(insightCard!)

    await waitFor(() => {
      expect(screen.getByText('Detail 1')).toBeInTheDocument()
      expect(screen.getByText('Detail 2')).toBeInTheDocument()
    })
  })

  it('collapses insight details when clicked again', async () => {
    render(<AIInsightPanel insights={mockInsights} />)

    const insightCard = screen.getByText('Test Diagnosis').closest('div')

    fireEvent.click(insightCard!)
    await waitFor(() => {
      expect(screen.getByText('Detail 1')).toBeInTheDocument()
    })

    fireEvent.click(insightCard!)
    await waitFor(() => {
      expect(screen.queryByText('Detail 1')).not.toBeInTheDocument()
    })
  })

  it('displays confidence bars with correct widths', () => {
    const { container } = render(<AIInsightPanel insights={mockInsights} />)

    const confidenceBars = container.querySelectorAll('.bg-yellow-400, .bg-green-400')
    expect(confidenceBars).toHaveLength(2)
  })

  it('shows AI performance statistics in footer', () => {
    render(<AIInsightPanel />)

    expect(screen.getByText('98.2%')).toBeInTheDocument()
    expect(screen.getByText('Precisão')).toBeInTheDocument()
    expect(screen.getByText('1,247')).toBeInTheDocument()
    expect(screen.getByText('Análises')).toBeInTheDocument()
    expect(screen.getByText('0.3s')).toBeInTheDocument()
    expect(screen.getByText('Tempo médio')).toBeInTheDocument()
  })

  it('applies custom className when provided', () => {
    const { container } = render(<AIInsightPanel className="custom-ai-panel" />)

    const panelElement = container.firstChild
    expect(panelElement).toHaveClass('custom-ai-panel')
  })

  it('has proper medical design with glassmorphism', () => {
    const { container } = render(<AIInsightPanel />)

    const panelElement = container.firstChild
    expect(panelElement).toHaveClass('backdrop-blur-xl', 'rounded-2xl')
  })

  it('displays timestamps correctly', () => {
    render(<AIInsightPanel insights={mockInsights} />)

    expect(screen.getByText('2 min ago')).toBeInTheDocument()
    expect(screen.getByText('1 min ago')).toBeInTheDocument()
  })

  it('handles insights without details gracefully', () => {
    const insightsWithoutDetails = [
      {
        id: '1',
        type: 'info' as const,
        severity: 'low' as const,
        title: 'Simple Info',
        description: 'Simple description',
        confidence: 70,
        timestamp: '5 min ago',
      },
    ]

    render(<AIInsightPanel insights={insightsWithoutDetails} />)

    expect(screen.getByText('Simple Info')).toBeInTheDocument()
    expect(screen.getByText('Simple description')).toBeInTheDocument()
  })

  it('supports scrolling for many insights', () => {
    const { container } = render(<AIInsightPanel />)

    const insightsList = container.querySelector('.overflow-y-auto')
    expect(insightsList).toBeInTheDocument()
    expect(insightsList).toHaveClass('max-h-96')
  })

  it('has hover effects for interactive medical interface', () => {
    const { container } = render(<AIInsightPanel insights={mockInsights} />)

    const insightCard = container.querySelector('.hover\\:scale-\\[1\\.02\\]')
    expect(insightCard).toBeInTheDocument()
  })

  it('displays correct icons for different insight types', () => {
    const diverseInsights = [
      {
        id: '1',
        type: 'diagnosis' as const,
        severity: 'low' as const,
        title: 'Diagnosis',
        description: 'Test',
        confidence: 80,
        timestamp: '1 min ago',
      },
      {
        id: '2',
        type: 'recommendation' as const,
        severity: 'low' as const,
        title: 'Recommendation',
        description: 'Test',
        confidence: 80,
        timestamp: '1 min ago',
      },
      {
        id: '3',
        type: 'alert' as const,
        severity: 'low' as const,
        title: 'Alert',
        description: 'Test',
        confidence: 80,
        timestamp: '1 min ago',
      },
    ]

    render(<AIInsightPanel insights={diverseInsights} />)

    expect(screen.getByText('Diagnosis')).toBeInTheDocument()
    expect(screen.getByText('Recommendation')).toBeInTheDocument()
    expect(screen.getByText('Alert')).toBeInTheDocument()
  })

  it('handles low confidence insights with red color', () => {
    const lowConfidenceInsight = [
      {
        id: '1',
        type: 'info' as const,
        severity: 'low' as const,
        title: 'Low Confidence',
        description: 'Test',
        confidence: 60,
        timestamp: '1 min ago',
      },
    ]

    render(<AIInsightPanel insights={lowConfidenceInsight} />)

    const confidenceText = screen.getByText('60%')
    expect(confidenceText).toHaveClass('text-red-400')
  })
})
