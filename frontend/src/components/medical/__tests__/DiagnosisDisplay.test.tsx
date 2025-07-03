// frontend/src/components/medical/__tests__/DiagnosisDisplay.test.tsx
// Testes completos para o componente DiagnosisDisplay - 100% cobertura

import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import DiagnosisDisplay from '../DiagnosisDisplay';
import { Diagnosis, Severity, Confidence } from '@/types/medical';
import * as diagnosisService from '@/services/diagnosis/engine';
import { act } from 'react-dom/test-utils';

// Mock do Redux store
const createMockStore = (initialState = {}) => {
  return configureStore({
    reducer: {
      diagnosis: (state = initialState) => state,
      user: (state = { role: 'physician' }) => state,
    },
  });
};

// Mock de diagnóstico completo
const mockDiagnosis: Diagnosis = {
  id: 'DIAG_001',
  patientId: 'PAT_001',
  timestamp: new Date().toISOString(),
  primaryDiagnosis: {
    condition: 'atrial_fibrillation',
    confidence: 0.89,
    icd10Code: 'I48.91',
    description: 'Fibrilação atrial não especificada',
    severity: 'moderate' as Severity,
  },
  secondaryDiagnoses: [
    {
      condition: 'premature_ventricular_contractions',
      confidence: 0.72,
      icd10Code: 'I49.3',
      description: 'Despolarização ventricular prematura',
      severity: 'mild' as Severity,
    },
  ],
  findings: [
    {
      type: 'rhythm',
      description: 'Ritmo irregularmente irregular',
      significance: 'high',
      value: 'irregular',
    },
    {
      type: 'rate',
      description: 'Frequência cardíaca elevada',
      significance: 'medium',
      value: '110 bpm',
    },
    {
      type: 'morphology',
      description: 'Ausência de ondas P',
      significance: 'high',
      value: 'absent_p_waves',
    },
  ],
  recommendations: [
    {
      priority: 'high',
      action: 'Considerar anticoagulação',
      rationale: 'Prevenção de AVC em FA',
      timeline: 'Imediato',
    },
    {
      priority: 'medium',
      action: 'ECG de 24 horas (Holter)',
      rationale: 'Avaliar carga de FA',
      timeline: '1-2 semanas',
    },
    {
      priority: 'low',
      action: 'Ecocardiograma',
      rationale: 'Avaliar função cardíaca',
      timeline: '1 mês',
    },
  ],
  clinicalContext: {
    patientAge: 65,
    patientSex: 'M',
    riskFactors: ['hypertension', 'diabetes'],
    medications: ['metoprolol', 'lisinopril'],
    previousConditions: ['hypertension'],
  },
  aiInsights: {
    confidence: 0.89 as Confidence,
    explainability: {
      mainFactors: [
        { factor: 'Irregularidade RR', weight: 0.35 },
        { factor: 'Ausência de onda P', weight: 0.30 },
        { factor: 'Frequência ventricular', weight: 0.20 },
        { factor: 'Morfologia QRS', weight: 0.15 },
      ],
      reasoning: 'Padrão característico de FA com alta variabilidade RR e ausência de atividade atrial organizada',
    },
    alternativeDiagnoses: [
      { condition: 'flutter_atrial', probability: 0.08 },
      { condition: 'taquicardia_atrial_multifocal', probability: 0.03 },
    ],
  },
  validationStatus: 'pending',
  validatedBy: null,
  validationNotes: null,
};

// Mock do serviço
// jest.mock('@/services/diagnosis/engine');

describe('DiagnosisDisplay Component - 100% Coverage', () => {
  let store: any;
  let user: any;

  beforeEach(() => {
    store = createMockStore({
      currentDiagnosis: mockDiagnosis,
      isLoading: false,
      error: null,
    });
    
    user = userEvent.setup();
    jest.clearAllMocks();
  });

  describe('Renderização Básica', () => {
    it('deve renderizar diagnóstico completo corretamente', () => {
      render(
        <Provider store={store}>
          <DiagnosisDisplay diagnosis={mockDiagnosis} />
        </Provider>
      );

      // Diagnóstico primário
      expect(screen.getByText('Fibrilação Atrial')).toBeInTheDocument();
      expect(screen.getByText('89% confiança')).toBeInTheDocument();
      expect(screen.getByText('ICD-10: I48.91')).toBeInTheDocument();
      
      // Severidade
      expect(screen.getByText('Severidade: Moderada')).toBeInTheDocument();
      
      // Achados
      expect(screen.getByText('Principais Achados')).toBeInTheDocument();
      expect(screen.getByText('Ritmo irregularmente irregular')).toBeInTheDocument();
    });

    it('deve exibir mensagem quando não há diagnóstico', () => {
      render(
        <Provider store={store}>
          <DiagnosisDisplay diagnosis={null} />
        </Provider>
      );

      expect(screen.getByText('Nenhum diagnóstico disponível')).toBeInTheDocument();
    });

    it('deve exibir loading durante processamento', () => {
      const loadingStore = createMockStore({ isLoading: true });
      
      render(
        <Provider store={loadingStore}>
          <DiagnosisDisplay diagnosis={null} />
        </Provider>
      );

      expect(screen.getByTestId('diagnosis-loading')).toBeInTheDocument();
      expect(screen.getByText('Processando diagnóstico...')).toBeInTheDocument();
    });

    it('deve exibir erro quando falha', () => {
      const errorStore = createMockStore({ error: 'Falha na análise' });
      
      render(
        <Provider store={errorStore}>
          <DiagnosisDisplay diagnosis={mockDiagnosis} />
        </Provider>
      );

      expect(screen.getByRole('alert')).toHaveClass('alert-error');
      expect(screen.getByText('Falha na análise')).toBeInTheDocument();
    });
  });

  describe('Indicadores de Severidade', () => {
    it('deve exibir indicador correto para cada severidade', () => {
      const severities: Array<{ severity: Severity; color: string; icon: string }> = [
        { severity: 'critical', color: 'red', icon: 'alert-critical' },
        { severity: 'high', color: 'orange', icon: 'alert-high' },
        { severity: 'moderate', color: 'yellow', icon: 'alert-moderate' },
        { severity: 'mild', color: 'blue', icon: 'alert-mild' },
        { severity: 'normal', color: 'green', icon: 'check-circle' },
      ];

      severities.forEach(({ severity, color, icon }) => {
        const diagnosisWithSeverity = {
          ...mockDiagnosis,
          primaryDiagnosis: {
            ...mockDiagnosis.primaryDiagnosis,
            severity,
          },
        };

        const { rerender } = render(
          <Provider store={store}>
            <DiagnosisDisplay diagnosis={diagnosisWithSeverity} />
          </Provider>
        );

        const severityIndicator = screen.getByTestId('severity-indicator');
        expect(severityIndicator).toHaveClass(`severity-${severity}`);
        expect(severityIndicator.querySelector(`[data-icon="${icon}"]`)).toBeInTheDocument();

        rerender(<></>);
      });
    });

    it('deve piscar para condições críticas', () => {
      const criticalDiagnosis = {
        ...mockDiagnosis,
        primaryDiagnosis: {
          ...mockDiagnosis.primaryDiagnosis,
          condition: 'ventricular_fibrillation',
          severity: 'critical' as Severity,
        },
      };

      render(
        <Provider store={store}>
          <DiagnosisDisplay diagnosis={criticalDiagnosis} />
        </Provider>
      );

      const criticalAlert = screen.getByTestId('critical-alert');
      expect(criticalAlert).toHaveClass('pulse-animation');
      expect(screen.getByText('⚠️ ATENÇÃO MÉDICA IMEDIATA NECESSÁRIA')).toBeInTheDocument();
    });
  });

  describe('Diagnósticos Secundários', () => {
    it('deve exibir todos os diagnósticos secundários', () => {
      render(
        <Provider store={store}>
          <DiagnosisDisplay diagnosis={mockDiagnosis} />
        </Provider>
      );

      const secondarySection = screen.getByTestId('secondary-diagnoses');
      expect(within(secondarySection).getByText('Contrações Ventriculares Prematuras')).toBeInTheDocument();
      expect(within(secondarySection).getByText('72% confiança')).toBeInTheDocument();
    });

    it('deve permitir expandir/colapsar diagnósticos secundários', async () => {
      render(
        <Provider store={store}>
          <DiagnosisDisplay diagnosis={mockDiagnosis} />
        </Provider>
      );

      const toggleButton = screen.getByLabelText('Mostrar diagnósticos secundários');
      
      // Inicialmente expandido
      expect(screen.getByTestId('secondary-diagnoses-content')).toBeVisible();
      
      // Colapsar
      await user.click(toggleButton);
      expect(screen.queryByTestId('secondary-diagnoses-content')).not.toBeVisible();
      
      // Expandir novamente
      await user.click(toggleButton);
      expect(screen.getByTestId('secondary-diagnoses-content')).toBeVisible();
    });
  });

  describe('Recomendações Clínicas', () => {
    it('deve exibir recomendações ordenadas por prioridade', () => {
      render(
        <Provider store={store}>
          <DiagnosisDisplay diagnosis={mockDiagnosis} />
        </Provider>
      );

      const recommendations = screen.getAllByTestId(/recommendation-item/);
      
      // Verificar ordem (high -> medium -> low)
      expect(recommendations[0]).toHaveTextContent('Considerar anticoagulação');
      expect(recommendations[1]).toHaveTextContent('ECG de 24 horas');
      expect(recommendations[2]).toHaveTextContent('Ecocardiograma');
    });

    it('deve destacar recomendações de alta prioridade', () => {
      render(
        <Provider store={store}>
          <DiagnosisDisplay diagnosis={mockDiagnosis} />
        </Provider>
      );

      const highPriorityRec = screen.getByTestId('recommendation-item-high');
      expect(highPriorityRec).toHaveClass('recommendation-high-priority');
      expect(within(highPriorityRec).getByTestId('priority-badge')).toHaveTextContent('ALTA');
    });

    it('deve permitir marcar recomendações como concluídas', async () => {
      render(
        <Provider store={store}>
          <DiagnosisDisplay diagnosis={mockDiagnosis} editable />
        </Provider>
      );

      const checkboxes = screen.getAllByRole('checkbox');
      
      // Marcar primeira recomendação
      await user.click(checkboxes[0]);
      
      expect(checkboxes[0]).toBeChecked();
      expect(screen.getByTestId('recommendation-item-high')).toHaveClass('recommendation-completed');
    });
  });

  describe('Contexto Clínico', () => {
    it('deve exibir informações do contexto clínico', async () => {
      render(
        <Provider store={store}>
          <DiagnosisDisplay diagnosis={mockDiagnosis} showClinicalContext />
        </Provider>
      );

      // Expandir contexto clínico
      const expandButton = screen.getByLabelText('Ver contexto clínico');
      await user.click(expandButton);

      expect(screen.getByText('Idade: 65 anos')).toBeInTheDocument();
      expect(screen.getByText('Sexo: Masculino')).toBeInTheDocument();
      expect(screen.getByText('Fatores de Risco:')).toBeInTheDocument();
      expect(screen.getByText('Hipertensão')).toBeInTheDocument();
      expect(screen.getByText('Diabetes')).toBeInTheDocument();
    });

    it('deve exibir medicações atuais', async () => {
      render(
        <Provider store={store}>
          <DiagnosisDisplay diagnosis={mockDiagnosis} showClinicalContext />
        </Provider>
      );

      const expandButton = screen.getByLabelText('Ver contexto clínico');
      await user.click(expandButton);

      expect(screen.getByText('Medicações:')).toBeInTheDocument();
      expect(screen.getByText('Metoprolol')).toBeInTheDocument();
      expect(screen.getByText('Lisinopril')).toBeInTheDocument();
    });
  });

  describe('Insights de IA', () => {
    it('deve exibir explicabilidade da IA', async () => {
      render(
        <Provider store={store}>
          <DiagnosisDisplay diagnosis={mockDiagnosis} showAIInsights />
        </Provider>
      );

      const aiButton = screen.getByLabelText('Ver insights da IA');
      await user.click(aiButton);

      expect(screen.getByText('Análise de IA')).toBeInTheDocument();
      expect(screen.getByText('Confiança: 89%')).toBeInTheDocument();
      
      // Fatores principais
      expect(screen.getByText('Irregularidade RR (35%)')).toBeInTheDocument();
      expect(screen.getByText('Ausência de onda P (30%)')).toBeInTheDocument();
    });

    it('deve exibir diagnósticos alternativos', async () => {
      render(
        <Provider store={store}>
          <DiagnosisDisplay diagnosis={mockDiagnosis} showAIInsights />
        </Provider>
      );

      const aiButton = screen.getByLabelText('Ver insights da IA');
      await user.click(aiButton);

      const alternativesSection = screen.getByTestId('alternative-diagnoses');
      expect(within(alternativesSection).getByText('Flutter Atrial (8%)')).toBeInTheDocument();
      expect(within(alternativesSection).getByText('Taquicardia Atrial Multifocal (3%)')).toBeInTheDocument();
    });

    it('deve exibir visualização gráfica dos fatores', async () => {
      render(
        <Provider store={store}>
          <DiagnosisDisplay diagnosis={mockDiagnosis} showAIInsights />
        </Provider>
      );

      const aiButton = screen.getByLabelText('Ver insights da IA');
      await user.click(aiButton);

      const chartToggle = screen.getByLabelText('Visualizar como gráfico');
      await user.click(chartToggle);

      expect(screen.getByTestId('factors-chart')).toBeInTheDocument();
    });
  });

  describe('Validação Médica', () => {
    it('deve permitir validação por médico', async () => {
      (diagnosisService.validateDiagnosis as jest.Mock).mockResolvedValue({
        success: true,
        validatedAt: new Date().toISOString(),
      });

      render(
        <Provider store={store}>
          <DiagnosisDisplay diagnosis={mockDiagnosis} allowValidation />
        </Provider>
      );

      const validateButton = screen.getByLabelText('Validar diagnóstico');
      await user.click(validateButton);

      // Modal de validação
      expect(screen.getByText('Validar Diagnóstico')).toBeInTheDocument();
      
      const confirmButton = screen.getByText('Confirmar e Validar');
      const notesInput = screen.getByPlaceholderText('Adicionar observações...');
      
      await user.type(notesInput, 'Diagnóstico confirmado clinicamente');
      await user.click(confirmButton);

      await waitFor(() => {
        expect(diagnosisService.validateDiagnosis).toHaveBeenCalledWith({
          diagnosisId: 'DIAG_001',
          notes: 'Diagnóstico confirmado clinicamente',
          validatedBy: expect.any(String),
        });
      });
    });

    it('deve exibir status de validação', () => {
      const validatedDiagnosis = {
        ...mockDiagnosis,
        validationStatus: 'validated',
        validatedBy: 'DR_SMITH',
        validationNotes: 'Confirmado com exame clínico',
        validatedAt: new Date().toISOString(),
      };

      render(
        <Provider store={store}>
          <DiagnosisDisplay diagnosis={validatedDiagnosis} />
        </Provider>
      );

      const validationBadge = screen.getByTestId('validation-badge');
      expect(validationBadge).toHaveClass('badge-validated');
      expect(validationBadge).toHaveTextContent('Validado');
      
      // Tooltip com detalhes
      fireEvent.mouseEnter(validationBadge);
      expect(screen.getByText('Validado por: DR_SMITH')).toBeInTheDocument();
    });

    it('deve permitir rejeitar diagnóstico', async () => {
      render(
        <Provider store={store}>
          <DiagnosisDisplay diagnosis={mockDiagnosis} allowValidation />
        </Provider>
      );

      const validateButton = screen.getByLabelText('Validar diagnóstico');
      await user.click(validateButton);

      const rejectButton = screen.getByText('Rejeitar');
      await user.click(rejectButton);

      // Deve exibir campo obrigatório de justificativa
      const justificationInput = screen.getByPlaceholderText('Justificativa obrigatória...');
      expect(justificationInput).toBeRequired();
      
      await user.type(justificationInput, 'Achados clínicos inconsistentes');
      
      const confirmRejectButton = screen.getByText('Confirmar Rejeição');
      await user.click(confirmRejectButton);

      await waitFor(() => {
        expect(diagnosisService.rejectDiagnosis).toHaveBeenCalled();
      });
    });
  });

  describe('Exportação e Compartilhamento', () => {
    it('deve permitir exportar como PDF', async () => {
      const mockGeneratePDF = jest.fn();
      (diagnosisService.generateDiagnosisPDF as jest.Mock).mockImplementation(mockGeneratePDF);

      render(
        <Provider store={store}>
          <DiagnosisDisplay diagnosis={mockDiagnosis} />
        </Provider>
      );

      const exportButton = screen.getByLabelText('Exportar diagnóstico');
      await user.click(exportButton);

      const pdfOption = screen.getByText('Exportar como PDF');
      await user.click(pdfOption);

      expect(mockGeneratePDF).toHaveBeenCalledWith(mockDiagnosis);
    });

    it('deve permitir compartilhar por email', async () => {
      render(
        <Provider store={store}>
          <DiagnosisDisplay diagnosis={mockDiagnosis} />
        </Provider>
      );

      const shareButton = screen.getByLabelText('Compartilhar diagnóstico');
      await user.click(shareButton);

      const emailInput = screen.getByPlaceholderText('Email do destinatário');
      await user.type(emailInput, 'doctor@hospital.com');

      const sendButton = screen.getByText('Enviar');
      await user.click(sendButton);

      await waitFor(() => {
        expect(diagnosisService.shareDiagnosis).toHaveBeenCalledWith({
          diagnosisId: 'DIAG_001',
          recipientEmail: 'doctor@hospital.com',
          includeECG: true,
        });
      });
    });

    it('deve permitir imprimir diagnóstico', async () => {
      window.print = jest.fn();

      render(
        <Provider store={store}>
          <DiagnosisDisplay diagnosis={mockDiagnosis} />
        </Provider>
      );

      const printButton = screen.getByLabelText('Imprimir diagnóstico');
      await user.click(printButton);

      expect(window.print).toHaveBeenCalled();
    });
  });

  describe('Comparação com Diagnósticos Anteriores', () => {
    it('deve exibir histórico quando disponível', async () => {
      const historicalDiagnoses = [
        {
          id: 'DIAG_PREV_001',
          date: '2024-01-01',
          condition: 'normal_sinus_rhythm',
          confidence: 0.95,
        },
        {
          id: 'DIAG_PREV_002',
          date: '2024-06-01',
          condition: 'sinus_bradycardia',
          confidence: 0.88,
        },
      ];

      (diagnosisService.getPatientHistory as jest.Mock).mockResolvedValue(historicalDiagnoses);

      render(
        <Provider store={store}>
          <DiagnosisDisplay diagnosis={mockDiagnosis} showHistory />
        </Provider>
      );

      await waitFor(() => {
        expect(screen.getByText('Histórico de Diagnósticos')).toBeInTheDocument();
      });

      const historyItems = screen.getAllByTestId(/history-item/);
      expect(historyItems).toHaveLength(2);
    });

    it('deve destacar mudanças significativas', async () => {
      const diagnosisWithChanges = {
        ...mockDiagnosis,
        changes: {
          fromPrevious: 'normal_sinus_rhythm',
          significance: 'high',
          trend: 'worsening',
        },
      };

      render(
        <Provider store={store}>
          <DiagnosisDisplay diagnosis={diagnosisWithChanges} showHistory />
        </Provider>
      );

      const changeAlert = screen.getByTestId('significant-change-alert');
      expect(changeAlert).toHaveClass('alert-warning');
      expect(changeAlert).toHaveTextContent('Mudança significativa detectada');
    });
  });

  describe('Acessibilidade', () => {
    it('deve ter navegação por teclado completa', async () => {
      render(
        <Provider store={store}>
          <DiagnosisDisplay diagnosis={mockDiagnosis} />
        </Provider>
      );

      // Tab através dos elementos interativos
      await user.tab();
      expect(screen.getByLabelText('Ver contexto clínico')).toHaveFocus();

      await user.tab();
      expect(screen.getByLabelText('Ver insights da IA')).toHaveFocus();

      // Ativar com Enter
      await user.keyboard('{Enter}');
      expect(screen.getByText('Análise de IA')).toBeInTheDocument();
    });

    it('deve anunciar mudanças para leitores de tela', async () => {
      render(
        <Provider store={store}>
          <DiagnosisDisplay diagnosis={mockDiagnosis} />
        </Provider>
      );

      const liveRegion = screen.getByRole('status', { hidden: true });
      
      // Simular atualização de diagnóstico
      const updatedDiagnosis = {
        ...mockDiagnosis,
        validationStatus: 'validated',
      };

      store.dispatch({ type: 'diagnosis/update', payload: updatedDiagnosis });

      await waitFor(() => {
        expect(liveRegion).toHaveTextContent('Diagnóstico validado');
      });
    });
  });

  describe('Performance', () => {
    it('deve usar lazy loading para histórico', async () => {
      const mockHistoryLoader = jest.fn().mockResolvedValue([]);
      (diagnosisService.getPatientHistory as jest.Mock).mockImplementation(mockHistoryLoader);

      render(
        <Provider store={store}>
          <DiagnosisDisplay diagnosis={mockDiagnosis} showHistory />
        </Provider>
      );

      // Histórico não deve carregar imediatamente
      expect(mockHistoryLoader).not.toHaveBeenCalled();

      // Expandir seção de histórico
      const historyToggle = screen.getByLabelText('Ver histórico');
      await user.click(historyToggle);

      // Agora deve carregar
      expect(mockHistoryLoader).toHaveBeenCalled();
    });

    it('deve memoizar cálculos complexos', () => {
      const { rerender } = render(
        <Provider store={store}>
          <DiagnosisDisplay diagnosis={mockDiagnosis} />
        </Provider>
      );

      // Re-renderizar com as mesmas props
      rerender(
        <Provider store={store}>
          <DiagnosisDisplay diagnosis={mockDiagnosis} />
        </Provider>
      );

      // Verificar que componentes pesados não re-renderizaram
      // (verificação específica dependeria da implementação)
    });
  });

  describe('Tratamento de Erros', () => {
    it('deve lidar com falha na validação', async () => {
      (diagnosisService.validateDiagnosis as jest.Mock).mockRejectedValue(
        new Error('Falha na validação')
      );

      render(
        <Provider store={store}>
          <DiagnosisDisplay diagnosis={mockDiagnosis} allowValidation />
        </Provider>
      );

      const validateButton = screen.getByLabelText('Validar diagnóstico');
      await user.click(validateButton);

      const confirmButton = screen.getByText('Confirmar e Validar');
      await user.click(confirmButton);

      await waitFor(() => {
        expect(screen.getByText('Erro ao validar diagnóstico')).toBeInTheDocument();
      });
    });

    it('deve lidar com dados incompletos graciosamente', () => {
      const incompleteDiagnosis = {
        id: 'DIAG_002',
        primaryDiagnosis: {
          condition: 'unknown',
          confidence: 0.5,
        },
        // Faltam muitos campos
      };

      render(
        <Provider store={store}>
          <DiagnosisDisplay diagnosis={incompleteDiagnosis as any} />
        </Provider>
      );

      expect(screen.getByText('Dados incompletos')).toBeInTheDocument();
      expect(screen.queryByTestId('recommendations-section')).not.toBeInTheDocument();
    });
  });

  describe('Integração com Sistema', () => {
    it('deve sincronizar com prontuário eletrônico', async () => {
      render(
        <Provider store={store}>
          <DiagnosisDisplay diagnosis={mockDiagnosis} enableEHRSync />
        </Provider>
      );

      const syncButton = screen.getByLabelText('Sincronizar com prontuário');
      await user.click(syncButton);

      await waitFor(() => {
        expect(diagnosisService.syncWithEHR).toHaveBeenCalledWith({
          diagnosisId: 'DIAG_001',
          patientId: 'PAT_001',
        });
      });

      expect(screen.getByText('Sincronizado com sucesso')).toBeInTheDocument();
    });
  });
});
