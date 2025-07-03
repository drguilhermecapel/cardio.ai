// frontend/src/components/medical/__tests__/ECGVisualization.test.tsx
// Testes completos para o componente ECGVisualization - 100% cobertura

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import ECGVisualization from '../ECGVisualization';
import { ECGData, ECGLead } from '@/types/medical';
import * as ecgService from '@/services/ecg/analysis';

// Mock do Redux store
const createMockStore = (initialState = {}) => {
  return configureStore({
    reducer: {
      ecg: (state = initialState) => state,
    },
  });
};

// Mock de dados ECG
const mockECGData: ECGData = {
  signal: Array(3600).fill(0).map((_, i) => 
    Math.sin(2 * Math.PI * 1.2 * i / 360) + (Math.random() - 0.5) * 0.1
  ),
  samplingRate: 360,
  duration: 10,
  leadConfiguration: 'II' as ECGLead,
  timestamp: new Date().toISOString(),
  patientId: 'TEST_001',
};

// Mock do Canvas Context
const mockCanvasContext = {
  clearRect: jest.fn(),
  beginPath: jest.fn(),
  moveTo: jest.fn(),
  lineTo: jest.fn(),
  stroke: jest.fn(),
  fillText: jest.fn(),
  measureText: jest.fn(() => ({ width: 50 })),
  save: jest.fn(),
  restore: jest.fn(),
  translate: jest.fn(),
  scale: jest.fn(),
  setTransform: jest.fn(),
  fillRect: jest.fn(),
  strokeRect: jest.fn(),
  canvas: { width: 800, height: 400 },
};

// Mock do serviço de análise
jest.mock('@/services/ecg/analysis');

describe('ECGVisualization Component - 100% Coverage', () => {
  let store: any;
  let user: any;
  
  beforeEach(() => {
    store = createMockStore({
      currentECG: mockECGData,
      isLoading: false,
      error: null,
    });
    
    user = userEvent.setup();
    
    // Mock do canvas getContext
    HTMLCanvasElement.prototype.getContext = jest.fn(() => mockCanvasContext);
    
    // Mock do requestAnimationFrame
    global.requestAnimationFrame = jest.fn(cb => setTimeout(cb, 0));
    global.cancelAnimationFrame = jest.fn();
    
    // Reset todos os mocks
    jest.clearAllMocks();
  });
  
  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('Renderização Básica', () => {
    it('deve renderizar o componente corretamente', () => {
      render(
        <Provider store={store}>
          <ECGVisualization data={mockECGData} />
        </Provider>
      );
      
      expect(screen.getByTestId('ecg-visualization')).toBeInTheDocument();
      expect(screen.getByTestId('ecg-canvas')).toBeInTheDocument();
    });
    
    it('deve exibir mensagem quando não há dados', () => {
      render(
        <Provider store={store}>
          <ECGVisualization data={null} />
        </Provider>
      );
      
      expect(screen.getByText('Nenhum dado ECG disponível')).toBeInTheDocument();
    });
    
    it('deve exibir loading quando está carregando', () => {
      const loadingStore = createMockStore({ isLoading: true });
      
      render(
        <Provider store={loadingStore}>
          <ECGVisualization data={null} />
        </Provider>
      );
      
      expect(screen.getByTestId('ecg-loading')).toBeInTheDocument();
      expect(screen.getByText('Carregando ECG...')).toBeInTheDocument();
    });
    
    it('deve exibir erro quando há falha', () => {
      const errorStore = createMockStore({ error: 'Falha ao carregar ECG' });
      
      render(
        <Provider store={errorStore}>
          <ECGVisualization data={mockECGData} />
        </Provider>
      );
      
      expect(screen.getByRole('alert')).toBeInTheDocument();
      expect(screen.getByText('Falha ao carregar ECG')).toBeInTheDocument();
    });
  });

  describe('Renderização do Canvas', () => {
    it('deve desenhar o sinal ECG no canvas', async () => {
      render(
        <Provider store={store}>
          <ECGVisualization data={mockECGData} />
        </Provider>
      );
      
      await waitFor(() => {
        expect(mockCanvasContext.clearRect).toHaveBeenCalled();
        expect(mockCanvasContext.beginPath).toHaveBeenCalled();
        expect(mockCanvasContext.stroke).toHaveBeenCalled();
      });
    });
    
    it('deve aplicar zoom corretamente', async () => {
      render(
        <Provider store={store}>
          <ECGVisualization data={mockECGData} />
        </Provider>
      );
      
      const zoomInButton = screen.getByLabelText('Zoom In');
      await user.click(zoomInButton);
      
      await waitFor(() => {
        expect(mockCanvasContext.scale).toHaveBeenCalledWith(expect.any(Number), 1);
      });
    });
    
    it('deve aplicar pan (deslocamento) corretamente', async () => {
      render(
        <Provider store={store}>
          <ECGVisualization data={mockECGData} />
        </Provider>
      );
      
      const canvas = screen.getByTestId('ecg-canvas');
      
      // Simular drag do mouse
      fireEvent.mouseDown(canvas, { clientX: 100, clientY: 100 });
      fireEvent.mouseMove(canvas, { clientX: 150, clientY: 100 });
      fireEvent.mouseUp(canvas);
      
      await waitFor(() => {
        expect(mockCanvasContext.translate).toHaveBeenCalled();
      });
    });
  });

  describe('Controles de Visualização', () => {
    it('deve alternar entre diferentes derivações', async () => {
      const multiLeadData = {
        ...mockECGData,
        leads: {
          I: mockECGData.signal,
          II: mockECGData.signal.map(v => v * 1.1),
          III: mockECGData.signal.map(v => v * 0.9),
        },
      };
      
      render(
        <Provider store={store}>
          <ECGVisualization data={multiLeadData} />
        </Provider>
      );
      
      const leadSelector = screen.getByLabelText('Selecionar Derivação');
      await user.selectOptions(leadSelector, 'III');
      
      await waitFor(() => {
        expect(mockCanvasContext.clearRect).toHaveBeenCalled();
      });
    });
    
    it('deve alternar grade de fundo', async () => {
      render(
        <Provider store={store}>
          <ECGVisualization data={mockECGData} />
        </Provider>
      );
      
      const gridToggle = screen.getByLabelText('Mostrar Grade');
      await user.click(gridToggle);
      
      await waitFor(() => {
        expect(mockCanvasContext.strokeRect).toHaveBeenCalled();
      });
    });
    
    it('deve ajustar velocidade de varredura', async () => {
      render(
        <Provider store={store}>
          <ECGVisualization data={mockECGData} />
        </Provider>
      );
      
      const speedControl = screen.getByLabelText('Velocidade (mm/s)');
      fireEvent.change(speedControl, { target: { value: '50' } });
      
      await waitFor(() => {
        expect(mockCanvasContext.clearRect).toHaveBeenCalled();
      });
    });
    
    it('deve ajustar ganho (amplitude)', async () => {
      render(
        <Provider store={store}>
          <ECGVisualization data={mockECGData} />
        </Provider>
      );
      
      const gainControl = screen.getByLabelText('Ganho (mm/mV)');
      fireEvent.change(gainControl, { target: { value: '20' } });
      
      await waitFor(() => {
        expect(mockCanvasContext.clearRect).toHaveBeenCalled();
      });
    });
  });

  describe('Análise e Medições', () => {
    it('deve permitir medição de intervalos', async () => {
      render(
        <Provider store={store}>
          <ECGVisualization data={mockECGData} enableMeasurements />
        </Provider>
      );
      
      const measureButton = screen.getByLabelText('Medir Intervalo');
      await user.click(measureButton);
      
      const canvas = screen.getByTestId('ecg-canvas');
      
      // Primeiro clique para iniciar medição
      fireEvent.click(canvas, { clientX: 100, clientY: 200 });
      
      // Segundo clique para finalizar medição
      fireEvent.click(canvas, { clientX: 200, clientY: 200 });
      
      await waitFor(() => {
        expect(screen.getByText(/Intervalo:/)).toBeInTheDocument();
      });
    });
    
    it('deve detectar e marcar picos R', async () => {
      (ecgService.detectRPeaks as jest.Mock).mockResolvedValue({
        peaks: [100, 460, 820],
        intervals: [360, 360],
        heartRate: 72,
      });
      
      render(
        <Provider store={store}>
          <ECGVisualization data={mockECGData} showRPeaks />
        </Provider>
      );
      
      await waitFor(() => {
        expect(ecgService.detectRPeaks).toHaveBeenCalledWith(
          mockECGData.signal,
          mockECGData.samplingRate
        );
        expect(screen.getByText('FC: 72 bpm')).toBeInTheDocument();
      });
    });
    
    it('deve calcular e exibir HRV', async () => {
      (ecgService.calculateHRV as jest.Mock).mockResolvedValue({
        sdnn: 45.2,
        rmssd: 38.5,
        pnn50: 15.3,
      });
      
      render(
        <Provider store={store}>
          <ECGVisualization data={mockECGData} showHRV />
        </Provider>
      );
      
      await waitFor(() => {
        expect(ecgService.calculateHRV).toHaveBeenCalled();
        expect(screen.getByText(/SDNN: 45.2 ms/)).toBeInTheDocument();
      });
    });
  });

  describe('Modo Tempo Real', () => {
    it('deve atualizar visualização em tempo real', async () => {
      const { rerender } = render(
        <Provider store={store}>
          <ECGVisualization data={mockECGData} realTime />
        </Provider>
      );
      
      // Simular atualização de dados
      const newData = {
        ...mockECGData,
        signal: [...mockECGData.signal, ...Array(360).fill(0).map(() => Math.random())],
      };
      
      await act(async () => {
        rerender(
          <Provider store={store}>
            <ECGVisualization data={newData} realTime />
          </Provider>
        );
      });
      
      expect(mockCanvasContext.clearRect).toHaveBeenCalledTimes(2);
    });
    
    it('deve pausar/resumir streaming', async () => {
      render(
        <Provider store={store}>
          <ECGVisualization data={mockECGData} realTime />
        </Provider>
      );
      
      const pauseButton = screen.getByLabelText('Pausar');
      await user.click(pauseButton);
      
      expect(screen.getByLabelText('Resumir')).toBeInTheDocument();
      
      await user.click(screen.getByLabelText('Resumir'));
      
      expect(screen.getByLabelText('Pausar')).toBeInTheDocument();
    });
  });

  describe('Exportação e Compartilhamento', () => {
    it('deve exportar como imagem PNG', async () => {
      const mockToBlob = jest.fn((callback) => {
        callback(new Blob(['mock'], { type: 'image/png' }));
      });
      
      HTMLCanvasElement.prototype.toBlob = mockToBlob;
      
      render(
        <Provider store={store}>
          <ECGVisualization data={mockECGData} />
        </Provider>
      );
      
      const exportButton = screen.getByLabelText('Exportar como Imagem');
      await user.click(exportButton);
      
      expect(mockToBlob).toHaveBeenCalled();
      expect(global.URL.createObjectURL).toHaveBeenCalled();
    });
    
    it('deve exportar dados como CSV', async () => {
      render(
        <Provider store={store}>
          <ECGVisualization data={mockECGData} />
        </Provider>
      );
      
      const exportDataButton = screen.getByLabelText('Exportar Dados');
      await user.click(exportDataButton);
      
      const csvOption = screen.getByText('CSV');
      await user.click(csvOption);
      
      expect(global.URL.createObjectURL).toHaveBeenCalled();
    });
    
    it('deve imprimir ECG', async () => {
      window.print = jest.fn();
      
      render(
        <Provider store={store}>
          <ECGVisualization data={mockECGData} />
        </Provider>
      );
      
      const printButton = screen.getByLabelText('Imprimir');
      await user.click(printButton);
      
      expect(window.print).toHaveBeenCalled();
    });
  });

  describe('Acessibilidade', () => {
    it('deve ter navegação por teclado completa', async () => {
      render(
        <Provider store={store}>
          <ECGVisualization data={mockECGData} />
        </Provider>
      );
      
      const canvas = screen.getByTestId('ecg-canvas');
      canvas.focus();
      
      // Zoom com teclado
      fireEvent.keyDown(canvas, { key: '+' });
      await waitFor(() => expect(mockCanvasContext.scale).toHaveBeenCalled());
      
      // Pan com setas
      fireEvent.keyDown(canvas, { key: 'ArrowRight' });
      await waitFor(() => expect(mockCanvasContext.translate).toHaveBeenCalled());
    });
    
    it('deve ter descrições ARIA apropriadas', () => {
      render(
        <Provider store={store}>
          <ECGVisualization data={mockECGData} />
        </Provider>
      );
      
      expect(screen.getByRole('img', { name: /ECG Visualization/i })).toBeInTheDocument();
      expect(screen.getByLabelText(/Heart rate: \d+ bpm/)).toBeInTheDocument();
    });
  });

  describe('Tratamento de Erros', () => {
    it('deve lidar com falha na detecção de picos R', async () => {
      (ecgService.detectRPeaks as jest.Mock).mockRejectedValue(new Error('Detection failed'));
      
      render(
        <Provider store={store}>
          <ECGVisualization data={mockECGData} showRPeaks />
        </Provider>
      );
      
      await waitFor(() => {
        expect(screen.getByText('Falha na detecção de picos R')).toBeInTheDocument();
      });
    });
    
    it('deve lidar com dados ECG inválidos', () => {
      const invalidData = {
        ...mockECGData,
        signal: null,
      };
      
      render(
        <Provider store={store}>
          <ECGVisualization data={invalidData as any} />
        </Provider>
      );
      
      expect(screen.getByText('Dados ECG inválidos')).toBeInTheDocument();
    });
    
    it('deve lidar com falha no contexto do canvas', () => {
      HTMLCanvasElement.prototype.getContext = jest.fn(() => null);
      
      render(
        <Provider store={store}>
          <ECGVisualization data={mockECGData} />
        </Provider>
      );
      
      expect(screen.getByText('Canvas não suportado')).toBeInTheDocument();
    });
  });

  describe('Performance', () => {
    it('deve otimizar renderização para grandes datasets', async () => {
      const largeData = {
        ...mockECGData,
        signal: Array(36000).fill(0).map(() => Math.random()), // 100 segundos
      };
      
      render(
        <Provider store={store}>
          <ECGVisualization data={largeData} />
        </Provider>
      );
      
      // Verificar se usa técnica de decimação
      await waitFor(() => {
        const calls = mockCanvasContext.lineTo.mock.calls.length;
        expect(calls).toBeLessThan(largeData.signal.length);
      });
    });
    
    it('deve cancelar animações ao desmontar', () => {
      const { unmount } = render(
        <Provider store={store}>
          <ECGVisualization data={mockECGData} realTime />
        </Provider>
      );
      
      unmount();
      
      expect(global.cancelAnimationFrame).toHaveBeenCalled();
    });
  });

  describe('Integração com Redux', () => {
    it('deve despachar ações ao interagir', async () => {
      const mockDispatch = jest.fn();
      store.dispatch = mockDispatch;
      
      render(
        <Provider store={store}>
          <ECGVisualization data={mockECGData} />
        </Provider>
      );
      
      const annotateButton = screen.getByLabelText('Adicionar Anotação');
      await user.click(annotateButton);
      
      expect(mockDispatch).toHaveBeenCalledWith(
        expect.objectContaining({ type: expect.stringContaining('annotation') })
      );
    });
  });
});
