// frontend/src/components/validation/ValidationPanel.tsx
import React, { useState, useEffect } from 'react'
import { 
  Card, 
  CardContent, 
  Typography, 
  Button, 
  TextField,
  FormControl,
  FormLabel,
  RadioGroup,
  FormControlLabel,
  Radio,
  Alert,
  Box,
  Chip,
  CircularProgress,
  Snackbar
} from '../ui/BasicComponents'

interface DiagnosticInfo {
  class: string
  probability: number
}

interface ValidationData {
  ecg_id: string
  ai_diagnosis: string
  ai_probability: number
  validator_id: string
  is_correct: boolean
  corrected_diagnosis?: string
  comments?: string
  model_version?: string
  confidence_score?: number
  processing_time_ms?: number
}

interface ValidationPanelProps {
  ecgId: string
  diagnostics: DiagnosticInfo[]
  validatorId: string
  modelVersion?: string
  confidenceScore?: number
  processingTime?: number
  onValidationSubmit?: (validation: ValidationData) => void
  className?: string
}

export const ValidationPanel: React.FC<ValidationPanelProps> = ({
  ecgId,
  diagnostics,
  validatorId,
  modelVersion,
  confidenceScore,
  processingTime,
  onValidationSubmit,
  className = ''
}) => {
  const [selectedDiagnosis, setSelectedDiagnosis] = useState<DiagnosticInfo | null>(null)
  const [isCorrect, setIsCorrect] = useState<string>('')
  const [correctedDiagnosis, setCorrectedDiagnosis] = useState('')
  const [comments, setComments] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [showSuccess, setShowSuccess] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Selecionar automaticamente o diagnóstico com maior probabilidade
  useEffect(() => {
    if (diagnostics && diagnostics.length > 0) {
      const topDiagnosis = diagnostics.reduce((prev, current) => 
        (prev.probability > current.probability) ? prev : current
      )
      setSelectedDiagnosis(topDiagnosis)
    }
  }, [diagnostics])

  const handleSubmitValidation = async () => {
    if (!selectedDiagnosis || !isCorrect) {
      setError('Por favor, selecione um diagnóstico e indique se está correto')
      return
    }

    if (isCorrect === 'false' && !correctedDiagnosis.trim()) {
      setError('Por favor, forneça o diagnóstico correto')
      return
    }

    setIsSubmitting(true)
    setError(null)

    try {
      const validationData: ValidationData = {
        ecg_id: ecgId,
        ai_diagnosis: selectedDiagnosis.class,
        ai_probability: selectedDiagnosis.probability,
        validator_id: validatorId,
        is_correct: isCorrect === 'true',
        corrected_diagnosis: isCorrect === 'false' ? correctedDiagnosis : undefined,
        comments: comments.trim() || undefined,
        model_version: modelVersion,
        confidence_score: confidenceScore,
        processing_time_ms: processingTime
      }

      // Enviar para API
      const response = await fetch('/api/v1/ecg/validate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(validationData)
      })

      if (!response.ok) {
        throw new Error(`Erro HTTP: ${response.status}`)
      }

      const result = await response.json()

      if (result.success) {
        setShowSuccess(true)
        
        // Limpar formulário
        setIsCorrect('')
        setCorrectedDiagnosis('')
        setComments('')
        
        // Callback para componente pai
        onValidationSubmit?.(validationData)
      } else {
        throw new Error(result.message || 'Erro ao salvar validação')
      }

    } catch (err) {
      console.error('Erro ao enviar validação:', err)
      setError(err instanceof Error ? err.message : 'Erro desconhecido')
    } finally {
      setIsSubmitting(false)
    }
  }

  const getConfidenceColor = (probability: number) => {
    if (probability >= 0.9) return 'success'
    if (probability >= 0.7) return 'warning'
    return 'error'
  }

  const getConfidenceLabel = (probability: number) => {
    if (probability >= 0.9) return 'Alta Confiança'
    if (probability >= 0.7) return 'Confiança Moderada'
    return 'Baixa Confiança'
  }

  return (
    <div className={`space-y-6 ${className}`}>
      
      {/* Header */}
      <Card variant="medical">
        <CardContent className="p-6">
          <Typography variant="h5" className="font-bold text-blue-700 mb-2">
            🩺 Validação Clínica
          </Typography>
          <Typography variant="body2" className="text-blue-600">
            Valide o diagnóstico da IA para melhorar a precisão do modelo
          </Typography>
          
          {/* ECG Info */}
          <Box className="mt-4 p-3 bg-blue-50 rounded-lg">
            <Typography variant="body2" className="text-blue-800">
              <strong>ECG ID:</strong> {ecgId}
            </Typography>
            {modelVersion && (
              <Typography variant="body2" className="text-blue-800">
                <strong>Modelo:</strong> {modelVersion}
              </Typography>
            )}
            {validatorId && (
              <Typography variant="body2" className="text-blue-800">
                <strong>Validador:</strong> {validatorId}
              </Typography>
            )}
          </Box>
        </CardContent>
      </Card>

      {/* Diagnósticos da IA */}
      <Card>
        <CardContent className="p-6">
          <Typography variant="h6" className="font-bold text-gray-900 mb-4">
            📊 Diagnósticos da IA
          </Typography>
          
          <div className="space-y-3">
            {diagnostics.map((diag, index) => (
              <Box 
                key={index} 
                className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                  selectedDiagnosis?.class === diag.class 
                    ? 'border-blue-500 bg-blue-50' 
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => setSelectedDiagnosis(diag)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <input
                      type="radio"
                      checked={selectedDiagnosis?.class === diag.class}
                      onChange={() => setSelectedDiagnosis(diag)}
                      className="text-blue-600"
                    />
                    <div>
                      <Typography variant="h6" className="font-bold text-gray-800">
                        {diag.class}
                      </Typography>
                      <Typography variant="body2" className="text-gray-600">
                        Probabilidade: {(diag.probability * 100).toFixed(1)}%
                      </Typography>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <Chip 
                      label={getConfidenceLabel(diag.probability)}
                      color={getConfidenceColor(diag.probability)}
                      size="small"
                    />
                    <Typography variant="h5" className="font-bold text-blue-600">
                      {(diag.probability * 100).toFixed(1)}%
                    </Typography>
                  </div>
                </div>
              </Box>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Formulário de Validação */}
      {selectedDiagnosis && (
        <Card>
          <CardContent className="p-6">
            <Typography variant="h6" className="font-bold text-gray-900 mb-4">
              ✅ Validação do Diagnóstico: {selectedDiagnosis.class}
            </Typography>
            
            {/* Pergunta principal */}
            <FormControl component="fieldset" className="mb-6">
              <FormLabel component="legend" className="text-gray-700 font-medium mb-3">
                O diagnóstico da IA está correto?
              </FormLabel>
              <RadioGroup
                value={isCorrect}
                onChange={(e) => setIsCorrect(e.target.value)}
                className="space-y-2"
              >
                <FormControlLabel 
                  value="true" 
                  control={<Radio color="primary" />} 
                  label="✅ Sim, o diagnóstico está correto"
                  className="text-green-700"
                />
                <FormControlLabel 
                  value="false" 
                  control={<Radio color="primary" />} 
                  label="❌ Não, o diagnóstico está incorreto"
                  className="text-red-700"
                />
              </RadioGroup>
            </FormControl>

            {/* Diagnóstico correto (se incorreto) */}
            {isCorrect === 'false' && (
              <Box className="mb-6">
                <TextField
                  fullWidth
                  label="Diagnóstico Correto"
                  placeholder="Digite o diagnóstico correto..."
                  value={correctedDiagnosis}
                  onChange={(e) => setCorrectedDiagnosis(e.target.value)}
                  required
                  variant="outlined"
                  className="mb-3"
                />
                <Typography variant="caption" className="text-gray-600">
                  Por favor, forneça o diagnóstico correto para este ECG
                </Typography>
              </Box>
            )}

            {/* Comentários */}
            <Box className="mb-6">
              <TextField
                fullWidth
                multiline
                rows={3}
                label="Comentários (Opcional)"
                placeholder="Adicione observações, justificativas ou comentários sobre este diagnóstico..."
                value={comments}
                onChange={(e) => setComments(e.target.value)}
                variant="outlined"
              />
              <Typography variant="caption" className="text-gray-600">
                Seus comentários ajudam a melhorar o modelo de IA
              </Typography>
            </Box>

            {/* Erro */}
            {error && (
              <Alert severity="error" className="mb-4">
                {error}
              </Alert>
            )}

            {/* Botões */}
            <div className="flex space-x-3">
              <Button
                variant="contained"
                color="primary"
                onClick={handleSubmitValidation}
                disabled={isSubmitting || !isCorrect}
                className="flex items-center space-x-2"
              >
                {isSubmitting ? (
                  <>
                    <CircularProgress size={20} />
                    <span>Enviando...</span>
                  </>
                ) : (
                  <>
                    <span>💾</span>
                    <span>Enviar Validação</span>
                  </>
                )}
              </Button>
              
              <Button
                variant="outlined"
                color="secondary"
                onClick={() => {
                  setIsCorrect('')
                  setCorrectedDiagnosis('')
                  setComments('')
                  setError(null)
                }}
                disabled={isSubmitting}
              >
                🔄 Limpar
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Snackbar de sucesso */}
      <Snackbar
        open={showSuccess}
        autoHideDuration={6000}
        onClose={() => setShowSuccess(false)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert severity="success" onClose={() => setShowSuccess(false)}>
          ✅ Validação enviada com sucesso! Obrigado pelo feedback.
        </Alert>
      </Snackbar>
    </div>
  )
}

export default ValidationPanel

