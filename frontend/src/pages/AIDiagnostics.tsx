import React, { useState } from 'react'

interface DiagnosisItem {
  condition: string
  confidence: number
  type: string
}

interface DiagnosticResponse {
  diagnoses: DiagnosisItem[]
  treatment_plan: string
  confidence_score: number
  compliance_notice: string
}

const AIDiagnostics: React.FC = (): JSX.Element => {
  const [symptoms, setSymptoms] = useState('')
  const [vitalSigns, setVitalSigns] = useState({
    blood_pressure: '',
    heart_rate: '',
    temperature: '',
    respiratory_rate: '',
  })
  const [patientHistory, setPatientHistory] = useState('')
  const [result, setResult] = useState<DiagnosticResponse | null>(null)
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent): Promise<void> => {
    e.preventDefault()
    setLoading(true)

    try {
      const token = localStorage.getItem('token')
      const response = await fetch(`/api/v1/ai/diagnose`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symptoms,
          vital_signs: vitalSigns,
          patient_history: patientHistory,
        }),
      })

      if (response.ok) {
        const data = await response.json()
        setResult(data)
      }
    } catch (error) {
      console.error('Error getting diagnosis:', error)
    } finally {
      setLoading(false)
    }
  }

  const getConfidenceColor = (confidence: number): string => {
    if (confidence >= 0.8) return 'text-green-600'
    if (confidence >= 0.6) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getConfidenceBadge = (confidence: number): string => {
    if (confidence >= 0.8) return 'bg-green-100 text-green-800'
    if (confidence >= 0.6) return 'bg-yellow-100 text-yellow-800'
    return 'bg-red-100 text-red-800'
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">IA Diagnóstica</h1>
        <p className="text-gray-600">Assistência diagnóstica com inteligência artificial</p>
      </div>

      <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4">
        <div className="flex">
          <div className="flex-shrink-0">
            <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
              <path
                fillRule="evenodd"
                d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                clipRule="evenodd"
              />
            </svg>
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-yellow-800">Aviso Importante</h3>
            <div className="mt-2 text-sm text-yellow-700">
              <p>
                Esta ferramenta de IA é apenas para assistência diagnóstica. Sempre consulte um
                profissional médico qualificado para diagnóstico e tratamento definitivos.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white shadow rounded-lg p-6">
          <h2 className="text-lg font-medium text-gray-900 mb-4">Dados do Paciente</h2>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">Sintomas</label>
              <textarea
                required
                rows={4}
                className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2"
                placeholder="Descreva os sintomas apresentados pelo paciente..."
                value={symptoms}
                onChange={e => setSymptoms(e.target.value)}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Sinais Vitais</label>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs text-gray-500">Pressão Arterial</label>
                  <input
                    type="text"
                    className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2"
                    placeholder="120/80"
                    value={vitalSigns.blood_pressure}
                    onChange={e => setVitalSigns({ ...vitalSigns, blood_pressure: e.target.value })}
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500">Frequência Cardíaca</label>
                  <input
                    type="text"
                    className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2"
                    placeholder="72 bpm"
                    value={vitalSigns.heart_rate}
                    onChange={e => setVitalSigns({ ...vitalSigns, heart_rate: e.target.value })}
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500">Temperatura</label>
                  <input
                    type="text"
                    className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2"
                    placeholder="36.5°C"
                    value={vitalSigns.temperature}
                    onChange={e => setVitalSigns({ ...vitalSigns, temperature: e.target.value })}
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500">Freq. Respiratória</label>
                  <input
                    type="text"
                    className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2"
                    placeholder="16 rpm"
                    value={vitalSigns.respiratory_rate}
                    onChange={e =>
                      setVitalSigns({ ...vitalSigns, respiratory_rate: e.target.value })
                    }
                  />
                </div>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">
                Histórico do Paciente
              </label>
              <textarea
                rows={3}
                className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2"
                placeholder="Histórico médico relevante, medicações, alergias..."
                value={patientHistory}
                onChange={e => setPatientHistory(e.target.value)}
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md disabled:opacity-50"
            >
              {loading ? 'Analisando...' : 'Obter Diagnóstico IA'}
            </button>
          </form>
        </div>

        <div className="bg-white shadow rounded-lg p-6">
          <h2 className="text-lg font-medium text-gray-900 mb-4">Resultado da Análise</h2>

          {result ? (
            <div className="space-y-4">
              <div>
                <h3 className="text-md font-medium text-gray-900 mb-2">Hipóteses Diagnósticas</h3>
                <div className="space-y-2">
                  {result.diagnoses.map((diagnosis, index) => (
                    <div key={index} className="border border-gray-200 rounded-lg p-3">
                      <div className="flex justify-between items-start">
                        <div>
                          <h4 className="font-medium text-gray-900">{diagnosis.condition}</h4>
                          <p className="text-sm text-gray-500 capitalize">{diagnosis.type}</p>
                        </div>
                        <span
                          className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getConfidenceBadge(diagnosis.confidence)}`}
                        >
                          {Math.round(diagnosis.confidence * 100)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="text-md font-medium text-gray-900 mb-2">
                  Plano de Tratamento Sugerido
                </h3>
                <div className="bg-gray-50 rounded-lg p-3">
                  <p className="text-sm text-gray-700">{result.treatment_plan}</p>
                </div>
              </div>

              <div>
                <h3 className="text-md font-medium text-gray-900 mb-2">Confiança Geral</h3>
                <div className="flex items-center space-x-2">
                  <div className="flex-1 bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full"
                      style={{ width: `${result.confidence_score * 100}%` }}
                    ></div>
                  </div>
                  <span
                    className={`text-sm font-medium ${getConfidenceColor(result.confidence_score)}`}
                  >
                    {Math.round(result.confidence_score * 100)}%
                  </span>
                </div>
              </div>

              <div className="bg-red-50 border border-red-200 rounded-md p-3">
                <p className="text-sm text-red-700">{result.compliance_notice}</p>
              </div>
            </div>
          ) : (
            <div className="text-center text-gray-500 py-8">
              <svg
                className="mx-auto h-12 w-12 text-gray-400"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                />
              </svg>
              <p className="mt-2">Insira os dados do paciente para obter uma análise diagnóstica</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default AIDiagnostics
