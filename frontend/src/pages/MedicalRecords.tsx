import React, { useState, useEffect } from 'react'

interface MedicalRecord {
  id: number
  patient_id: number
  document_type: string
  chief_complaint: string
  history_present_illness: string
  assessment: string
  treatment_plan: string
  created_at: string
}

interface Patient {
  id: number
  name: string
  cpf: string
}

const MedicalRecords: React.FC = (): JSX.Element => {
  const [records, setRecords] = useState<MedicalRecord[]>([])
  const [patients, setPatients] = useState<Patient[]>([])
  const [showForm, setShowForm] = useState(false)
  const [formData, setFormData] = useState({
    patient_id: '',
    document_type: '',
    chief_complaint: '',
    history_present_illness: '',
    assessment: '',
    treatment_plan: '',
  })

  useEffect(() => {
    fetchRecords()
    fetchPatients()
  }, [])

  const fetchRecords = async (): Promise<void> => {
    try {
      const token = localStorage.getItem('token')
      const response = await fetch(`/api/v1/medical-records`, {
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      })
      if (response.ok) {
        const data = await response.json()
        setRecords(data)
      }
    } catch (error) {
      console.error('Error fetching records:', error)
    }
  }

  const fetchPatients = async (): Promise<void> => {
    try {
      const token = localStorage.getItem('token')
      const response = await fetch(`/api/v1/patients`, {
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      })
      if (response.ok) {
        const data = await response.json()
        setPatients(data)
      }
    } catch (error) {
      console.error('Error fetching patients:', error)
    }
  }

  const handleSubmit = async (e: React.FormEvent): Promise<void> => {
    e.preventDefault()
    try {
      const token = localStorage.getItem('token')
      const response = await fetch(`/api/v1/medical-records`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...formData,
          patient_id: parseInt(formData.patient_id),
        }),
      })

      if (response.ok) {
        setFormData({
          patient_id: '',
          document_type: '',
          chief_complaint: '',
          history_present_illness: '',
          assessment: '',
          treatment_plan: '',
        })
        setShowForm(false)
        fetchRecords()
      }
    } catch (error) {
      console.error('Error creating record:', error)
    }
  }

  const getPatientName = (patientId: number): string => {
    const patient = patients.find(p => p.id === patientId)
    return patient ? patient.name : 'Paciente não encontrado'
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900">Prontuários Eletrônicos</h1>
        <button
          onClick={() => setShowForm(true)}
          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md"
        >
          Novo Prontuário
        </button>
      </div>

      {showForm && (
        <div className="bg-white shadow rounded-lg p-6">
          <h2 className="text-lg font-medium text-gray-900 mb-4">Criar Novo Prontuário</h2>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">Paciente</label>
                <select
                  required
                  className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2"
                  value={formData.patient_id}
                  onChange={e => setFormData({ ...formData, patient_id: e.target.value })}
                >
                  <option value="">Selecione um paciente</option>
                  {patients.map(patient => (
                    <option key={patient.id} value={patient.id}>
                      {patient.name} - {patient.cpf}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Tipo de Documento</label>
                <select
                  required
                  className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2"
                  value={formData.document_type}
                  onChange={e => setFormData({ ...formData, document_type: e.target.value })}
                >
                  <option value="">Selecione o tipo</option>
                  <option value="anamnese">Anamnese</option>
                  <option value="evolução">Evolução</option>
                  <option value="admissão">Admissão</option>
                  <option value="alta">Alta</option>
                  <option value="cirúrgico">Cirúrgico</option>
                  <option value="consulta">Consulta</option>
                </select>
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Queixa Principal</label>
              <textarea
                required
                rows={3}
                className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2"
                value={formData.chief_complaint}
                onChange={e => setFormData({ ...formData, chief_complaint: e.target.value })}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">
                História da Doença Atual
              </label>
              <textarea
                required
                rows={4}
                className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2"
                value={formData.history_present_illness}
                onChange={e =>
                  setFormData({ ...formData, history_present_illness: e.target.value })
                }
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Avaliação</label>
              <textarea
                required
                rows={4}
                className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2"
                value={formData.assessment}
                onChange={e => setFormData({ ...formData, assessment: e.target.value })}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Plano de Tratamento</label>
              <textarea
                required
                rows={4}
                className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2"
                value={formData.treatment_plan}
                onChange={e => setFormData({ ...formData, treatment_plan: e.target.value })}
              />
            </div>
            <div className="flex space-x-4">
              <button
                type="submit"
                className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md"
              >
                Salvar
              </button>
              <button
                type="button"
                onClick={() => setShowForm(false)}
                className="bg-gray-300 hover:bg-gray-400 text-gray-700 px-4 py-2 rounded-md"
              >
                Cancelar
              </button>
            </div>
          </form>
        </div>
      )}

      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">Lista de Prontuários</h3>
          <div className="space-y-4">
            {records.map(record => (
              <div key={record.id} className="border border-gray-200 rounded-lg p-4">
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <h4 className="text-lg font-medium text-gray-900">
                      {getPatientName(record.patient_id)}
                    </h4>
                    <p className="text-sm text-gray-500">
                      {record.document_type} -{' '}
                      {new Date(record.created_at).toLocaleDateString('pt-BR')}
                    </p>
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <strong>Queixa Principal:</strong>
                    <p className="text-gray-700">{record.chief_complaint}</p>
                  </div>
                  <div>
                    <strong>História:</strong>
                    <p className="text-gray-700">{record.history_present_illness}</p>
                  </div>
                  <div>
                    <strong>Avaliação:</strong>
                    <p className="text-gray-700">{record.assessment}</p>
                  </div>
                  <div>
                    <strong>Plano:</strong>
                    <p className="text-gray-700">{record.treatment_plan}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export default MedicalRecords
