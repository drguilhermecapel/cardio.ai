import React, { useState, useEffect } from 'react'

interface Consultation {
  id: number
  patient_id: number
  consultation_type: string
  scheduled_date: string
  status: string
  notes?: string
  created_at: string
}

interface Patient {
  id: number
  name: string
  cpf: string
}

const Telemedicine: React.FC = (): JSX.Element => {
  const [consultations, setConsultations] = useState<Consultation[]>([])
  const [patients, setPatients] = useState<Patient[]>([])
  const [showForm, setShowForm] = useState(false)
  const [formData, setFormData] = useState({
    patient_id: '',
    consultation_type: '',
    scheduled_date: '',
  })

  useEffect(() => {
    fetchConsultations()
    fetchPatients()
  }, [])

  const fetchConsultations = async (): Promise<void> => {
    try {
      const token = localStorage.getItem('token')
      const response = await fetch(`/api/v1/consultations`, {
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      })
      if (response.ok) {
        const data = await response.json()
        setConsultations(data)
      }
    } catch (error) {
      console.error('Error fetching consultations:', error)
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
      const response = await fetch(`/api/v1/consultations`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...formData,
          patient_id: parseInt(formData.patient_id),
          scheduled_date: new Date(formData.scheduled_date).toISOString(),
        }),
      })

      if (response.ok) {
        setFormData({
          patient_id: '',
          consultation_type: '',
          scheduled_date: '',
        })
        setShowForm(false)
        fetchConsultations()
      }
    } catch (error) {
      console.error('Error creating consultation:', error)
    }
  }

  const getPatientName = (patientId: number): string => {
    const patient = patients.find(p => p.id === patientId)
    return patient ? patient.name : 'Paciente não encontrado'
  }

  const getStatusBadge = (status: string): JSX.Element => {
    const statusMap = {
      scheduled: { color: 'bg-blue-100 text-blue-800', text: 'Agendada' },
      in_progress: { color: 'bg-yellow-100 text-yellow-800', text: 'Em Andamento' },
      completed: { color: 'bg-green-100 text-green-800', text: 'Concluída' },
      cancelled: { color: 'bg-red-100 text-red-800', text: 'Cancelada' },
    }
    const statusInfo = statusMap[status as keyof typeof statusMap] || statusMap.scheduled
    return (
      <span
        className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${statusInfo.color}`}
      >
        {statusInfo.text}
      </span>
    )
  }

  const getTypeIcon = (type: string): JSX.Element | null => {
    switch (type) {
      case 'video':
        return (
          <svg
            className="h-5 w-5 text-blue-500"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
            />
          </svg>
        )
      case 'audio':
        return (
          <svg
            className="h-5 w-5 text-green-500"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z"
            />
          </svg>
        )
      case 'chat':
        return (
          <svg
            className="h-5 w-5 text-purple-500"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
            />
          </svg>
        )
      default:
        return null
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900">Telemedicina</h1>
        <button
          onClick={() => setShowForm(true)}
          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md"
        >
          Nova Consulta
        </button>
      </div>

      {showForm && (
        <div className="bg-white shadow rounded-lg p-6">
          <h2 className="text-lg font-medium text-gray-900 mb-4">Agendar Nova Consulta</h2>
          <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-3 gap-4">
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
              <label className="block text-sm font-medium text-gray-700">Tipo de Consulta</label>
              <select
                required
                className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2"
                value={formData.consultation_type}
                onChange={e => setFormData({ ...formData, consultation_type: e.target.value })}
              >
                <option value="">Selecione o tipo</option>
                <option value="video">Videochamada</option>
                <option value="audio">Chamada de Áudio</option>
                <option value="chat">Chat</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Data e Hora</label>
              <input
                type="datetime-local"
                required
                className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2"
                value={formData.scheduled_date}
                onChange={e => setFormData({ ...formData, scheduled_date: e.target.value })}
              />
            </div>
            <div className="md:col-span-3 flex space-x-4">
              <button
                type="submit"
                className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md"
              >
                Agendar
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
          <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">Consultas Agendadas</h3>
          <div className="space-y-4">
            {consultations.map(consultation => (
              <div key={consultation.id} className="border border-gray-200 rounded-lg p-4">
                <div className="flex justify-between items-start">
                  <div className="flex items-center space-x-3">
                    {getTypeIcon(consultation.consultation_type)}
                    <div>
                      <h4 className="text-lg font-medium text-gray-900">
                        {getPatientName(consultation.patient_id)}
                      </h4>
                      <p className="text-sm text-gray-500">
                        {new Date(consultation.scheduled_date).toLocaleString('pt-BR')}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    {getStatusBadge(consultation.status)}
                    <span className="text-sm text-gray-500 capitalize">
                      {consultation.consultation_type}
                    </span>
                  </div>
                </div>
                {consultation.notes && (
                  <div className="mt-3 text-sm text-gray-700">
                    <strong>Observações:</strong> {consultation.notes}
                  </div>
                )}
                {consultation.status === 'scheduled' && (
                  <div className="mt-3 flex space-x-2">
                    <button className="bg-green-600 hover:bg-green-700 text-white px-3 py-1 rounded text-sm">
                      Iniciar Consulta
                    </button>
                    <button className="bg-red-600 hover:bg-red-700 text-white px-3 py-1 rounded text-sm">
                      Cancelar
                    </button>
                  </div>
                )}
              </div>
            ))}
            {consultations.length === 0 && (
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
                    d="M8 7V3a2 2 0 012-2h4a2 2 0 012 2v4m-6 9l6-6m0 0l6 6m-6-6v9a6 6 0 01-12 0v-9a6 6 0 016-6z"
                  />
                </svg>
                <p className="mt-2">Nenhuma consulta agendada</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default Telemedicine
