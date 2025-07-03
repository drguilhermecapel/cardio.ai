import React, { useState, useEffect } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  Grid,
  Alert,
} from '../components/ui/BasicComponents'
import { useAppDispatch, useAppSelector } from '../hooks/redux'
import { fetchPatients, createPatient, clearError } from '../store/slices/patientSlice'

interface PatientFormData {
  patientId: string
  firstName: string
  lastName: string
  dateOfBirth: string
  gender: string
  phone: string
  email: string
}

const PatientsPage: React.FC = () => {
  const [createDialogOpen, setCreateDialogOpen] = useState(false)
  const [formData, setFormData] = useState<PatientFormData>({
    patientId: '',
    firstName: '',
    lastName: '',
    dateOfBirth: '',
    gender: '',
    phone: '',
    email: '',
  })

  const dispatch = useAppDispatch()
  const { patients, isLoading, error } = useAppSelector(state => state.patient)

  useEffect(() => {
    dispatch(fetchPatients({}))
  }, [dispatch])

  const handleInputChange =
    (field: keyof PatientFormData) =>
    (event: React.ChangeEvent<HTMLInputElement>) => {
      setFormData(prev => ({
        ...prev,
        [field]: event.target.value,
      }))
    }

  const handleGenderChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setFormData(prev => ({
      ...prev,
      gender: event.target.value,
    }))
  }

  const handleCreatePatient = async () => {
    dispatch(clearError())
    await dispatch(createPatient(formData))
    setCreateDialogOpen(false)
    setFormData({
      patientId: '',
      firstName: '',
      lastName: '',
      dateOfBirth: '',
      gender: '',
      phone: '',
      email: '',
    })
    dispatch(fetchPatients({}))
  }

  const isFormValid =
    formData.patientId &&
    formData.firstName &&
    formData.lastName &&
    formData.dateOfBirth &&
    formData.gender

  return (
    <Box>
      <div className="flex justify-between items-center mb-6">
        <Typography variant="h4">Patients</Typography>
        <Button variant="contained" onClick={() => setCreateDialogOpen(true)} className="flex items-center space-x-2">
          <span>➕</span>
          <span>Add Patient</span>
        </Button>
      </div>

      {error && (
        <Alert severity="error" className="mb-4">
          {error}
        </Alert>
      )}

      {isLoading && (
        <div className="w-full bg-gray-200 rounded-full h-2.5 mb-4">
          <div className="bg-blue-600 h-2.5 rounded-full animate-pulse" style={{width: '45%'}}></div>
        </div>
      )}

      <Card>
        <CardContent>
          <Typography variant="h6" className="mb-4">
            Patient List
          </Typography>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white border border-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Patient ID</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Name</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date of Birth</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Gender</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Phone</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Email</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {patients.map(patient => (
                  <tr key={patient.id}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{patient.patientId}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {patient.firstName} {patient.lastName}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {new Date(patient.dateOfBirth).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{patient.gender}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{patient.phone || 'N/A'}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{patient.email || 'N/A'}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      <Button variant="outlined" onClick={() => {}} className="flex items-center space-x-1">
                        <span>✏️</span>
                        <span>Edit</span>
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {createDialogOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-2xl mx-4 max-h-[90vh] overflow-y-auto">
            <Typography variant="h6" className="mb-4">Add New Patient</Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Patient ID"
                  value={formData.patientId}
                  onChange={handleInputChange('patientId')}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <div className="w-full">
                  <label className="block text-sm font-medium text-gray-700 mb-1">Gender</label>
                  <select
                    value={formData.gender}
                    onChange={handleGenderChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="">Select Gender</option>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                    <option value="other">Other</option>
                  </select>
                </div>
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="First Name"
                  value={formData.firstName}
                  onChange={handleInputChange('firstName')}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Last Name"
                  value={formData.lastName}
                  onChange={handleInputChange('lastName')}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Date of Birth"
                  type="date"
                  value={formData.dateOfBirth}
                  onChange={handleInputChange('dateOfBirth')}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Phone"
                  value={formData.phone}
                  onChange={handleInputChange('phone')}
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Email"
                  type="email"
                  value={formData.email}
                  onChange={handleInputChange('email')}
                />
              </Grid>
            </Grid>
            <div className="flex justify-end space-x-3 mt-6">
              <Button variant="outlined" onClick={() => setCreateDialogOpen(false)}>
                Cancel
              </Button>
              <Button
                onClick={handleCreatePatient}
                variant="contained"
                disabled={!isFormValid || isLoading}
              >
                Create Patient
              </Button>
            </div>
          </div>
        </div>
      )}
    </Box>
  )
}

export default PatientsPage
