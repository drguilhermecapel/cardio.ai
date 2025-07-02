import React, { useState, useEffect } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Alert,
} from '../components/ui/BasicComponents'
import { useAppDispatch, useAppSelector } from '../hooks/redux'
import { uploadECG, fetchAnalyses, clearError } from '../store/slices/ecgSlice'
import { fetchPatients } from '../store/slices/patientSlice'

const ECGAnalysisPage: React.FC = () => {
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [selectedPatientId, setSelectedPatientId] = useState<number | ''>('')

  const dispatch = useAppDispatch()
  const { analyses, isLoading, error, uploadProgress } = useAppSelector(state => state.ecg)
  const { patients } = useAppSelector(state => state.patient)

  useEffect(() => {
    dispatch(fetchAnalyses({}))
    dispatch(fetchPatients({}))
  }, [dispatch])

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setSelectedFile(file)
    }
  }

  const handleUpload = async () => {
    if (selectedFile && selectedPatientId) {
      dispatch(clearError())
      await dispatch(
        uploadECG({
          patientId: selectedPatientId as number,
          file: selectedFile,
        })
      )
      setUploadDialogOpen(false)
      setSelectedFile(null)
      setSelectedPatientId('')
      dispatch(fetchAnalyses({}))
    }
  }

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800'
      case 'processing':
        return 'bg-blue-100 text-blue-800'
      case 'pending':
        return 'bg-yellow-100 text-yellow-800'
      case 'failed':
        return 'bg-red-100 text-red-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  const getUrgencyColor = (urgency: string): string => {
    switch (urgency) {
      case 'critical':
        return 'bg-red-100 text-red-800'
      case 'high':
        return 'bg-yellow-100 text-yellow-800'
      case 'medium':
        return 'bg-blue-100 text-blue-800'
      case 'low':
        return 'bg-green-100 text-green-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  return (
    <Box>
      <div className="flex justify-between items-center mb-6">
        <Typography variant="h4">ECG Analysis</Typography>
        <Button
          variant="contained"
          onClick={() => setUploadDialogOpen(true)}
          className="flex items-center space-x-2"
        >
          <span>☁️</span>
          <span>Upload ECG</span>
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
            ECG Analyses
          </Typography>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white border border-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Analysis ID</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Patient ID</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Diagnosis</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Urgency</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {analyses.map(analysis => (
                  <tr key={analysis.id}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{analysis.analysisId}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{analysis.patientId}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(analysis.status)}`}>
                        {analysis.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{analysis.diagnosis || 'Pending'}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getUrgencyColor(analysis.clinicalUrgency)}`}>
                        {analysis.clinicalUrgency}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {analysis.confidence ? `${(analysis.confidence * 100).toFixed(1)}%` : 'N/A'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {new Date(analysis.createdAt).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      <Button variant="outlined" onClick={() => {}} className="flex items-center space-x-1">
                        <span>👁️</span>
                        <span>View</span>
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {uploadDialogOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md mx-4">
            <Typography variant="h6" className="mb-4">Upload ECG File</Typography>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <div className="w-full">
                  <label className="block text-sm font-medium text-gray-700 mb-1">Select Patient</label>
                  <select
                    value={selectedPatientId}
                    onChange={(e) => setSelectedPatientId(e.target.value as any)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="">Select a patient</option>
                    {patients.map(patient => (
                      <option key={patient.id} value={patient.id}>
                        {patient.firstName} {patient.lastName} ({patient.patientId})
                      </option>
                    ))}
                  </select>
                </div>
              </Grid>
              <Grid item xs={12}>
                <label className="block w-full">
                  <div className="w-full px-3 py-4 border-2 border-dashed border-gray-300 rounded-md text-center cursor-pointer hover:border-blue-500">
                    {selectedFile ? selectedFile.name : 'Choose ECG File (.csv, .txt, .xml, .dat)'}
                  </div>
                  <input
                    type="file"
                    className="hidden"
                    accept=".csv,.txt,.xml,.dat"
                    onChange={handleFileSelect}
                  />
                </label>
              </Grid>
              {uploadProgress > 0 && uploadProgress < 100 && (
                <Grid item xs={12}>
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div 
                      className="bg-blue-600 h-2.5 rounded-full" 
                      style={{width: `${uploadProgress}%`}}
                    ></div>
                  </div>
                  <Typography variant="body2" className="text-gray-600 mt-1">
                    Uploading... {uploadProgress}%
                  </Typography>
                </Grid>
              )}
            </Grid>
            <div className="flex justify-end space-x-3 mt-6">
              <Button variant="outlined" onClick={() => setUploadDialogOpen(false)}>
                Cancel
              </Button>
              <Button
                onClick={handleUpload}
                variant="contained"
                disabled={!selectedFile || !selectedPatientId || isLoading}
              >
                Upload
              </Button>
            </div>
          </div>
        </div>
      )}
    </Box>
  )
}

export default ECGAnalysisPage
