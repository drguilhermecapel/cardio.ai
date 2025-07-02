import React, { useEffect } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Alert,
} from '../components/ui/BasicComponents'
import { useAppDispatch, useAppSelector } from '../hooks/redux'
import { fetchMyValidations } from '../store/slices/validationSlice'

const ValidationsPage: React.FC = () => {
  const dispatch = useAppDispatch()
  const { validations, isLoading, error } = useAppSelector(state => state.validation)

  useEffect(() => {
    dispatch(fetchMyValidations({}))
  }, [dispatch])

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800'
      case 'pending':
        return 'bg-yellow-100 text-yellow-800'
      case 'in_progress':
        return 'bg-blue-100 text-blue-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  return (
    <Box>
      <Typography variant="h4" className="mb-6">
        Validations
      </Typography>

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
            My Validation Assignments
          </Typography>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white border border-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Validation ID</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Analysis ID</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Approved</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Clinical Notes</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Created Date</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {validations.map(validation => (
                  <tr key={validation.id}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{validation.id}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{validation.analysisId}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(validation.status)}`}>
                        {validation.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {validation.approved !== undefined ? (
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          validation.approved ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                        }`}>
                          {validation.approved ? 'Yes' : 'No'}
                        </span>
                      ) : (
                        'Pending'
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 max-w-xs truncate">
                      {validation.clinicalNotes || 'No notes'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {new Date(validation.createdAt).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {validation.status === 'pending' ? (
                        <Button
                          variant="contained"
                          onClick={() => {}}
                          className="flex items-center space-x-1"
                        >
                          <span>📋</span>
                          <span>Review</span>
                        </Button>
                      ) : (
                        <Button
                          variant="outlined"
                          onClick={() => {}}
                          className="flex items-center space-x-1"
                        >
                          <span>✅</span>
                          <span>View</span>
                        </Button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </Box>
  )
}

export default ValidationsPage
