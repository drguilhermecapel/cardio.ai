import React, { useState } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Grid,
  Alert,
  CircularProgress,
} from '../components/ui/BasicComponents'
import { useAppSelector } from '../hooks/redux'

interface ProfileFormData {
  firstName: string
  lastName: string
  email: string
  phone: string
  currentPassword: string
  newPassword: string
  confirmPassword: string
}

const ProfilePage: React.FC = () => {
  const { user } = useAppSelector(state => state.auth)
  const [isLoading, setIsLoading] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const [formData, setFormData] = useState<ProfileFormData>({
    firstName: user?.firstName || '',
    lastName: user?.lastName || '',
    email: user?.email || '',
    phone: '',
    currentPassword: '',
    newPassword: '',
    confirmPassword: '',
  })

  const handleInputChange =
    (field: keyof ProfileFormData) =>
    (event: React.ChangeEvent<HTMLInputElement>): void => {
      setFormData(prev => ({
        ...prev,
        [field]: event.target.value,
      }))
    }

  const handleUpdateProfile = async (): Promise<void> => {
    setIsLoading(true)
    setMessage(null)

    try {
      await new Promise(resolve => setTimeout(resolve, 1000))
      setMessage({ type: 'success', text: 'Profile updated successfully!' })
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to update profile. Please try again.' })
    } finally {
      setIsLoading(false)
    }
  }

  const handleChangePassword = async (): Promise<void> => {
    if (formData.newPassword !== formData.confirmPassword) {
      setMessage({ type: 'error', text: 'New passwords do not match.' })
      return
    }

    if (formData.newPassword.length < 8) {
      setMessage({ type: 'error', text: 'Password must be at least 8 characters long.' })
      return
    }

    setIsLoading(true)
    setMessage(null)

    try {
      await new Promise(resolve => setTimeout(resolve, 1000))
      setMessage({ type: 'success', text: 'Password changed successfully!' })
      setFormData(prev => ({
        ...prev,
        currentPassword: '',
        newPassword: '',
        confirmPassword: '',
      }))
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to change password. Please try again.' })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <Box>
      <Typography variant="h4" className="mb-6">
        Profile
      </Typography>

      {message && (
        <Alert severity={message.type} className="mb-4">
          {message.text}
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent className="text-center">
              <div className="w-24 h-24 mx-auto mb-4 bg-gray-300 rounded-full flex items-center justify-center text-2xl font-bold text-gray-600">
                {user?.firstName?.[0] || '👤'}
              </div>
              <Typography variant="h6" className="mb-2">
                {user?.firstName} {user?.lastName}
              </Typography>
              <Typography variant="body2" className="text-gray-600 mb-1">
                {user?.role}
              </Typography>
              <Typography variant="body2" className="text-gray-600">
                {user?.email}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" className="mb-4">
                Personal Information
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="First Name"
                    value={formData.firstName}
                    onChange={handleInputChange('firstName')}
                    disabled={isLoading}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Last Name"
                    value={formData.lastName}
                    onChange={handleInputChange('lastName')}
                    disabled={isLoading}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Email"
                    type="email"
                    value={formData.email}
                    onChange={handleInputChange('email')}
                    disabled={isLoading}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Phone"
                    value={formData.phone}
                    onChange={handleInputChange('phone')}
                    disabled={isLoading}
                  />
                </Grid>
                <Grid item xs={12}>
                  <Button
                    variant="contained"
                    onClick={handleUpdateProfile}
                    disabled={isLoading}
                    className="flex items-center space-x-2"
                  >
                    {isLoading ? <CircularProgress size={20} /> : <span>💾</span>}
                    <span>Update Profile</span>
                  </Button>
                </Grid>
              </Grid>
            </CardContent>
          </Card>

          <Card className="mt-6">
            <CardContent>
              <Typography variant="h6" className="mb-4">
                Change Password
              </Typography>
              <div className="border-b border-gray-200 mb-4"></div>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Current Password"
                    type="password"
                    value={formData.currentPassword}
                    onChange={handleInputChange('currentPassword')}
                    disabled={isLoading}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="New Password"
                    type="password"
                    value={formData.newPassword}
                    onChange={handleInputChange('newPassword')}
                    disabled={isLoading}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Confirm New Password"
                    type="password"
                    value={formData.confirmPassword}
                    onChange={handleInputChange('confirmPassword')}
                    disabled={isLoading}
                  />
                </Grid>
                <Grid item xs={12}>
                  <Button
                    variant="contained"
                    color="secondary"
                    onClick={handleChangePassword}
                    disabled={
                      isLoading ||
                      !formData.currentPassword ||
                      !formData.newPassword ||
                      !formData.confirmPassword
                    }
                    className="flex items-center space-x-2"
                  >
                    {isLoading ? <CircularProgress size={20} /> : <span>💾</span>}
                    <span>Change Password</span>
                  </Button>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}

export default ProfilePage
