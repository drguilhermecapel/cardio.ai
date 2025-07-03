import React, { useEffect } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Alert,
  Button,
} from '../components/ui/BasicComponents'
import { useAppDispatch, useAppSelector } from '../hooks/redux'
import { fetchNotifications, markAsRead } from '../store/slices/notificationSlice'

const NotificationsPage: React.FC = () => {
  const dispatch = useAppDispatch()
  const { notifications, isLoading, error } = useAppSelector(state => state.notification)

  useEffect(() => {
    dispatch(fetchNotifications({}))
  }, [dispatch])

  const handleMarkAsRead = (notificationId: number): void => {
    dispatch(markAsRead(notificationId))
  }

  const getPriorityColor = (priority: string): string => {
    switch (priority) {
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

  const getNotificationIcon = (type: string): string => {
    switch (type) {
      case 'validation_assignment':
        return '✅'
      case 'urgent_alert':
        return '⚠️'
      case 'completion_notification':
        return 'ℹ️'
      default:
        return '🔔'
    }
  }

  return (
    <Box>
      <Typography variant="h4" className="mb-6">
        Notifications
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
            Recent Notifications
          </Typography>
          <div className="space-y-3">
            {notifications.map(notification => (
              <div
                key={notification.id}
                className={`border border-gray-200 rounded-lg p-4 ${
                  notification.isRead ? 'bg-white' : 'bg-gray-50'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-3">
                    <span className="text-2xl">
                      {getNotificationIcon(notification.notificationType)}
                    </span>
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <Typography variant="h6" className="font-medium">
                          {notification.title}
                        </Typography>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getPriorityColor(notification.priority)}`}>
                          {notification.priority}
                        </span>
                        {!notification.isRead && (
                          <span className="px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                            New
                          </span>
                        )}
                      </div>
                      <Typography variant="body2" className="text-gray-600 mb-2">
                        {notification.message}
                      </Typography>
                      <Typography variant="caption" className="text-gray-500">
                        {new Date(notification.createdAt).toLocaleString()}
                      </Typography>
                    </div>
                  </div>
                  {!notification.isRead && (
                    <Button
                      variant="outlined"
                      onClick={() => handleMarkAsRead(notification.id)}
                      className="ml-4"
                    >
                      Mark as Read
                    </Button>
                  )}
                </div>
              </div>
            ))}
            {notifications.length === 0 && !isLoading && (
              <div className="text-center py-8">
                <Typography variant="body1" className="text-gray-600">
                  No notifications
                </Typography>
                <Typography variant="body2" className="text-gray-500">
                  You're all caught up!
                </Typography>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </Box>
  )
}

export default NotificationsPage
