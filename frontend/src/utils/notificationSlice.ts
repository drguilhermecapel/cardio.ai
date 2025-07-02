import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'

interface Notification {
  id: number
  title: string
  message: string
  notificationType: string
  priority: string
  isRead: boolean
  createdAt: string
}

interface NotificationState {
  notifications: Notification[]
  unreadCount: number
  isLoading: boolean
  error: string | null
}

const initialState: NotificationState = {
  notifications: [],
  unreadCount: 0,
  isLoading: false,
  error: null,
}

export const fetchNotifications = createAsyncThunk(
  'notification/fetchNotifications',
  async (params: { limit?: number; offset?: number; unreadOnly?: boolean } = {}) => {
    const searchParams = new URLSearchParams()
    if (params.limit) searchParams.append('limit', params.limit.toString())
    if (params.offset) searchParams.append('offset', params.offset.toString())
    if (params.unreadOnly) searchParams.append('unread_only', 'true')

    const response = await fetch(`/api/v1/notifications/?${searchParams}`, {
      headers: {
        Authorization: `Bearer ${localStorage.getItem('token')}`,
      },
      credentials: 'omit',
    })

    if (!response.ok) {
      throw new Error('Failed to fetch notifications')
    }

    return response.json()
  }
)

export const markAsRead = createAsyncThunk(
  'notification/markAsRead',
  async (notificationId: number) => {
    const response = await fetch(`/api/v1/notifications/${notificationId}/read`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${localStorage.getItem('token')}`,
      },
      credentials: 'omit',
    })

    if (!response.ok) {
      throw new Error('Failed to mark notification as read')
    }

    return notificationId
  }
)

export const fetchUnreadCount = createAsyncThunk('notification/fetchUnreadCount', async () => {
  const response = await fetch('/api/v1/notifications/unread-count', {
    headers: {
      Authorization: `Bearer ${localStorage.getItem('token')}`,
    },
    credentials: 'omit',
  })

  if (!response.ok) {
    throw new Error('Failed to fetch unread count')
  }

  return response.json()
})

const notificationSlice = createSlice({
  name: 'notification',
  initialState,
  reducers: {
    clearError: state => {
      state.error = null
    },
  },
  extraReducers: builder => {
    builder
      .addCase(fetchNotifications.pending, state => {
        state.isLoading = true
        state.error = null
      })
      .addCase(fetchNotifications.fulfilled, (state, action) => {
        state.isLoading = false
        state.notifications = action.payload.notifications
      })
      .addCase(fetchNotifications.rejected, (state, action) => {
        state.isLoading = false
        state.error = action.error.message || 'Failed to fetch notifications'
      })
      .addCase(markAsRead.fulfilled, (state, action) => {
        const notification = state.notifications.find(n => n.id === action.payload)
        if (notification) {
          notification.isRead = true
          state.unreadCount = Math.max(0, state.unreadCount - 1)
        }
      })
      .addCase(fetchUnreadCount.fulfilled, (state, action) => {
        state.unreadCount = action.payload.unread_count
      })
  },
})

export const { clearError } = notificationSlice.actions
export default notificationSlice.reducer
