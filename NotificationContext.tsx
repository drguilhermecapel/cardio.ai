// Real-time Notifications System for CardioAI Pro
// Sistema de notificações em tempo real com WebSocket e Service Workers

import React, { createContext, useContext, useReducer, useEffect, useCallback } from 'react'
import { useAuth } from './AuthContext'

// Types
interface Notification {
  id: string
  type: 'critical' | 'warning' | 'info' | 'success'
  category: 'ecg' | 'patient' | 'system' | 'security' | 'ai'
  title: string
  message: string
  timestamp: string
  read: boolean
  priority: 'low' | 'medium' | 'high' | 'critical'
  patientId?: string
  ecgId?: string
  actions?: NotificationAction[]
  metadata?: Record<string, any>
}

interface NotificationAction {
  id: string
  label: string
  type: 'primary' | 'secondary' | 'danger'
  action: string
  params?: Record<string, any>
}

interface NotificationState {
  notifications: Notification[]
  unreadCount: number
  isConnected: boolean
  soundEnabled: boolean
  pushEnabled: boolean
  filters: NotificationFilters
}

interface NotificationFilters {
  types: string[]
  categories: string[]
  priorities: string[]
  showRead: boolean
}

interface NotificationContextType extends NotificationState {
  markAsRead: (id: string) => void
  markAllAsRead: () => void
  deleteNotification: (id: string) => void
  clearAll: () => void
  updateFilters: (filters: Partial<NotificationFilters>) => void
  toggleSound: () => void
  togglePush: () => void
  executeAction: (notificationId: string, actionId: string) => Promise<boolean>
  requestPermission: () => Promise<boolean>
}

// Actions
type NotificationAction_Type =
  | { type: 'ADD_NOTIFICATION'; payload: Notification }
  | { type: 'MARK_AS_READ'; payload: string }
  | { type: 'MARK_ALL_AS_READ' }
  | { type: 'DELETE_NOTIFICATION'; payload: string }
  | { type: 'CLEAR_ALL' }
  | { type: 'SET_NOTIFICATIONS'; payload: Notification[] }
  | { type: 'UPDATE_FILTERS'; payload: Partial<NotificationFilters> }
  | { type: 'SET_CONNECTION_STATUS'; payload: boolean }
  | { type: 'TOGGLE_SOUND' }
  | { type: 'TOGGLE_PUSH' }

// Reducer
const notificationReducer = (state: NotificationState, action: NotificationAction_Type): NotificationState => {
  switch (action.type) {
    case 'ADD_NOTIFICATION':
      const newNotifications = [action.payload, ...state.notifications]
      return {
        ...state,
        notifications: newNotifications,
        unreadCount: state.unreadCount + 1
      }

    case 'MARK_AS_READ':
      return {
        ...state,
        notifications: state.notifications.map(n =>
          n.id === action.payload ? { ...n, read: true } : n
        ),
        unreadCount: Math.max(0, state.unreadCount - 1)
      }

    case 'MARK_ALL_AS_READ':
      return {
        ...state,
        notifications: state.notifications.map(n => ({ ...n, read: true })),
        unreadCount: 0
      }

    case 'DELETE_NOTIFICATION':
      const notification = state.notifications.find(n => n.id === action.payload)
      return {
        ...state,
        notifications: state.notifications.filter(n => n.id !== action.payload),
        unreadCount: notification && !notification.read ? state.unreadCount - 1 : state.unreadCount
      }

    case 'CLEAR_ALL':
      return {
        ...state,
        notifications: [],
        unreadCount: 0
      }

    case 'SET_NOTIFICATIONS':
      const unread = action.payload.filter(n => !n.read).length
      return {
        ...state,
        notifications: action.payload,
        unreadCount: unread
      }

    case 'UPDATE_FILTERS':
      return {
        ...state,
        filters: { ...state.filters, ...action.payload }
      }

    case 'SET_CONNECTION_STATUS':
      return {
        ...state,
        isConnected: action.payload
      }

    case 'TOGGLE_SOUND':
      const soundEnabled = !state.soundEnabled
      localStorage.setItem('cardioai_sound_enabled', soundEnabled.toString())
      return {
        ...state,
        soundEnabled
      }

    case 'TOGGLE_PUSH':
      const pushEnabled = !state.pushEnabled
      localStorage.setItem('cardioai_push_enabled', pushEnabled.toString())
      return {
        ...state,
        pushEnabled
      }

    default:
      return state
  }
}

// Initial state
const initialState: NotificationState = {
  notifications: [],
  unreadCount: 0,
  isConnected: false,
  soundEnabled: localStorage.getItem('cardioai_sound_enabled') !== 'false',
  pushEnabled: localStorage.getItem('cardioai_push_enabled') === 'true',
  filters: {
    types: ['critical', 'warning', 'info', 'success'],
    categories: ['ecg', 'patient', 'system', 'security', 'ai'],
    priorities: ['low', 'medium', 'high', 'critical'],
    showRead: false
  }
}

// Context
const NotificationContext = createContext<NotificationContextType | undefined>(undefined)

// Provider
export const NotificationProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(notificationReducer, initialState)
  const { user, token, isAuthenticated } = useAuth()
  const wsRef = React.useRef<WebSocket | null>(null)

  // WebSocket connection
  const connectWebSocket = useCallback(() => {
    if (!isAuthenticated || !token) return

    const wsUrl = `${process.env.VITE_WS_URL || 'ws://localhost:8080'}/notifications`
    const ws = new WebSocket(`${wsUrl}?token=${token}`)

    ws.onopen = () => {
      console.log('Notification WebSocket connected')
      dispatch({ type: 'SET_CONNECTION_STATUS', payload: true })
    }

    ws.onmessage = (event) => {
      try {
        const notification: Notification = JSON.parse(event.data)
        dispatch({ type: 'ADD_NOTIFICATION', payload: notification })
        
        // Play sound if enabled
        if (state.soundEnabled && notification.priority !== 'low') {
          playNotificationSound(notification.priority)
        }

        // Show browser notification if enabled
        if (state.pushEnabled && 'Notification' in window && Notification.permission === 'granted') {
          showBrowserNotification(notification)
        }
      } catch (error) {
        console.error('Error parsing notification:', error)
      }
    }

    ws.onclose = () => {
      console.log('Notification WebSocket disconnected')
      dispatch({ type: 'SET_CONNECTION_STATUS', payload: false })
      
      // Reconnect after 5 seconds
      setTimeout(() => {
        if (isAuthenticated) {
          connectWebSocket()
        }
      }, 5000)
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      dispatch({ type: 'SET_CONNECTION_STATUS', payload: false })
    }

    wsRef.current = ws
  }, [isAuthenticated, token, state.soundEnabled, state.pushEnabled])

  // Connect WebSocket when authenticated
  useEffect(() => {
    if (isAuthenticated) {
      connectWebSocket()
      loadNotifications()
    } else {
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [isAuthenticated, connectWebSocket])

  // Load initial notifications
  const loadNotifications = async () => {
    try {
      const response = await fetch('/api/notifications', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      if (response.ok) {
        const notifications = await response.json()
        dispatch({ type: 'SET_NOTIFICATIONS', payload: notifications })
      }
    } catch (error) {
      console.error('Error loading notifications:', error)
    }
  }

  // Play notification sound
  const playNotificationSound = (priority: string) => {
    const audio = new Audio()
    
    switch (priority) {
      case 'critical':
        audio.src = '/sounds/critical-alert.mp3'
        break
      case 'high':
        audio.src = '/sounds/high-priority.mp3'
        break
      case 'medium':
        audio.src = '/sounds/medium-priority.mp3'
        break
      default:
        audio.src = '/sounds/notification.mp3'
    }

    audio.play().catch(() => {
      // Ignore audio play errors (user interaction required)
    })
  }

  // Show browser notification
  const showBrowserNotification = (notification: Notification) => {
    const browserNotification = new Notification(notification.title, {
      body: notification.message,
      icon: '/icons/cardioai-icon.png',
      badge: '/icons/cardioai-badge.png',
      tag: notification.id,
      requireInteraction: notification.priority === 'critical',
      actions: notification.actions?.slice(0, 2).map(action => ({
        action: action.id,
        title: action.label
      }))
    })

    browserNotification.onclick = () => {
      window.focus()
      markAsRead(notification.id)
      browserNotification.close()
    }

    // Auto-close after 10 seconds for non-critical notifications
    if (notification.priority !== 'critical') {
      setTimeout(() => {
        browserNotification.close()
      }, 10000)
    }
  }

  // Service Worker for background notifications
  useEffect(() => {
    if ('serviceWorker' in navigator && state.pushEnabled) {
      navigator.serviceWorker.register('/sw.js')
        .then((registration) => {
          console.log('Service Worker registered:', registration)
        })
        .catch((error) => {
          console.error('Service Worker registration failed:', error)
        })
    }
  }, [state.pushEnabled])

  // Actions
  const markAsRead = async (id: string) => {
    dispatch({ type: 'MARK_AS_READ', payload: id })
    
    try {
      await fetch(`/api/notifications/${id}/read`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
    } catch (error) {
      console.error('Error marking notification as read:', error)
    }
  }

  const markAllAsRead = async () => {
    dispatch({ type: 'MARK_ALL_AS_READ' })
    
    try {
      await fetch('/api/notifications/read-all', {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
    } catch (error) {
      console.error('Error marking all notifications as read:', error)
    }
  }

  const deleteNotification = async (id: string) => {
    dispatch({ type: 'DELETE_NOTIFICATION', payload: id })
    
    try {
      await fetch(`/api/notifications/${id}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
    } catch (error) {
      console.error('Error deleting notification:', error)
    }
  }

  const clearAll = async () => {
    dispatch({ type: 'CLEAR_ALL' })
    
    try {
      await fetch('/api/notifications', {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
    } catch (error) {
      console.error('Error clearing all notifications:', error)
    }
  }

  const updateFilters = (filters: Partial<NotificationFilters>) => {
    dispatch({ type: 'UPDATE_FILTERS', payload: filters })
  }

  const toggleSound = () => {
    dispatch({ type: 'TOGGLE_SOUND' })
  }

  const togglePush = async () => {
    if (!state.pushEnabled) {
      const granted = await requestPermission()
      if (!granted) return
    }
    
    dispatch({ type: 'TOGGLE_PUSH' })
  }

  const executeAction = async (notificationId: string, actionId: string): Promise<boolean> => {
    try {
      const response = await fetch(`/api/notifications/${notificationId}/actions/${actionId}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      })

      if (response.ok) {
        markAsRead(notificationId)
        return true
      }
      return false
    } catch (error) {
      console.error('Error executing notification action:', error)
      return false
    }
  }

  const requestPermission = async (): Promise<boolean> => {
    if (!('Notification' in window)) {
      console.warn('This browser does not support notifications')
      return false
    }

    if (Notification.permission === 'granted') {
      return true
    }

    if (Notification.permission === 'denied') {
      return false
    }

    const permission = await Notification.requestPermission()
    return permission === 'granted'
  }

  const value: NotificationContextType = {
    ...state,
    markAsRead,
    markAllAsRead,
    deleteNotification,
    clearAll,
    updateFilters,
    toggleSound,
    togglePush,
    executeAction,
    requestPermission
  }

  return (
    <NotificationContext.Provider value={value}>
      {children}
    </NotificationContext.Provider>
  )
}

// Hook
export const useNotifications = (): NotificationContextType => {
  const context = useContext(NotificationContext)
  if (context === undefined) {
    throw new Error('useNotifications must be used within a NotificationProvider')
  }
  return context
}

// Notification component
export const NotificationToast: React.FC<{ notification: Notification; onClose: () => void }> = ({
  notification,
  onClose
}) => {
  const { executeAction } = useNotifications()

  const getIcon = () => {
    switch (notification.type) {
      case 'critical': return '🚨'
      case 'warning': return '⚠️'
      case 'success': return '✅'
      default: return 'ℹ️'
    }
  }

  const getColorClass = () => {
    switch (notification.type) {
      case 'critical': return 'border-red-500 bg-red-50'
      case 'warning': return 'border-yellow-500 bg-yellow-50'
      case 'success': return 'border-green-500 bg-green-50'
      default: return 'border-blue-500 bg-blue-50'
    }
  }

  return (
    <div className={`p-4 border-l-4 rounded-lg shadow-lg ${getColorClass()} max-w-sm`}>
      <div className="flex items-start">
        <span className="text-2xl mr-3">{getIcon()}</span>
        <div className="flex-1">
          <h4 className="font-semibold text-gray-900">{notification.title}</h4>
          <p className="text-sm text-gray-700 mt-1">{notification.message}</p>
          
          {notification.actions && notification.actions.length > 0 && (
            <div className="mt-3 space-x-2">
              {notification.actions.map(action => (
                <button
                  key={action.id}
                  onClick={() => executeAction(notification.id, action.id)}
                  className={`px-3 py-1 text-xs rounded ${
                    action.type === 'primary' ? 'bg-blue-600 text-white' :
                    action.type === 'danger' ? 'bg-red-600 text-white' :
                    'bg-gray-200 text-gray-700'
                  }`}
                >
                  {action.label}
                </button>
              ))}
            </div>
          )}
        </div>
        
        <button
          onClick={onClose}
          className="ml-2 text-gray-400 hover:text-gray-600"
        >
          ✕
        </button>
      </div>
    </div>
  )
}

export default NotificationProvider

