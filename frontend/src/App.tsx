import { Routes, Route, Navigate } from 'react-router-dom'
import AppLayout from '@/components/AppLayout'
import RecordPage from '@/pages/RecordPage'
import HistoryPage from '@/pages/HistoryPage'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<AppLayout />}>
        <Route index element={<Navigate to="/record" replace />} />
        <Route path="record" element={<RecordPage />} />
        <Route path="history" element={<HistoryPage />} />
      </Route>
    </Routes>
  )
}
