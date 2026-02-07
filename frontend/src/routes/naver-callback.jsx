import { createFileRoute } from '@tanstack/react-router'
import NaverCallbackPage from '@/pages/NaverCallback/NaverCallbackPage.jsx'

export const Route = createFileRoute('/naver-callback')({
  component: NaverCallbackPage,
})
