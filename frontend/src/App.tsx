/** @jsxImportSource react */
import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'
import MessageComponent from './components/Message'
import ThinkingIndicator from './components/ThinkingIndicator'
import EmptyState from './components/EmptyState'
import { useTranslation } from 'react-i18next'
import profileImage from './assets/profile.png'

// 로봇 아이콘 대신 프로필 이미지를 사용하는 함수 추가
const RobotIcon = () => (
  <div className="flex items-center justify-center h-8 w-8 rounded-full bg-green-500 text-white flex-shrink-0 mr-3 ml-1">
    <img 
      src={profileImage}
      alt="AI Profile"
      className="w-full h-full object-cover rounded-full"
    />
  </div>
);

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const API_PREFIX = import.meta.env.VITE_API_PREFIX || '/api/v1';

function App() {
  const { t, i18n } = useTranslation();

  const [messages, setMessages] = useState<any[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [isThinking, setIsThinking] = useState(false)
  const [thinkingContent, setThinkingContent] = useState<string | null>(null)
  const [sessionId, setSessionId] = useState('')
  const [showSidebar, setShowSidebar] = useState(false)
  const [currentChatId, setCurrentChatId] = useState<string | null>(null)
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const sessionIdRef = useRef<string | null>(null)

  const changeLanguage = (lng: string) => {
    i18n.changeLanguage(lng);
  };

  const getLanguageToggleLabel = () => {
    return i18n.language === 'ko' ? 'English' : '한국어';
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || isThinking) return;

    const newUserMessage: any = {
      id: Date.now().toString(),
      sender: 'user',
      content: inputMessage,
    };
    setMessages((prev) => [...prev, newUserMessage]);
    setInputMessage('');
    resizeTextarea();
    setIsThinking(true);
    setThinkingContent(null);
    
    const currentSessionId = sessionIdRef.current || `session_${Date.now()}`;
    if (!sessionIdRef.current) {
      sessionIdRef.current = currentSessionId;
    }
    
    let finalAnswerReceived = false; 

    try {
      const apiUrl = `${API_BASE_URL}${API_PREFIX}/stream`;
      console.log("Sending POST request to:", apiUrl, "with session:", currentSessionId);

      const response = await fetch(apiUrl, { 
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          text: newUserMessage.content, 
          session_id: currentSessionId, 
        }),
      });

      if (!response.ok) {
          let errorBody = 'Unknown error';
          try { errorBody = await response.text(); } catch (e) {}
          throw new Error(`API request failed with status ${response.status}: ${errorBody}`);
      }

      if (!response.body) {
        throw new Error('Streaming response body is null');
      }

      const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
      let accumulatedResponse = '';
      
      while (true) {
        const { value, done } = await reader.read();
        if (done) {
           if (!finalAnswerReceived) {
               console.warn("Stream ended without a final answer event.");
               setMessages((prev) => [
                 ...prev,
                 {
                   id: (Date.now() + 1).toString(), 
                   sender: 'assistant',
                   content: "스트림 연결이 종료되었거나 응답을 받지 못했습니다.",
                   isError: true,
                 } as any,
               ]);
           }
           break;
        }

        const lines = value.split('\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const jsonData = JSON.parse(line.substring(6));
              
              if (jsonData.type === 'thinking') {
                setThinkingContent(jsonData.content || ''); 
              } else if (jsonData.type === 'final') {
                accumulatedResponse = jsonData.content || '';
                finalAnswerReceived = true; 
                setMessages((prev) => [
                  ...prev,
                  {
                    id: (Date.now() + 1).toString(), 
                    sender: 'assistant',
                    content: accumulatedResponse,
                  } as any,
                ]);
                setIsThinking(false);
                setThinkingContent(null); 
              } else if (jsonData.type === 'error') {
                console.error('Streaming error from backend:', jsonData.content);
                setMessages((prev) => [
                  ...prev,
                  {
                    id: (Date.now() + 1).toString(), 
                    sender: 'assistant',
                    content: `오류: ${jsonData.content}`,
                    isError: true,
                  } as any,
                ]);
                 setIsThinking(false); 
                 setThinkingContent(null);
              } else if (jsonData.type === 'feedback_provided') { 
                 console.log("Feedback received from backend:", jsonData.feedback_provided);
              }
            } catch (e) {
              console.error('Failed to parse streaming data chunk:', line, e);
            }
          }
        }
      }

    } catch (error) {
      console.error('Failed to send message or process stream:', error);
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(), 
          sender: 'assistant',
          content: `메시지 전송/처리 중 오류: ${error instanceof Error ? error.message : String(error)}`,
          isError: true,
        } as any,
      ]);
      setIsThinking(false);
      setThinkingContent(null);
    } finally {
      if (!finalAnswerReceived) { 
         if (isThinking) {
             setIsThinking(false);
             setThinkingContent(null); 
             console.log("Reset thinking state in finally block as no final answer was processed.");
         }
      }
    }
  };

  useEffect(() => {
    const newSessionId = Date.now().toString()
    setSessionId(newSessionId)
    setCurrentChatId(newSessionId)
    sessionIdRef.current = newSessionId

    const handleResize = () => {
      const mobile = window.innerWidth < 768
      setIsMobile(mobile)
      if (!mobile) setShowSidebar(false)
    }
    window.addEventListener('resize', handleResize)
    handleResize()
    return () => {
      window.removeEventListener('resize', handleResize)
    }
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, isThinking])

  useEffect(() => {
    resizeTextarea()
  }, [inputMessage])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const resizeTextarea = () => {
    const textarea = inputRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      const scrollHeight = textarea.scrollHeight
      const maxHeight = 5 * 24
      textarea.style.height = `${Math.min(scrollHeight, maxHeight)}px`
    }
  }

  const handleNewChat = () => {
    setMessages([])
    setThinkingContent(null)
    setIsThinking(false)
    setInputMessage('')
    const newSessionId = `session_${Date.now()}`;
    sessionIdRef.current = newSessionId;
    setShowSidebar(false)
    inputRef.current?.focus()
  }

  const handleSubmit = useCallback((event?: React.FormEvent) => {
    event?.preventDefault()
    sendMessage();

  }, [inputMessage, isThinking, sessionIdRef])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !isMobile) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="flex h-screen bg-white dark:bg-gray-950 text-gray-900 dark:text-gray-100 antialiased">

      <aside className={`hidden md:flex md:w-64 lg:w-72 flex-col flex-shrink-0 bg-gray-50 dark:bg-gray-900 border-r border-gray-200 dark:border-gray-800`}>
        <div className="h-full flex flex-col">
          <div className="p-4 border-b border-gray-200 dark:border-gray-800 flex items-center flex-shrink-0 space-x-4">
            <div className="flex items-center space-x-2.5 flex-grow">
               <div className="w-8 h-8 rounded-lg overflow-hidden flex items-center justify-center shadow flex-shrink-0">
                 <img src={profileImage} alt="Profile" className="w-full h-full object-cover" />
               </div>
               <h1 className="text-base font-semibold">{t('appTitle')}</h1>
            </div>
            <button
             onClick={() => changeLanguage(i18n.language === 'ko' ? 'en' : 'ko')}
             className="px-2 py-1 mt-1 text-xs text-gray-500 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-md transition-colors border border-gray-300 dark:border-gray-600 flex-shrink-0"
             title="Change Language"
            >
              {getLanguageToggleLabel()}
            </button>
          </div>
          
          <nav className="flex-1 overflow-y-auto p-3 space-y-1">
            <a href="#" className="flex items-center px-3 py-2 rounded-md bg-gray-200 dark:bg-gray-700/60 text-sm font-medium text-gray-800 dark:text-gray-200">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2 text-gray-500 dark:text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                 <path strokeLinecap="round" strokeLinejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
              <span className="truncate">{t('currentChat')} ({messages.length})</span>
            </a>
            <div className="text-center text-xs text-gray-400 dark:text-gray-600 pt-6 px-4">대화 기록 기능은<br/>곧 추가될 예정입니다.</div>
          </nav>
          
          <div className="p-4 border-t border-gray-200 dark:border-gray-800 flex-shrink-0">
            <div className="text-xs text-gray-500 dark:text-gray-400">
              MCP Agent v0.2.0
            </div>
          </div>
        </div>
      </aside>
      
      <div 
        className={`fixed inset-0 bg-black/50 z-30 md:hidden ${showSidebar ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-300 ease-in-out`}
        onClick={() => setShowSidebar(false)}
        aria-hidden="true"
      />
      <aside 
        className={`fixed inset-y-0 left-0 z-40 w-72 bg-gray-50 dark:bg-gray-900 border-r border-gray-200 dark:border-gray-800 transform ${showSidebar ? 'translate-x-0' : '-translate-x-full'} transition-transform duration-300 ease-in-out md:hidden`}
        role="dialog" 
        aria-modal="true"
      >
         <div className="h-full flex flex-col">
           <div className="p-4 border-b border-gray-200 dark:border-gray-800 flex items-center justify-between flex-shrink-0">
            <div className="flex items-center space-x-2.5">
               <div className="w-8 h-8 rounded-lg overflow-hidden flex items-center justify-center shadow flex-shrink-0">
                 <img src={profileImage} alt="Profile" className="w-full h-full object-cover" />
               </div>
               <h1 className="text-base font-semibold">{t('appTitle')}</h1>
            </div>
             <button 
               onClick={() => setShowSidebar(false)}
               className="p-1.5 rounded-md text-gray-500 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500"
               title="닫기"
             >
               <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                 <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
               </svg>
             </button>
           </div>
           <nav className="flex-1 overflow-y-auto p-3 space-y-1">
             <button 
               onClick={() => { handleNewChat(); setShowSidebar(false); }}
               className="flex items-center w-full px-3 py-2 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700/60 text-sm font-medium text-gray-700 dark:text-gray-300"
             >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                   <path strokeLinecap="round" strokeLinejoin="round" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                </svg>
                 Start New Chat
             </button>
             <a href="#" className="flex items-center px-3 py-2 rounded-md bg-gray-200 dark:bg-gray-700/60 text-sm font-medium text-gray-800 dark:text-gray-200">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2 text-gray-500 dark:text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                 <path strokeLinecap="round" strokeLinejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
              <span className="truncate">{t('currentChat')} ({messages.length})</span>
             </a>
             <div className="text-center text-xs text-gray-400 dark:text-gray-600 pt-6 px-4">대화 기록 기능은<br/>곧 추가될 예정입니다.</div>
           </nav>
           <div className="p-4 border-t border-gray-200 dark:border-gray-800 flex-shrink-0">
             <button
               onClick={() => {
                 changeLanguage(i18n.language === 'ko' ? 'en' : 'ko');
                 setShowSidebar(false);
               }}
               className="w-full px-3 py-2 text-sm text-left text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md border border-gray-300 dark:border-gray-600"
             >
               {getLanguageToggleLabel()}
             </button>
             <div className="text-xs text-gray-500 dark:text-gray-400 mt-3">
                MCP Agent v0.2.0
             </div>
           </div>
         </div>
       </aside>

      <main className="flex-1 flex flex-col overflow-hidden">
        <header className="md:hidden border-b border-gray-200 dark:border-gray-800 p-3 flex items-center justify-between flex-shrink-0 bg-white dark:bg-gray-950">
           <button 
             onClick={() => setShowSidebar(true)}
             className="p-1.5 rounded-md text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500"
             title="메뉴 열기"
           >
             <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
               <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
             </svg>
           </button>
           <h1 className="text-base font-semibold truncate px-2">{t('appTitle')}</h1>
            <button 
              onClick={handleNewChat}
              className="p-1.5 rounded-md text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500"
              title="새 대화 시작"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                 <path strokeLinecap="round" strokeLinejoin="round" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
              </svg>
            </button>
        </header>

        <div 
          className="flex-1 overflow-y-auto scroll-smooth" 
          id="message-container"
        >
          {messages.length === 0 && !isThinking && (
             <div className="flex-1 flex flex-col items-center justify-center">
               <EmptyState t={t} setInputMessage={setInputMessage} />
             </div>
          )}
          
          <div className="space-y-4 md:space-y-6 px-4 md:px-6 pt-4 pb-4">
            {messages.map((msg: any) => (
              <MessageComponent 
                key={msg.id}
                message={msg}
              />
            ))}
            
            {isThinking && messages.length > 0 && (
              <div className="flex justify-start mb-4 message-container group">
                <div className="flex max-w-2xl w-full">
                  <div className={`flex items-center justify-center h-8 w-8 rounded-lg flex-shrink-0 mr-3 ml-1 overflow-hidden`}>
                    <img 
                      src={profileImage}
                      alt="AI Profile"
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <ThinkingIndicator content={thinkingContent} t={t} />
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} className="h-0" />
          </div>
      </div>

        <div className="p-3 md:p-4 border-t border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-950 flex-shrink-0">
          <div className="max-w-3xl mx-auto">
            <div className="relative flex items-start space-x-2">
              <textarea
                ref={inputRef}
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={t('inputPlaceholder')}
                className="flex-grow p-2 pr-12 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-1 focus:ring-blue-500 dark:bg-gray-700 dark:text-white resize-none overflow-hidden text-sm leading-tight" 
                rows={1}
                style={{ minHeight: '40px', maxHeight: '120px' }} 
              />
              <button
                onClick={sendMessage} 
                disabled={isThinking || !inputMessage.trim()}
                className={`flex-shrink-0 flex items-center justify-center h-9 w-9 rounded-md p-1 text-white ${isThinking || !inputMessage.trim() ? 'bg-gray-400 dark:bg-gray-600 cursor-not-allowed' : 'bg-blue-500 hover:bg-blue-600 dark:bg-blue-600 dark:hover:bg-blue-700'} transition-colors disabled:opacity-50 mt-[1px]`}
                aria-label={t('sendMessage')}
              >
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" className="w-5 h-5" strokeWidth="2" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 12 3.269 3.126A59.768 59.768 0 0 1 21.485 12 59.77 59.77 0 0 1 3.27 20.876L5.999 12zm0 0h7.5" />
                </svg>
        </button>
            </div>
            <div className="text-xs text-center text-gray-500 dark:text-gray-500 pt-2 px-2">
              {t('disclaimer')}
            </div>
          </div>
        </div>
      </main>

      </div>
  )
}

export default App
