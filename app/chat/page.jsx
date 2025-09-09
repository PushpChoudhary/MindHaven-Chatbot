'use client';

import { useEffect, useRef, useState } from 'react';

export default function ChatPage() {
  // ✅ DEBUG: Let's check the value of the environment variable
  console.log("Python API URL from chat page:", process.env.NEXT_PUBLIC_PYTHON_API_URL);

  // Initialize sessionId from localStorage or generate new
  const [sessionId, setSessionId] = useState(() => {
    if (typeof window !== 'undefined') {
      let sid = localStorage.getItem('session_id');
      if (!sid) {
        sid = 'sess-' + Math.random().toString(36).substring(2, 11);
        localStorage.setItem('session_id', sid);
      }
      return sid;
    }
    return null;
  });

  const [message, setMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [serverReady, setServerReady] = useState(true);
  const chatEndRef = useRef(null);

  // Scroll chat to bottom when chatHistory changes
  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatHistory]);

  // Check backend health on mount, retry every 5 seconds if not ready
  useEffect(() => {
    let isMounted = true;

    const checkHealth = () => {
      // ✅ FIX: Use backticks (`) instead of single quotes (')
      fetch(`${process.env.NEXT_PUBLIC_PYTHON_API_URL}/health`)
        .then(res => {
            if (res.ok) return res.json();
            throw new Error('Network response was not ok.');
        })
        .then(data => {
          if (isMounted) {
            setServerReady(data.status === 'ready');
          }
        })
        .catch(() => {
          if (isMounted) setServerReady(false);
        });
    };

    checkHealth();
    const intervalId = setInterval(checkHealth, 5000);

    return () => {
      isMounted = false;
      clearInterval(intervalId);
    };
  }, []);

  // Send message handler
  const handleSend = async () => {
    if (!message.trim() || !serverReady || loading) return;

    if (!sessionId) {
      alert('Session ID missing. Please reload the page.');
      return;
    }

    const userMessage = { sender: 'user', text: message };
    setChatHistory(prev => [...prev, userMessage]);
    setLoading(true);

    try {
      // ✅ FIX: Use backticks (`) instead of single quotes (')
      const res = await fetch(`${process.env.NEXT_PUBLIC_PYTHON_API_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: message, session_id: sessionId }),
      });

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`);
      }

      const data = await res.json();

      const botReply = data.response || data.error || 'No response from server.';

      setChatHistory(prev => [...prev, { sender: 'bot', text: botReply }]);
    } catch (error) {
      setChatHistory(prev => [
        ...prev,
        { sender: 'bot', text: '⚠️ Failed to connect to backend.' },
      ]);
    }

    setMessage('');
    setLoading(false);
  };

  // Send on Enter key (except Shift+Enter)
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-start p-6 bg-white">
      <h1 className="text-3xl font-bold mb-4 text-blue-600">🧘 MindHaven Chat</h1>

      {!serverReady && (
        <div className="text-red-600 mb-4">
          ⚠️ Backend not ready. Please wait or try again later.
        </div>
      )}

      <div className="w-full max-w-xl h-[400px] overflow-y-auto bg-gray-100 border border-gray-300 rounded p-4 mb-4 shadow-inner">
        {chatHistory.length === 0 && (
          <p className="text-gray-500 text-center mt-20">
            Start the conversation by typing below...
          </p>
        )}
        {chatHistory.map((msg, idx) => (
          <div
            key={idx}
            className={`mb-2 ${msg.sender === 'user' ? 'text-right' : 'text-left'}`}
          >
            <div
              className={`inline-block px-4 py-2 rounded-lg whitespace-pre-wrap ${
                msg.sender === 'user' ? 'bg-blue-200' : 'bg-green-200'
              }`}
            >
              {msg.text}
            </div>
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>

      <textarea
        className="w-full max-w-xl p-4 border border-gray-300 rounded-lg shadow-sm mb-4 resize-none focus:outline-blue-500"
        rows={3}
        placeholder="Type your thoughts or questions here..."
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={loading || !serverReady}
      />

      <button
        className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700 transition disabled:opacity-50 disabled:cursor-not-allowed"
        onClick={handleSend}
        disabled={loading || !serverReady}
      >
        {loading ? 'Sending...' : 'Send'}
      </button>
    </div>
  );
}

