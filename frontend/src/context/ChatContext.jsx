/**
 * ChatContext.jsx — manages the active conversation (session_id).
 *
 * Why a separate context from UserContext?
 *   • UserContext = WHO is logged in
 *   • ChatContext = WHICH conversation they're viewing
 *   • Switching users should reset the active conversation
 *
 * Provides:
 *   - activeSessionId : currently selected conversation
 *   - setActiveSessionId : function to switch / open a conversation
 *   - clearActive : returns to the empty welcome state
 */
import React, { createContext, useContext, useState, useEffect } from 'react';
import { useUser } from './UserContext';


const ChatContext = createContext(null);


export function ChatProvider({ children }) {
  const { activeUserId } = useUser();
  const [activeSessionId, setActiveSessionId] = useState(null);

  // When the user switches → clear the active conversation
  // (Bob shouldn't see Alice's messages still loaded)
  useEffect(() => {
    setActiveSessionId(null);
  }, [activeUserId]);

  return (
    <ChatContext.Provider value={{
      activeSessionId,
      setActiveSessionId,
      clearActive: () => setActiveSessionId(null),
    }}>
      {children}
    </ChatContext.Provider>
  );
}


export function useChat() {
  const ctx = useContext(ChatContext);
  if (!ctx) {
    throw new Error('useChat must be called inside <ChatProvider>');
  }
  return ctx;
}