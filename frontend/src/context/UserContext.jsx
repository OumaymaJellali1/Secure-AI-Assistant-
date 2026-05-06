import React, { createContext, useContext, useState } from 'react';

const TOKEN_KEY = 'rag_auth_token';
const USER_ID_KEY = 'rag_user_id';
const USER_NAME_KEY = 'rag_display_name';
const USER_EMAIL_KEY = 'rag_email';

const UserContext = createContext(null);

export function UserProvider({ children }) {
  const [token, setTokenState] = useState(() => localStorage.getItem(TOKEN_KEY) || '');
  const [activeUserId, setActiveUserIdState] = useState(() => localStorage.getItem(USER_ID_KEY) || '');
  const [displayName, setDisplayNameState] = useState(() => localStorage.getItem(USER_NAME_KEY) || '');
  const [email, setEmailState] = useState(() => localStorage.getItem(USER_EMAIL_KEY) || '');

  const login = (userData) => {
    localStorage.setItem(TOKEN_KEY, userData.token);
    localStorage.setItem(USER_ID_KEY, userData.user_id);
    localStorage.setItem(USER_NAME_KEY, userData.display_name);
    localStorage.setItem(USER_EMAIL_KEY, userData.email || '');
    setTokenState(userData.token);
    setActiveUserIdState(userData.user_id);
    setDisplayNameState(userData.display_name);
    setEmailState(userData.email || '');
  };

  const logout = () => {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_ID_KEY);
    localStorage.removeItem(USER_NAME_KEY);
    localStorage.removeItem(USER_EMAIL_KEY);
    setTokenState('');
    setActiveUserIdState('');
    setDisplayNameState('');
    setEmailState('');
  };

  const isLoggedIn = Boolean(token && activeUserId);

  return (
    <UserContext.Provider value={{
      token, activeUserId, displayName, email, isLoggedIn, login, logout,
      activeUser: isLoggedIn ? { id: activeUserId, display_name: displayName, email } : null,
    }}>
      {children}
    </UserContext.Provider>
  );
}

export function useUser() {
  const ctx = useContext(UserContext);
  if (!ctx) throw new Error('useUser must be called inside <UserProvider>');
  return ctx;
}
