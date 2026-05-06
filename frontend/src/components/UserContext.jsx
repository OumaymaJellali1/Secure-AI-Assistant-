import React, { createContext, useContext, useState, useEffect } from 'react';
import api from '../api';

const STORAGE_KEY = 'rag_active_user_id';
const DEFAULT_USER = 'dev_test';

const UserContext = createContext(null);

export function UserProvider({ children, initialUserId }) {
  const [activeUserId, setActiveUserIdState] = useState(() => {
    return initialUserId || localStorage.getItem(STORAGE_KEY) || DEFAULT_USER;
  });
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (initialUserId) {
      setActiveUserIdState(initialUserId);
      localStorage.setItem(STORAGE_KEY, initialUserId);
    }
  }, [initialUserId]);

  useEffect(() => {
    let cancelled = false;
    async function loadUsers() {
      try {
        setLoading(true);
        setError(null);
        const data = await api.listUsers();
        if (!cancelled) {
          setUsers(data);
          const userExists = data.some(u => u.id === activeUserId);
          if (!userExists && data.length > 0) {
            setActiveUserIdState(data[0].id);
            localStorage.setItem(STORAGE_KEY, data[0].id);
          }
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    loadUsers();
    return () => { cancelled = true; };
  }, []);

  const setActiveUserId = (newUserId) => {
    setActiveUserIdState(newUserId);
    localStorage.setItem(STORAGE_KEY, newUserId);
  };

  const activeUser = users.find(u => u.id === activeUserId);

  return (
    <UserContext.Provider value={{ activeUserId, activeUser, setActiveUserId, users, loading, error }}>
      {children}
    </UserContext.Provider>
  );
}

export function useUser() {
  const ctx = useContext(UserContext);
  if (!ctx) throw new Error('useUser must be called inside <UserProvider>');
  return ctx;
}
