/**
 * UserContext.jsx — shares the active user across all components.
 *
 * Why context? Because EVERY API call needs to know which user
 * is "logged in." Without context, we'd have to pass userId as a
 * prop down through every component layer.
 *
 * Phase 1 (now): user is picked from dropdown
 * Phase 3 (later): user comes from MS Graph token
 *   → only this file changes, not the rest of the app
 */
import React, { createContext, useContext, useState, useEffect } from 'react';
import api from '../api';

// localStorage key — remembers user choice across page reloads
const STORAGE_KEY = 'rag_active_user_id';

// Default user if nothing in storage
const DEFAULT_USER = 'dev_test';


// ── CONTEXT ────────────────────────────────────────────────────────
const UserContext = createContext(null);


// ── PROVIDER ───────────────────────────────────────────────────────
export function UserProvider({ children }) {
  // Load the active user from localStorage on first render
  const [activeUserId, setActiveUserIdState] = useState(() => {
    return localStorage.getItem(STORAGE_KEY) || DEFAULT_USER;
  });

  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch the list of users on mount
  useEffect(() => {
    let cancelled = false;
    
    async function loadUsers() {
      try {
        setLoading(true);
        setError(null);
        const data = await api.listUsers();
        if (!cancelled) {
          setUsers(data);
          
          // If our active user isn't in the list, fall back to the first one
          const userExists = data.some(u => u.id === activeUserId);
          if (!userExists && data.length > 0) {
            setActiveUserIdState(data[0].id);
            localStorage.setItem(STORAGE_KEY, data[0].id);
          }
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    
    loadUsers();
    return () => { cancelled = true; };
  }, []);

  // Wrapper that also persists to localStorage
  const setActiveUserId = (newUserId) => {
    setActiveUserIdState(newUserId);
    localStorage.setItem(STORAGE_KEY, newUserId);
  };

  const activeUser = users.find(u => u.id === activeUserId);

  return (
    <UserContext.Provider value={{
      activeUserId,
      activeUser,
      setActiveUserId,
      users,
      loading,
      error,
    }}>
      {children}
    </UserContext.Provider>
  );
}


// ── HOOK ──────────────────────────────────────────────────────────
export function useUser() {
  const ctx = useContext(UserContext);
  if (!ctx) {
    throw new Error('useUser must be called inside <UserProvider>');
  }
  return ctx;
}