import React, { useState } from 'react';
import { FluentProvider, webLightTheme, makeStyles } from '@fluentui/react-components';

import { UserProvider } from './context/UserContext';
import { ChatProvider } from './context/ChatContext';
import AuthPage from './components/AuthPage';
import UserSwitcher from './components/UserSwitcher';
import Sidebar from './components/Sidebar';
import ChatPanel from './components/ChatPanel';

// Import DM Sans font
const fontLink = document.createElement('link');
fontLink.rel = 'stylesheet';
fontLink.href = 'https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&display=swap';
document.head.appendChild(fontLink);

const useStyles = makeStyles({
  app: {
    display: 'flex',
    height: '100vh',
    fontFamily: "'DM Sans', 'Segoe UI', sans-serif",
    background: '#F7F6F3',
    overflow: 'hidden',
  },
  sidebar: {
    width: '260px',
    flexShrink: 0,
    display: 'flex',
    flexDirection: 'column',
    background: '#ffffff',
    borderRight: '1px solid rgba(0,0,0,0.08)',
    overflow: 'hidden',
  },
  sidebarHeader: {
    padding: '16px 16px 12px',
    borderBottom: '1px solid rgba(0,0,0,0.07)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexShrink: 0,
  },
  brand: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
  },
  brandLogo: {
    width: '32px',
    height: '32px',
    borderRadius: '9px',
    background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '16px',
    flexShrink: 0,
  },
  brandName: {
    fontSize: '14px',
    fontWeight: '600',
    color: '#1a1a2e',
    letterSpacing: '-0.2px',
    fontFamily: "'DM Sans', sans-serif",
  },
  sidebarBody: {
    flex: 1,
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    padding: '12px 10px',
  },
  main: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    background: '#F7F6F3',
  },
});

function AppShell({ onLogout }) {
  const styles = useStyles();
  const [sidebarRefreshKey, setSidebarRefreshKey] = useState(0);
  const refreshSidebar = () => setSidebarRefreshKey(k => k + 1);

  return (
    <div className={styles.app}>
      {/* SIDEBAR */}
      <div className={styles.sidebar}>
        <div className={styles.sidebarHeader}>
          <div className={styles.brand}>
            <div className={styles.brandLogo}>🧠</div>
            <span className={styles.brandName}>RAG Assistant</span>
          </div>
          <UserSwitcher onLogout={onLogout} />
        </div>
        <div className={styles.sidebarBody}>
          <Sidebar externalRefreshKey={sidebarRefreshKey} />
        </div>
      </div>

      {/* MAIN CONTENT */}
      <div className={styles.main}>
        <ChatPanel onSidebarRefresh={refreshSidebar} />
      </div>
    </div>
  );
}

export default function App() {
  const [authed, setAuthed] = useState(false);
  const [initialUserId, setInitialUserId] = useState('dev_test');

  const handleLogin = (userId) => {
    setInitialUserId(userId);
    setAuthed(true);
  };

  const handleLogout = () => {
    setAuthed(false);
  };

  if (!authed) {
    return (
      <FluentProvider theme={webLightTheme}>
        <AuthPage onLogin={handleLogin} />
      </FluentProvider>
    );
  }

  return (
    <FluentProvider theme={webLightTheme}>
      <UserProvider initialUserId={initialUserId}>
        <ChatProvider>
          <AppShell onLogout={handleLogout} />
        </ChatProvider>
      </UserProvider>
    </FluentProvider>
  );
}
