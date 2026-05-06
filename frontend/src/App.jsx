import React, { useState } from 'react';
import { FluentProvider, webLightTheme, makeStyles } from '@fluentui/react-components';
import { UserProvider, useUser } from './context/UserContext';
import { ChatProvider } from './context/ChatContext';
import AuthPage from './components/AuthPage';
import UserSwitcher from './components/UserSwitcher';
import Sidebar from './components/Sidebar';
import ChatPanel from './components/ChatPanel';

const F = "'Lexend', 'Segoe UI', sans-serif";

const useStyles = makeStyles({
  app: {
    display: 'flex',
    height: '100vh',
    fontFamily: F,
    background: '#F7F6F3',
    overflow: 'hidden',
  },
  sidebar: {
    width: '272px',
    flexShrink: 0,
    display: 'flex',
    flexDirection: 'column',
    background: '#ffffff',
    borderRight: '1px solid rgba(0,0,0,0.08)',
    overflow: 'hidden',
    boxShadow: '2px 0 12px rgba(0,0,0,.04)',
  },
  sidebarHeader: {
    padding: '18px 18px 14px',
    borderBottom: '1px solid rgba(0,0,0,0.07)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexShrink: 0,
    background: '#fafafa',
  },
  brand: {
    display: 'flex',
    alignItems: 'center',
    gap: '11px',
  },
  brandLogo: {
    width: '36px',
    height: '36px',
    borderRadius: '10px',
    background: 'linear-gradient(135deg, #1a1a2e 0%, #4f46e5 100%)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '18px',
    flexShrink: 0,
    boxShadow: '0 4px 12px rgba(79,70,229,.3)',
  },
  brandName: {
    fontSize: '16px',
    fontWeight: '700',
    color: '#1a1a2e',
    letterSpacing: '-0.3px',
    fontFamily: F,
  },
  brandSub: {
    fontSize: '11px',
    color: '#9ca3af',
    fontWeight: '500',
    fontFamily: F,
    letterSpacing: '0.3px',
  },
  sidebarBody: {
    flex: 1,
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    padding: '14px 12px',
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
            <div>
              <div className={styles.brandName}>RAG Assistant</div>
              <div className={styles.brandSub}>AI Knowledge Platform</div>
            </div>
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

// Inner component — has access to UserContext
function AuthGate() {
  const { isLoggedIn, login, logout } = useUser();

  const handleLogin = (userData) => {
    // userData = { user_id, email, display_name, token }
    login(userData);
  };

  const handleLogout = () => {
    logout();
  };

  if (!isLoggedIn) {
    return <AuthPage onLogin={handleLogin} />;
  }

  return (
    <ChatProvider>
      <AppShell onLogout={handleLogout} />
    </ChatProvider>
  );
}

export default function App() {
  return (
    <FluentProvider theme={webLightTheme}>
      <UserProvider>
        <AuthGate />
      </UserProvider>
    </FluentProvider>
  );
}