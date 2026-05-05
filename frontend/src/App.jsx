import React, { useState } from 'react';
import {
  FluentProvider,
  webLightTheme,
  webDarkTheme,
  Button,
  Text,
  tokens,
  makeStyles,
} from '@fluentui/react-components';
import {
  WeatherSunny24Regular,
  WeatherMoon24Regular,
  ChatMultiple24Regular,
} from '@fluentui/react-icons';

import { UserProvider } from './context/UserContext';
import { ChatProvider } from './context/ChatContext';
import UserSwitcher from './components/UserSwitcher';
import Sidebar from './components/Sidebar';
import ChatPanel from './components/ChatPanel';


const useStyles = makeStyles({
  root: {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    backgroundColor: tokens.colorNeutralBackground2,
    color: tokens.colorNeutralForeground1,
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '8px 20px',
    backgroundColor: tokens.colorNeutralBackground1,
    borderBottom: `1px solid ${tokens.colorNeutralStroke2}`,
    minHeight: '52px',
  },
  brand: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
  },
  brandIcon: {
    fontSize: '24px',
    color: tokens.colorBrandForeground1,
  },
  brandText: {
    fontWeight: 600,
    fontSize: '16px',
  },
  headerActions: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  body: {
    display: 'flex',
    flex: 1,
    overflow: 'hidden',
  },
  sidebar: {
    width: '280px',
    backgroundColor: tokens.colorNeutralBackground1,
    borderRight: `1px solid ${tokens.colorNeutralStroke2}`,
    padding: '16px',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
  },
});


function App() {
  const [isDark, setIsDark] = useState(false);
  const theme = isDark ? webDarkTheme : webLightTheme;

  return (
    <FluentProvider theme={theme}>
      <UserProvider>
        <ChatProvider>
          <AppContent isDark={isDark} setIsDark={setIsDark} />
        </ChatProvider>
      </UserProvider>
    </FluentProvider>
  );
}


function AppContent({ isDark, setIsDark }) {
  const styles = useStyles();
  // Bumping this triggers the sidebar to re-fetch conversations
  // (used when chat panel sends a message and gets an auto-title)
  const [sidebarRefreshKey, setSidebarRefreshKey] = useState(0);
  const refreshSidebar = () => setSidebarRefreshKey(k => k + 1);

  return (
    <div className={styles.root}>
      {/* HEADER */}
      <header className={styles.header}>
        <div className={styles.brand}>
          <ChatMultiple24Regular className={styles.brandIcon} />
          <Text className={styles.brandText}>RAG Assistant</Text>
        </div>

        <div className={styles.headerActions}>
          <Button
            appearance="subtle"
            icon={isDark ? <WeatherSunny24Regular /> : <WeatherMoon24Regular />}
            onClick={() => setIsDark(!isDark)}
            title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
          />
          <UserSwitcher />
        </div>
      </header>

      {/* BODY */}
      <div className={styles.body}>
        <aside className={styles.sidebar}>
          <Sidebar externalRefreshKey={sidebarRefreshKey} />
        </aside>

        <ChatPanel onSidebarRefresh={refreshSidebar} />
      </div>
    </div>
  );
}

export default App;