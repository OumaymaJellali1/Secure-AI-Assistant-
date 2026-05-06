import React, { useState } from 'react';
import Sidebar from './Sidebar';
import ChatPanel from './ChatPanel';
import { useUser } from '../context/UserContext';

export default function MainLayout({ onLogout }) {
  const [sidebarRefreshKey, setSidebarRefreshKey] = useState(0);
  const refreshSidebar = () => setSidebarRefreshKey(k => k + 1);

  return (
    <div style={{
      display: 'flex',
      height: '100vh',
      background: '#f4f6fb',
      fontFamily: "'DM Sans', 'Segoe UI', sans-serif",
      overflow: 'hidden',
    }}>
      <Sidebar externalRefreshKey={sidebarRefreshKey} onLogout={onLogout} />
      <ChatPanel onSidebarRefresh={refreshSidebar} />
    </div>
  );
}
