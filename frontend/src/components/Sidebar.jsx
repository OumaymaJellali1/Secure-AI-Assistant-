/**
 * Sidebar.jsx — Left sidebar with:
 *   1. New Chat button
 *   2. Documents section (Phase E — NEW)
 *   3. Conversations section
 *
 * Documents section comes BEFORE conversations because:
 *   • Documents are persistent assets (you reference them often)
 *   • Conversations are activity logs (longer list, scrollable)
 */
import React, { useState } from 'react';
import {
  tokens,
  makeStyles,
} from '@fluentui/react-components';

import NewChatButton from './NewChatButton';
import DocumentsSection from './DocumentsSection';
import ConversationsSection from './ConversationsSection';


const useStyles = makeStyles({
  root: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    overflow: 'hidden',
  },
  scrollArea: {
    flex: 1,
    overflow: 'auto',
    paddingRight: '4px',
  },
  divider: {
    height: '1px',
    backgroundColor: tokens.colorNeutralStroke2,
    margin: '12px 0',
  },
});


export default function Sidebar({ externalRefreshKey = 0 }) {
  const styles = useStyles();
  const [localRefreshKey, setLocalRefreshKey] = useState(0);
  const refresh = () => setLocalRefreshKey(k => k + 1);

  // Combined key: any refresh trigger updates downstream
  const combinedKey = localRefreshKey + externalRefreshKey * 1000;

  return (
    <div className={styles.root}>
      <NewChatButton onCreated={refresh} />
      
      <div className={styles.scrollArea}>
        <DocumentsSection refreshKey={combinedKey} />
        <div className={styles.divider} />
        <ConversationsSection refreshKey={combinedKey} />
      </div>
    </div>
  );
}