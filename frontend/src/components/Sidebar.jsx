import React, { useState } from 'react';
import { makeStyles } from '@fluentui/react-components';
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
    overflowY: 'auto',
    overflowX: 'hidden',
    paddingRight: '2px',
  },
  divider: {
    height: '1px',
    background: 'rgba(0,0,0,0.07)',
    margin: '8px 0',
  },
  bottomArea: {
    borderTop: '1px solid rgba(0,0,0,0.07)',
    paddingTop: '12px',
    marginTop: '4px',
  },
});

export default function Sidebar({ externalRefreshKey = 0 }) {
  const styles = useStyles();
  const [localRefreshKey, setLocalRefreshKey] = useState(0);
  const refresh = () => setLocalRefreshKey(k => k + 1);
  const combinedKey = localRefreshKey + externalRefreshKey * 1000;

  return (
    <div className={styles.root}>
      <div className={styles.scrollArea}>
        <DocumentsSection refreshKey={combinedKey} />
        <div className={styles.divider} />
        <ConversationsSection refreshKey={combinedKey} onNewChat={refresh} />
      </div>
      <div className={styles.bottomArea}>
        <NewChatButton onCreated={refresh} />
      </div>
    </div>
  );
}
