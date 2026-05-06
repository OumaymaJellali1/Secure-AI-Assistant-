import React, { useState } from 'react';
import { Spinner, makeStyles } from '@fluentui/react-components';
import { Add20Regular } from '@fluentui/react-icons';
import api from '../api';
import { useUser } from '../context/UserContext';
import { useChat } from '../context/ChatContext';

const useStyles = makeStyles({
  root: {
    width: '100%',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '10px 12px',
    borderRadius: '10px',
    border: '1.5px dashed #d1d5db',
    background: 'transparent',
    color: '#6b7280',
    fontSize: '13px',
    fontWeight: '500',
    fontFamily: "'DM Sans', sans-serif",
    cursor: 'pointer',
    transition: 'all 0.15s',
    ':hover': {
      background: '#f9fafb',
      borderColor: '#9ca3af',
      color: '#374151',
    },
  },
});

export default function NewChatButton({ onCreated }) {
  const styles = useStyles();
  const { activeUserId } = useUser();
  const { setActiveSessionId } = useChat();
  const [loading, setLoading] = useState(false);

  const handleClick = async () => {
    if (loading) return;
    try {
      setLoading(true);
      const newConv = await api.createConversation(activeUserId);
      setActiveSessionId(newConv.session_id);
      if (onCreated) onCreated();
    } catch (err) {
      alert(`Failed to create conversation: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <button className={styles.root} onClick={handleClick} disabled={loading}>
      {loading ? <Spinner size="tiny" /> : <Add20Regular style={{ fontSize: 16 }} />}
      New conversation
    </button>
  );
}
