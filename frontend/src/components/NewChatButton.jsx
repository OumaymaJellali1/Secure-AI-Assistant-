/**
 * NewChatButton.jsx — Creates a new conversation.
 *
 * Click → POST /conversations → set as active → trigger sidebar refresh.
 */
import React, { useState } from 'react';
import {
  Button,
  Spinner,
  tokens,
  makeStyles,
} from '@fluentui/react-components';
import { Add20Regular } from '@fluentui/react-icons';

import api from '../api';
import { useUser } from '../context/UserContext';
import { useChat } from '../context/ChatContext';


const useStyles = makeStyles({
  root: {
    width: '100%',
    justifyContent: 'flex-start',
    backgroundColor: tokens.colorNeutralBackground1,
    border: `1px solid ${tokens.colorNeutralStroke1}`,
    marginBottom: '12px',
    ':hover': {
      backgroundColor: tokens.colorNeutralBackground1Hover,
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
      // Notify parent (sidebar) to refresh the list
      if (onCreated) onCreated();
    } catch (err) {
      alert(`Failed to create conversation: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Button
      className={styles.root}
      icon={loading ? <Spinner size="tiny" /> : <Add20Regular />}
      onClick={handleClick}
      disabled={loading}
    >
      New chat
    </Button>
  );
}