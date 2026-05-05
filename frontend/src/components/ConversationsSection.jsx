/**
 * ConversationsSection.jsx — Sidebar list of the user's conversations.
 *
 * Features:
 *   • Groups by date (Today / Yesterday / This week / Older)
 *   • Click → opens the conversation
 *   • Highlights the active one
 *   • Right-click / hover menu → rename / delete
 *   • Auto-refreshes when user switches or new chat is created
 */
import React, { useState, useEffect, useCallback } from 'react';
import {
  Text,
  Button,
  Menu,
  MenuTrigger,
  MenuPopover,
  MenuList,
  MenuItem,
  Spinner,
  tokens,
  makeStyles,
  mergeClasses,
} from '@fluentui/react-components';
import {
  ChatRegular,
  MoreHorizontal20Regular,
  DeleteRegular,
  EditRegular,
} from '@fluentui/react-icons';

import api from '../api';
import { useUser } from '../context/UserContext';
import { useChat } from '../context/ChatContext';


const useStyles = makeStyles({
  section: {
    marginTop: '8px',
  },
  sectionHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    padding: '8px 4px 4px',
    color: tokens.colorNeutralForeground3,
    fontSize: '11px',
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  groupLabel: {
    padding: '12px 4px 4px',
    color: tokens.colorNeutralForeground3,
    fontSize: '11px',
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  emptyMsg: {
    color: tokens.colorNeutralForeground3,
    fontSize: '13px',
    textAlign: 'center',
    padding: '20px 8px',
    fontStyle: 'italic',
  },
  errorMsg: {
    color: tokens.colorPaletteRedForeground1,
    fontSize: '12px',
    padding: '8px',
  },
  
  // Conversation item
  item: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '8px 10px',
    borderRadius: '6px',
    cursor: 'pointer',
    color: tokens.colorNeutralForeground1,
    transitionProperty: 'background-color',
    transitionDuration: '0.1s',
    minHeight: '36px',
    ':hover': {
      backgroundColor: tokens.colorNeutralBackground1Hover,
    },
  },
  itemActive: {
    backgroundColor: tokens.colorBrandBackground2,
    ':hover': {
      backgroundColor: tokens.colorBrandBackground2Hover,
    },
  },
  itemIcon: {
    fontSize: '16px',
    color: tokens.colorNeutralForeground3,
    flexShrink: 0,
  },
  itemTitle: {
    flex: 1,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    fontSize: '13px',
  },
  itemTitleActive: {
    fontWeight: 600,
    color: tokens.colorBrandForeground1,
  },
  itemMenu: {
    flexShrink: 0,
    opacity: 0,
    transitionProperty: 'opacity',
    transitionDuration: '0.1s',
  },
  itemMenuVisible: {
    opacity: 1,
  },
  loadingBox: {
    display: 'flex',
    justifyContent: 'center',
    padding: '20px',
  },
});


// ── DATE BUCKETING ─────────────────────────────────────────────────
function bucketByDate(conversations) {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000);
  const weekAgo = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);

  const buckets = {
    'Today': [],
    'Yesterday': [],
    'This week': [],
    'Older': [],
  };

  conversations.forEach((c) => {
    const date = new Date(c.last_active);
    if (date >= today) buckets['Today'].push(c);
    else if (date >= yesterday) buckets['Yesterday'].push(c);
    else if (date >= weekAgo) buckets['This week'].push(c);
    else buckets['Older'].push(c);
  });

  return buckets;
}


// ── INDIVIDUAL CONVERSATION ITEM ───────────────────────────────────
function ConversationItem({ conv, isActive, onClick, onRename, onDelete }) {
  const styles = useStyles();
  const [hovered, setHovered] = useState(false);
  const title = conv.title || '(untitled)';

  return (
    <div
      className={mergeClasses(styles.item, isActive && styles.itemActive)}
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      title={title}
    >
      <ChatRegular className={styles.itemIcon} />
      <span className={mergeClasses(styles.itemTitle, isActive && styles.itemTitleActive)}>
        {title}
      </span>

      <Menu>
        <MenuTrigger disableButtonEnhancement>
          <Button
            className={mergeClasses(styles.itemMenu, hovered && styles.itemMenuVisible)}
            appearance="subtle"
            size="small"
            icon={<MoreHorizontal20Regular />}
            onClick={(e) => e.stopPropagation()}
          />
        </MenuTrigger>
        <MenuPopover>
          <MenuList>
            <MenuItem
              icon={<EditRegular />}
              onClick={(e) => {
                e.stopPropagation();
                onRename(conv);
              }}
            >
              Rename
            </MenuItem>
            <MenuItem
              icon={<DeleteRegular />}
              onClick={(e) => {
                e.stopPropagation();
                onDelete(conv);
              }}
            >
              Delete
            </MenuItem>
          </MenuList>
        </MenuPopover>
      </Menu>
    </div>
  );
}


// ── MAIN COMPONENT ─────────────────────────────────────────────────
export default function ConversationsSection({ refreshKey }) {
  const styles = useStyles();
  const { activeUserId } = useUser();
  const { activeSessionId, setActiveSessionId, clearActive } = useChat();

  const [conversations, setConversations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // ── Load conversations whenever user changes or refreshKey bumps ──
  const loadConversations = useCallback(async () => {
    if (!activeUserId) return;
    
    try {
      setLoading(true);
      setError(null);
      const data = await api.listConversations(activeUserId);
      setConversations(data || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [activeUserId]);

  useEffect(() => {
    loadConversations();
  }, [loadConversations, refreshKey]);

  // ── Handlers ─────────────────────────────────────────────────────
  const handleOpen = (sessionId) => {
    setActiveSessionId(sessionId);
  };

  const handleRename = async (conv) => {
    const currentTitle = conv.title || '';
    const newTitle = prompt('New name:', currentTitle);
    if (newTitle == null || newTitle.trim() === '' || newTitle === currentTitle) {
      return;
    }
    
    try {
      await api.renameConversation(activeUserId, conv.session_id, newTitle.trim());
      await loadConversations();
    } catch (err) {
      alert(`Failed to rename: ${err.message}`);
    }
  };

  const handleDelete = async (conv) => {
    const title = conv.title || '(untitled)';
    if (!confirm(`Delete conversation "${title}"? This cannot be undone.`)) {
      return;
    }
    
    try {
      await api.deleteConversation(activeUserId, conv.session_id);
      // If we deleted the active one, clear it
      if (conv.session_id === activeSessionId) {
        clearActive();
      }
      await loadConversations();
    } catch (err) {
      alert(`Failed to delete: ${err.message}`);
    }
  };

  // ── Render ───────────────────────────────────────────────────────
  return (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <ChatRegular style={{ fontSize: 14 }} />
        Conversations
      </div>

      {loading && (
        <div className={styles.loadingBox}>
          <Spinner size="tiny" />
        </div>
      )}

      {error && (
        <div className={styles.errorMsg}>
          {error}
        </div>
      )}

      {!loading && !error && conversations.length === 0 && (
        <Text className={styles.emptyMsg}>
          No conversations yet.<br />
          Click "New chat" to start.
        </Text>
      )}

      {!loading && !error && conversations.length > 0 && (
        <>
          {Object.entries(bucketByDate(conversations)).map(([label, items]) => {
            if (items.length === 0) return null;
            return (
              <div key={label}>
                <Text className={styles.groupLabel}>{label}</Text>
                {items.map((conv) => (
                  <ConversationItem
                    key={conv.session_id}
                    conv={conv}
                    isActive={conv.session_id === activeSessionId}
                    onClick={() => handleOpen(conv.session_id)}
                    onRename={handleRename}
                    onDelete={handleDelete}
                  />
                ))}
              </div>
            );
          })}
        </>
      )}
    </div>
  );
}