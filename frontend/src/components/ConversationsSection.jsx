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
  makeStyles,
  mergeClasses,
} from '@fluentui/react-components';
import {
  ChatRegular,
  MoreHorizontal20Regular,
  DeleteRegular,
  EditRegular,
  Add16Regular,
} from '@fluentui/react-icons';

import api from '../api';
import { useUser } from '../context/UserContext';
import { useChat } from '../context/ChatContext';

const useStyles = makeStyles({
  section: {
    marginTop: '4px',
  },
  sectionHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '6px 4px 4px',
    marginBottom: '2px',
  },
  sectionLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    color: '#9ca3af',
    fontSize: '11px',
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: '0.6px',
    fontFamily: "'DM Sans', sans-serif",
  },
  addBtn: {
    width: '22px',
    height: '22px',
    borderRadius: '6px',
    border: '1.5px solid #e5e7eb',
    background: 'transparent',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    cursor: 'pointer',
    color: '#6b7280',
    transition: 'all 0.15s',
    padding: 0,
    flexShrink: 0,
    ':hover': {
      background: '#f3f4f6',
      borderColor: '#d1d5db',
      color: '#374151',
    },
  },
  groupLabel: {
    padding: '8px 6px 3px',
    color: '#c4c9d4',
    fontSize: '10px',
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
    fontFamily: "'DM Sans', sans-serif",
  },
  emptyMsg: {
    color: '#9ca3af',
    fontSize: '13px',
    textAlign: 'center',
    padding: '16px 8px',
    fontStyle: 'italic',
    lineHeight: '1.5',
  },
  errorMsg: {
    color: '#ef4444',
    fontSize: '12px',
    padding: '8px',
  },
  item: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '7px 8px',
    borderRadius: '8px',
    cursor: 'pointer',
    color: '#374151',
    transition: 'background 0.12s',
    minHeight: '34px',
    ':hover': {
      background: '#f3f4f6',
    },
  },
  itemActive: {
    background: '#eef2ff',
    ':hover': {
      background: '#e0e7ff',
    },
  },
  itemIcon: {
    fontSize: '14px',
    color: '#9ca3af',
    flexShrink: 0,
  },
  itemIconActive: {
    color: '#6366f1',
  },
  itemTitle: {
    flex: 1,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    fontSize: '13px',
    fontFamily: "'DM Sans', sans-serif",
    fontWeight: '400',
  },
  itemTitleActive: {
    fontWeight: '500',
    color: '#4f46e5',
  },
  itemMenu: {
    flexShrink: 0,
    opacity: 0,
    transition: 'opacity 0.1s',
    minWidth: 'auto',
    height: '22px',
    padding: '0 4px',
  },
  itemMenuVisible: {
    opacity: 1,
  },
  loadingBox: {
    display: 'flex',
    justifyContent: 'center',
    padding: '16px',
  },
});

function bucketByDate(conversations) {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000);
  const weekAgo = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);

  const buckets = { 'Today': [], 'Yesterday': [], 'This week': [], 'Older': [] };
  conversations.forEach((c) => {
    const date = new Date(c.last_active);
    if (date >= today) buckets['Today'].push(c);
    else if (date >= yesterday) buckets['Yesterday'].push(c);
    else if (date >= weekAgo) buckets['This week'].push(c);
    else buckets['Older'].push(c);
  });
  return buckets;
}

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
      <ChatRegular className={mergeClasses(styles.itemIcon, isActive && styles.itemIconActive)} />
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
            <MenuItem icon={<EditRegular />} onClick={(e) => { e.stopPropagation(); onRename(conv); }}>
              Rename
            </MenuItem>
            <MenuItem icon={<DeleteRegular />} onClick={(e) => { e.stopPropagation(); onDelete(conv); }}>
              Delete
            </MenuItem>
          </MenuList>
        </MenuPopover>
      </Menu>
    </div>
  );
}

export default function ConversationsSection({ refreshKey, onNewChat }) {
  const styles = useStyles();
  const { activeUserId } = useUser();
  const { activeSessionId, setActiveSessionId, clearActive } = useChat();
  const [conversations, setConversations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

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

  useEffect(() => { loadConversations(); }, [loadConversations, refreshKey]);

  const handleNewChat = async () => {
    try {
      const newConv = await api.createConversation(activeUserId);
      setActiveSessionId(newConv.session_id);
      if (onNewChat) onNewChat();
    } catch (err) {
      alert(`Failed: ${err.message}`);
    }
  };

  const handleRename = async (conv) => {
    const newTitle = prompt('New name:', conv.title || '');
    if (!newTitle?.trim() || newTitle === conv.title) return;
    try {
      await api.renameConversation(activeUserId, conv.session_id, newTitle.trim());
      await loadConversations();
    } catch (err) {
      alert(`Failed to rename: ${err.message}`);
    }
  };

  const handleDelete = async (conv) => {
    if (!confirm(`Delete "${conv.title || '(untitled)'}"?`)) return;
    try {
      await api.deleteConversation(activeUserId, conv.session_id);
      if (conv.session_id === activeSessionId) clearActive();
      await loadConversations();
    } catch (err) {
      alert(`Failed to delete: ${err.message}`);
    }
  };

  return (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <div className={styles.sectionLabel}>
          <ChatRegular style={{ fontSize: 13 }} />
          Conversations
        </div>
        <button className={styles.addBtn} onClick={handleNewChat} title="New conversation">
          <Add16Regular style={{ fontSize: 12 }} />
        </button>
      </div>

      {loading && <div className={styles.loadingBox}><Spinner size="tiny" /></div>}
      {error && <div className={styles.errorMsg}>{error}</div>}

      {!loading && !error && conversations.length === 0 && (
        <Text className={styles.emptyMsg}>No conversations yet.<br />Click + to start.</Text>
      )}

      {!loading && !error && conversations.length > 0 && (
        <>
          {Object.entries(bucketByDate(conversations)).map(([label, items]) => {
            if (!items.length) return null;
            return (
              <div key={label}>
                <div className={styles.groupLabel}>{label}</div>
                {items.map((conv) => (
                  <ConversationItem
                    key={conv.session_id}
                    conv={conv}
                    isActive={conv.session_id === activeSessionId}
                    onClick={() => setActiveSessionId(conv.session_id)}
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
