import React, { useState, useEffect, useCallback } from 'react';
import {
  Text, Button, Menu, MenuTrigger, MenuPopover, MenuList, MenuItem,
  Spinner, makeStyles, mergeClasses,
} from '@fluentui/react-components';
import {
  ChatRegular, MoreHorizontal20Regular, DeleteRegular,
  EditRegular, Add16Regular, Warning24Regular,
} from '@fluentui/react-icons';

import api from '../api';
import { useUser } from '../context/UserContext';
import { useChat } from '../context/ChatContext';

const F = "'Lexend', 'DM Sans', sans-serif";

const useStyles = makeStyles({
  section: { marginTop: '4px' },
  sectionHeader: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '6px 4px 4px', marginBottom: '2px',
  },
  sectionLabel: {
    display: 'flex', alignItems: 'center', gap: '6px',
    color: '#9ca3af', fontSize: '12px', fontWeight: '700',
    textTransform: 'uppercase', letterSpacing: '0.7px', fontFamily: F,
  },
  addBtn: {
    width: '24px', height: '24px', borderRadius: '7px',
    border: '1.5px solid #e5e7eb', background: 'transparent',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    cursor: 'pointer', color: '#6b7280', transition: 'all 0.15s', padding: 0, flexShrink: 0,
    ':hover': { background: '#f3f4f6', borderColor: '#d1d5db', color: '#374151' },
  },
  groupLabel: {
    padding: '8px 6px 3px', color: '#c4c9d4', fontSize: '10px',
    fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.6px', fontFamily: F,
  },
  emptyMsg: {
    color: '#9ca3af', fontSize: '13px', textAlign: 'center',
    padding: '16px 8px', fontStyle: 'italic', lineHeight: '1.5', fontFamily: F,
  },
  item: {
    display: 'flex', alignItems: 'center', gap: '8px',
    padding: '8px 8px', borderRadius: '9px', cursor: 'pointer',
    color: '#374151', transition: 'background 0.12s', minHeight: '36px',
    ':hover': { background: '#f3f4f6' },
  },
  itemActive: {
    background: '#eef2ff',
    ':hover': { background: '#e0e7ff' },
  },
  itemIcon: { fontSize: '14px', color: '#9ca3af', flexShrink: 0 },
  itemIconActive: { color: '#6366f1' },
  itemTitle: {
    flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
    fontSize: '13px', fontFamily: F, fontWeight: '400',
  },
  itemTitleActive: { fontWeight: '500', color: '#4f46e5' },
  itemMenu: {
    flexShrink: 0, opacity: 0, transition: 'opacity 0.1s',
    minWidth: 'auto', height: '22px', padding: '0 4px',
  },
  itemMenuVisible: { opacity: 1 },
  loadingBox: { display: 'flex', justifyContent: 'center', padding: '16px' },
});

/* ── DELETE MODAL ──────────────────────────────────────────────────────────── */
function DeleteModal({ conv, onConfirm, onCancel }) {
  const [confirming, setConfirming] = useState(false);

  const handleDelete = async () => {
    setConfirming(true);
    await onConfirm();
    setConfirming(false);
  };

  return (
    <>
      <style>{`
        @keyframes modalOverlayIn { from{opacity:0} to{opacity:1} }
        @keyframes modalCardIn { from{opacity:0;transform:scale(.93) translateY(-12px)} to{opacity:1;transform:scale(1) translateY(0)} }
        @keyframes warnPulse { 0%,100%{transform:scale(1)} 50%{transform:scale(1.08)} }
      `}</style>
      <div onClick={onCancel} style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,.45)',
        backdropFilter: 'blur(4px)', zIndex: 1000,
        animation: 'modalOverlayIn .2s ease',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        <div onClick={e => e.stopPropagation()} style={{
          background: '#fff', borderRadius: '20px', padding: '32px 32px 28px',
          width: '100%', maxWidth: '400px', margin: '0 16px',
          boxShadow: '0 24px 64px rgba(0,0,0,.2), 0 0 0 1px rgba(0,0,0,.06)',
          animation: 'modalCardIn .3s cubic-bezier(.34,1.56,.64,1)',
          fontFamily: F,
        }}>
          <div style={{
            width: 60, height: 60, borderRadius: '50%',
            background: 'linear-gradient(135deg, #fef2f2, #fee2e2)',
            border: '2px solid #fecaca', display: 'flex', alignItems: 'center',
            justifyContent: 'center', marginBottom: 20, animation: 'warnPulse 2s ease-in-out infinite',
          }}>
            <Warning24Regular style={{ fontSize: 28, color: '#dc2626' }} />
          </div>
          <h3 style={{ margin: '0 0 8px', fontSize: 20, fontWeight: 700, color: '#111827', letterSpacing: '-0.3px' }}>
            Delete conversation?
          </h3>
          <p style={{ margin: '0 0 8px', fontSize: 15, color: '#6b7280', lineHeight: 1.6 }}>
            You're about to permanently delete:
          </p>
          <div style={{
            background: '#f9fafb', border: '1.5px solid #e5e7eb', borderRadius: 10,
            padding: '10px 14px', marginBottom: 20, fontSize: 14,
            color: '#374151', fontWeight: 600,
          }}>
            "{conv?.title || '(untitled)'}"
          </div>
          <p style={{ margin: '0 0 24px', fontSize: 14, color: '#9ca3af', lineHeight: 1.5 }}>
            All messages will be permanently removed. This action <strong style={{ color: '#dc2626' }}>cannot be undone</strong>.
          </p>
          <div style={{ display: 'flex', gap: 10 }}>
            <button onClick={onCancel} style={{
              flex: 1, padding: '13px', borderRadius: 12, border: '2px solid #e5e7eb',
              background: '#fff', color: '#374151', fontSize: 15, fontWeight: 600,
              cursor: 'pointer', fontFamily: F, transition: 'all .15s',
            }}>Cancel</button>
            <button onClick={handleDelete} disabled={confirming} style={{
              flex: 1, padding: '13px', borderRadius: 12, border: 'none',
              background: confirming ? '#fca5a5' : 'linear-gradient(135deg, #dc2626, #ef4444)',
              color: '#fff', fontSize: 15, fontWeight: 700,
              cursor: confirming ? 'not-allowed' : 'pointer', fontFamily: F,
              transition: 'all .15s', display: 'flex', alignItems: 'center',
              justifyContent: 'center', gap: 8,
              boxShadow: '0 4px 14px rgba(220,38,38,.35)',
            }}>
              {confirming ? <Spinner size="tiny" appearance="inverted" /> : <DeleteRegular />}
              {confirming ? 'Deleting…' : 'Delete Forever'}
            </button>
          </div>
        </div>
      </div>
    </>
  );
}

/* ── RENAME MODAL ──────────────────────────────────────────────────────────── */
function RenameModal({ conv, onConfirm, onCancel }) {
  const [value, setValue] = useState(conv?.title || '');
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    if (!value.trim()) return;
    setSaving(true);
    await onConfirm(value.trim());
    setSaving(false);
  };

  return (
    <div onClick={onCancel} style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,.45)',
      backdropFilter: 'blur(4px)', zIndex: 1000,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
    }}>
      <div onClick={e => e.stopPropagation()} style={{
        background: '#fff', borderRadius: '20px', padding: '32px',
        width: '100%', maxWidth: '380px', margin: '0 16px',
        boxShadow: '0 24px 64px rgba(0,0,0,.2)',
        animation: 'modalCardIn .3s cubic-bezier(.34,1.56,.64,1)',
        fontFamily: F,
      }}>
        <h3 style={{ margin: '0 0 6px', fontSize: 20, fontWeight: 700, color: '#111827' }}>Rename conversation</h3>
        <p style={{ margin: '0 0 20px', fontSize: 14, color: '#9ca3af' }}>Give this conversation a new name.</p>
        <input
          autoFocus value={value} onChange={e => setValue(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') handleSave(); if (e.key === 'Escape') onCancel(); }}
          placeholder="Conversation name…"
          style={{
            width: '100%', padding: '13px 16px', borderRadius: 12,
            border: '2px solid #e5e7eb', fontSize: 15, fontFamily: F,
            outline: 'none', marginBottom: 20, boxSizing: 'border-box', transition: 'border-color .2s',
          }}
          onFocus={e => { e.currentTarget.style.borderColor = '#6366f1'; }}
          onBlur={e => { e.currentTarget.style.borderColor = '#e5e7eb'; }}
        />
        <div style={{ display: 'flex', gap: 10 }}>
          <button onClick={onCancel} style={{
            flex: 1, padding: '12px', borderRadius: 12, border: '2px solid #e5e7eb',
            background: '#fff', color: '#374151', fontSize: 14, fontWeight: 600,
            cursor: 'pointer', fontFamily: F,
          }}>Cancel</button>
          <button onClick={handleSave} disabled={!value.trim() || saving} style={{
            flex: 1, padding: '12px', borderRadius: 12, border: 'none',
            background: value.trim() ? 'linear-gradient(135deg, #1a1a2e, #4f46e5)' : '#e5e7eb',
            color: value.trim() ? '#fff' : '#9ca3af', fontSize: 14, fontWeight: 700,
            cursor: value.trim() ? 'pointer' : 'not-allowed', fontFamily: F,
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
          }}>
            {saving ? <Spinner size="tiny" appearance="inverted" /> : null}
            Save
          </button>
        </div>
      </div>
    </div>
  );
}

function bucketByDate(conversations) {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today.getTime() - 86400000);
  const weekAgo = new Date(today.getTime() - 7 * 86400000);
  const buckets = { 'Today': [], 'Yesterday': [], 'This week': [], 'Older': [] };
  conversations.forEach(c => {
    const d = new Date(c.last_active);
    if (d >= today) buckets['Today'].push(c);
    else if (d >= yesterday) buckets['Yesterday'].push(c);
    else if (d >= weekAgo) buckets['This week'].push(c);
    else buckets['Older'].push(c);
  });
  return buckets;
}

function ConversationItem({ conv, isActive, onClick, onRename, onDelete }) {
  const styles = useStyles();
  const [hovered, setHovered] = useState(false);
  return (
    <div
      className={mergeClasses(styles.item, isActive && styles.itemActive)}
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      title={conv.title || '(untitled)'}>
      <ChatRegular className={mergeClasses(styles.itemIcon, isActive && styles.itemIconActive)} />
      <span className={mergeClasses(styles.itemTitle, isActive && styles.itemTitleActive)}>
        {conv.title || '(untitled)'}
      </span>
      <Menu>
        <MenuTrigger disableButtonEnhancement>
          <Button
            className={mergeClasses(styles.itemMenu, hovered && styles.itemMenuVisible)}
            appearance="subtle" size="small"
            icon={<MoreHorizontal20Regular />}
            onClick={(e) => e.stopPropagation()} />
        </MenuTrigger>
        <MenuPopover>
          <MenuList>
            <MenuItem icon={<EditRegular />} onClick={(e) => { e.stopPropagation(); onRename(conv); }}>Rename</MenuItem>
            <MenuItem icon={<DeleteRegular />} onClick={(e) => { e.stopPropagation(); onDelete(conv); }}>Delete</MenuItem>
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
  const [deleteModal, setDeleteModal] = useState(null);
  const [renameModal, setRenameModal] = useState(null);

  const loadConversations = useCallback(async () => {
    if (!activeUserId) return;
    try {
      setLoading(true);
      setError(null);
      const data = await api.listConversations();   // ← no userId
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
      const newConv = await api.createConversation();   // ← no userId
      setActiveSessionId(newConv.session_id);
      if (onNewChat) onNewChat();
    } catch (err) {
      alert(`Failed: ${err.message}`);
    }
  };

  const handleConfirmDelete = async () => {
    if (!deleteModal) return;
    try {
      await api.deleteConversation(deleteModal.session_id);   // ← no userId
      if (deleteModal.session_id === activeSessionId) clearActive();
      await loadConversations();
    } catch (err) {
      alert(`Failed to delete: ${err.message}`);
    } finally {
      setDeleteModal(null);
    }
  };

  const handleConfirmRename = async (newTitle) => {
    if (!renameModal) return;
    try {
      await api.renameConversation(renameModal.session_id, newTitle);   // ← no userId
      await loadConversations();
    } catch (err) {
      alert(`Failed to rename: ${err.message}`);
    } finally {
      setRenameModal(null);
    }
  };

  return (
    <>
      {deleteModal && <DeleteModal conv={deleteModal} onConfirm={handleConfirmDelete} onCancel={() => setDeleteModal(null)} />}
      {renameModal && <RenameModal conv={renameModal} onConfirm={handleConfirmRename} onCancel={() => setRenameModal(null)} />}

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
        {error && <div style={{ color: '#ef4444', fontSize: 12, padding: '8px', fontFamily: F }}>{error}</div>}

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
                  {items.map(conv => (
                    <ConversationItem
                      key={conv.session_id}
                      conv={conv}
                      isActive={conv.session_id === activeSessionId}
                      onClick={() => setActiveSessionId(conv.session_id)}
                      onRename={(c) => setRenameModal(c)}
                      onDelete={(c) => setDeleteModal(c)} />
                  ))}
                </div>
              );
            })}
          </>
        )}
      </div>
    </>
  );
}