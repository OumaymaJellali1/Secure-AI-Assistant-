import React, { useState } from 'react';
import { makeStyles, mergeClasses } from '@fluentui/react-components';
import { DocumentRegular, ChevronDown16Regular, ChevronUp16Regular } from '@fluentui/react-icons';
import { useUser } from '../context/UserContext';

const F = "'Lexend', 'Segoe UI', sans-serif";

const useStyles = makeStyles({
  row: {
    display: 'flex',
    gap: '12px',
    marginBottom: '14px',
    alignItems: 'flex-start',
    fontFamily: F,
  },
  rowUser: { flexDirection: 'row-reverse' },
  avatar: {
    width: '34px',
    height: '34px',
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '13px',
    fontWeight: '700',
    flexShrink: 0,
    marginTop: '2px',
    fontFamily: F,
  },
  avatarUser: {
    background: 'linear-gradient(135deg, #1a1a2e, #4f46e5)',
    color: '#ffffff',
  },
  avatarAssistant: {
    background: 'linear-gradient(135deg, #0f3460, #1a1a2e)',
    color: '#ffffff',
    fontSize: '15px',
  },
  content: {
    maxWidth: '74%',
    display: 'flex',
    flexDirection: 'column',
    gap: '5px',
  },
  contentUser: { alignItems: 'flex-end' },
  bubble: {
    padding: '12px 16px',
    borderRadius: '16px',
    fontSize: '15px',
    lineHeight: '1.65',
    wordBreak: 'break-word',
    whiteSpace: 'pre-wrap',
    fontFamily: F,
    fontWeight: '400',
  },
  bubbleUser: {
    background: '#1a1a2e',
    color: '#ffffff',
    borderBottomRightRadius: '5px',
  },
  bubbleAssistant: {
    background: '#ffffff',
    color: '#1f2937',
    border: '1px solid rgba(0,0,0,0.08)',
    borderBottomLeftRadius: '5px',
    boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
  },
  bubbleEmpty: { minWidth: '60px', minHeight: '22px' },
  cursor: {
    display: 'inline-block',
    width: '2px',
    height: '14px',
    background: '#4f46e5',
    marginLeft: '2px',
    verticalAlign: 'middle',
    borderRadius: '1px',
    animationName: { '0%,50%': { opacity: 1 }, '51%,100%': { opacity: 0 } },
    animationDuration: '0.9s',
    animationIterationCount: 'infinite',
  },
  meta: {
    fontSize: '12px',
    color: '#9ca3af',
    fontFamily: F,
    paddingLeft: '2px',
    paddingRight: '2px',
    fontWeight: '400',
  },
  sourcesToggle: {
    display: 'flex',
    alignItems: 'center',
    gap: '5px',
    border: 'none',
    background: 'transparent',
    fontSize: '13px',
    color: '#6b7280',
    cursor: 'pointer',
    padding: '3px 6px',
    borderRadius: '6px',
    fontFamily: F,
    fontWeight: '500',
    transition: 'color 0.12s',
    ':hover': { color: '#374151' },
  },
  sourcesList: {
    marginTop: '4px',
    padding: '10px 14px',
    background: '#f9fafb',
    border: '1px solid #e5e7eb',
    borderRadius: '10px',
    display: 'flex',
    flexDirection: 'column',
    gap: '7px',
  },
  sourceItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '7px',
    fontSize: '13px',
    color: '#4b5563',
    fontFamily: F,
  },
  sourceIcon: { fontSize: '14px', color: '#9ca3af', flexShrink: 0 },
});

function formatTime(iso) {
  if (!iso) return '';
  return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function initials(name) {
  if (!name) return '?';
  return name.split(' ').map(p => p[0]).join('').toUpperCase().slice(0, 2);
}

export default function MessageBubble({ message }) {
  const styles = useStyles();
  const { activeUser, activeUserId } = useUser();
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const isUser = message.role === 'user';
  const sources = message.sources || [];
  const isStreaming = !!message.streaming;
  const isEmpty = !message.content;
  const userName = activeUser?.display_name || activeUserId || 'You';

  return (
    <div className={mergeClasses(styles.row, isUser && styles.rowUser)}>
      <div className={mergeClasses(styles.avatar, isUser ? styles.avatarUser : styles.avatarAssistant)}>
        {isUser ? initials(userName) : '🧠'}
      </div>

      <div className={mergeClasses(styles.content, isUser && styles.contentUser)}>
        <div className={mergeClasses(
          styles.bubble,
          isUser ? styles.bubbleUser : styles.bubbleAssistant,
          isEmpty && isStreaming && styles.bubbleEmpty,
        )}>
          {message.content}
          {isStreaming && <span className={styles.cursor} />}
        </div>

        <div className={styles.meta}>
          {isUser ? userName : 'Assistant'} · {formatTime(message.created_at)}
        </div>

        {sources.length > 0 && (
          <div>
            <button className={styles.sourcesToggle} onClick={() => setSourcesOpen(!sourcesOpen)}>
              📎 {sources.length} source{sources.length !== 1 ? 's' : ''}
              {sourcesOpen ? <ChevronUp16Regular style={{ fontSize: 12 }} /> : <ChevronDown16Regular style={{ fontSize: 12 }} />}
            </button>
            {sourcesOpen && (
              <div className={styles.sourcesList}>
                {sources.map((src, i) => (
                  <div key={i} className={styles.sourceItem}>
                    <DocumentRegular className={styles.sourceIcon} />
                    [{i + 1}] {src.source || 'Unknown'}{src.page ? `, p. ${src.page}` : ''}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}