import React, { useState } from 'react';
import { makeStyles, mergeClasses } from '@fluentui/react-components';
import { DocumentRegular, ChevronDown16Regular, ChevronUp16Regular } from '@fluentui/react-icons';
import { useUser } from '../context/UserContext';

const useStyles = makeStyles({
  row: {
    display: 'flex',
    gap: '10px',
    marginBottom: '12px',
    alignItems: 'flex-start',
    fontFamily: "'DM Sans', sans-serif",
  },
  rowUser: {
    flexDirection: 'row-reverse',
  },
  avatar: {
    width: '32px',
    height: '32px',
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '13px',
    fontWeight: '600',
    flexShrink: 0,
    marginTop: '2px',
  },
  avatarUser: {
    background: 'linear-gradient(135deg, #1a1a2e, #4f46e5)',
    color: '#ffffff',
  },
  avatarAssistant: {
    background: 'linear-gradient(135deg, #0f3460, #1a1a2e)',
    color: '#ffffff',
    fontSize: '14px',
  },
  content: {
    maxWidth: '72%',
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  contentUser: {
    alignItems: 'flex-end',
  },
  bubble: {
    padding: '10px 14px',
    borderRadius: '14px',
    fontSize: '14px',
    lineHeight: '1.55',
    wordBreak: 'break-word',
    whiteSpace: 'pre-wrap',
    fontFamily: "'DM Sans', sans-serif",
  },
  bubbleUser: {
    background: '#1a1a2e',
    color: '#ffffff',
    borderBottomRightRadius: '4px',
  },
  bubbleAssistant: {
    background: '#ffffff',
    color: '#1f2937',
    border: '1px solid rgba(0,0,0,0.08)',
    borderBottomLeftRadius: '4px',
    boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
  },
  bubbleEmpty: {
    minWidth: '60px',
    minHeight: '20px',
  },
  cursor: {
    display: 'inline-block',
    width: '2px',
    height: '13px',
    background: '#4f46e5',
    marginLeft: '2px',
    verticalAlign: 'middle',
    borderRadius: '1px',
    animationName: {
      '0%, 50%': { opacity: 1 },
      '51%, 100%': { opacity: 0 },
    },
    animationDuration: '0.9s',
    animationIterationCount: 'infinite',
  },
  meta: {
    fontSize: '11px',
    color: '#9ca3af',
    fontFamily: "'DM Sans', sans-serif",
    paddingLeft: '2px',
    paddingRight: '2px',
  },
  sourcesToggle: {
    display: 'flex',
    alignItems: 'center',
    gap: '5px',
    border: 'none',
    background: 'transparent',
    fontSize: '12px',
    color: '#6b7280',
    cursor: 'pointer',
    padding: '2px 4px',
    borderRadius: '5px',
    fontFamily: "'DM Sans', sans-serif",
    transition: 'color 0.12s',
    ':hover': {
      color: '#374151',
    },
  },
  sourcesList: {
    marginTop: '4px',
    padding: '8px 12px',
    background: '#f9fafb',
    border: '1px solid #e5e7eb',
    borderRadius: '8px',
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  },
  sourceItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    fontSize: '12px',
    color: '#4b5563',
    fontFamily: "'DM Sans', sans-serif",
  },
  sourceIcon: {
    fontSize: '13px',
    color: '#9ca3af',
    flexShrink: 0,
  },
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
