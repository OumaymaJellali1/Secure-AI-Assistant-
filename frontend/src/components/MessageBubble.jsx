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
    cursor: 'pointer',
    textDecoration: 'none',
    borderRadius: '6px',
    padding: '2px 4px',
    transition: 'background 0.12s, color 0.12s',
    ':hover': { background: '#f3f4f6', color: '#1f2937' },
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

/**
 * Strips the internal doc ID prefix from a filename.
 * "doc_1317132caea5_test3.pptx" → "test3.pptx"
 */
function cleanSourceName(raw) {
  if (!raw) return 'Unknown';
  return raw.replace(/^doc_[a-f0-9]+_/i, '');
}

/**
 * Replaces [Source: filename, page X] citations in the answer text
 * with a small clean inline badge showing just the human-readable filename.
 */
function renderContentWithCitations(content) {
  if (!content) return null;

  const pattern = /\[Source:\s*([^\],]+?)(?:,\s*page\s+(\d+))?\]/gi;
  const parts = [];
  let lastIndex = 0;
  let match;

  while ((match = pattern.exec(content)) !== null) {
    // Plain text before this citation
    if (match.index > lastIndex) {
      parts.push(content.slice(lastIndex, match.index));
    }
    const name = cleanSourceName(match[1].trim());
    const page = match[2] ? `, p. ${match[2]}` : '';
    parts.push(
      <span
        key={match.index}
        style={{
          fontSize: '11.5px',
          color: '#6b7280',
          fontStyle: 'italic',
          marginLeft: '2px',
        }}
      >
        [{name}{page}]
      </span>
    );
    lastIndex = pattern.lastIndex;
  }

  if (lastIndex < content.length) {
    parts.push(content.slice(lastIndex));
  }

  return parts;
}

export default function MessageBubble({ message }) {
  const styles = useStyles();
  const { activeUser, activeUserId ,token } = useUser();
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
          {isUser
            ? message.content
            : renderContentWithCitations(message.content)}
          {isStreaming && <span className={styles.cursor} />}
        </div>

        <div className={styles.meta}>
          {isUser ? userName : 'Assistant'} · {formatTime(message.created_at)}
        </div>

        {sources.length > 0 && (
          <div>
            <button
              className={styles.sourcesToggle}
              onClick={() => setSourcesOpen(!sourcesOpen)}
            >
              📎 {sources.length} source{sources.length !== 1 ? 's' : ''}
              {sourcesOpen
                ? <ChevronUp16Regular style={{ fontSize: 12 }} />
                : <ChevronDown16Regular style={{ fontSize: 12 }} />}
            </button>

            {sourcesOpen && (
              <div className={styles.sourcesList}>
                {sources.map((src, i) => {
                  const displayName = cleanSourceName(src.source);
                  const page = src.page ? `, p. ${src.page}` : '';

                  // src.url  → direct link your backend provides (preferred)
                  // src.path → fallback: served via a dedicated file endpoint
                  // Without either, render non-clickable.
                  const fileUrl = src.document_id
  ? `/api/documents/${src.document_id}/download?token=${encodeURIComponent(token || '')}`
  : null;

                  return fileUrl ? (
                    <a
                      key={i}
                      className={styles.sourceItem}
                      href={fileUrl}
                      target="_blank"          // opens in new tab, not inside your app
                      rel="noopener noreferrer"
                      title={`Open ${displayName}`}
                    >
                      <DocumentRegular className={styles.sourceIcon} />
                      {displayName}{page}
                    </a>
                  ) : (
                    <div key={i} className={styles.sourceItem} style={{ cursor: 'default' }}>
                      <DocumentRegular className={styles.sourceIcon} />
                      {displayName}{page}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}