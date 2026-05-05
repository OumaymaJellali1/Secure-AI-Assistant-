/**
 * MessageBubble.jsx — One message (user or assistant).
 *
 * NEW: shows a blinking cursor at end of streaming messages
 */
import React, { useState } from 'react';
import {
  Avatar,
  Text,
  Button,
  tokens,
  makeStyles,
  mergeClasses,
} from '@fluentui/react-components';
import {
  Bot24Regular,
  DocumentRegular,
  ChevronDown16Regular,
  ChevronUp16Regular,
} from '@fluentui/react-icons';

import { useUser } from '../context/UserContext';


const useStyles = makeStyles({
  row: {
    display: 'flex',
    gap: '12px',
    marginBottom: '20px',
    alignItems: 'flex-start',
  },
  rowUser: {
    flexDirection: 'row-reverse',
  },
  avatar: {
    flexShrink: 0,
  },
  assistantIcon: {
    width: '36px',
    height: '36px',
    borderRadius: '50%',
    backgroundColor: tokens.colorBrandBackground,
    color: tokens.colorNeutralForegroundOnBrand,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
  },
  
  bubble: {
    maxWidth: '70%',
    padding: '12px 16px',
    borderRadius: '12px',
    wordBreak: 'break-word',
    lineHeight: '1.5',
    fontSize: '14px',
    whiteSpace: 'pre-wrap',
  },
  bubbleUser: {
    backgroundColor: tokens.colorBrandBackground,
    color: tokens.colorNeutralForegroundOnBrand,
    borderBottomRightRadius: '4px',
  },
  bubbleAssistant: {
    backgroundColor: tokens.colorNeutralBackground1,
    color: tokens.colorNeutralForeground1,
    border: `1px solid ${tokens.colorNeutralStroke2}`,
    borderBottomLeftRadius: '4px',
  },
  bubbleEmpty: {
    minWidth: '80px',
    minHeight: '24px',
  },
  
  // Blinking cursor (for streaming)
  cursor: {
    display: 'inline-block',
    width: '2px',
    height: '14px',
    backgroundColor: tokens.colorBrandForeground1,
    marginLeft: '2px',
    verticalAlign: 'middle',
    animationName: {
      '0%, 50%': { opacity: 1 },
      '51%, 100%': { opacity: 0 },
    },
    animationDuration: '1s',
    animationIterationCount: 'infinite',
  },
  
  meta: {
    fontSize: '11px',
    color: tokens.colorNeutralForeground3,
    marginTop: '4px',
  },
  
  sourcesContainer: {
    marginTop: '8px',
    width: '70%',
  },
  sourcesToggle: {
    fontSize: '12px',
    padding: '4px 8px',
    minHeight: 'auto',
  },
  sourcesList: {
    marginTop: '6px',
    padding: '8px 12px',
    backgroundColor: tokens.colorNeutralBackground2,
    border: `1px solid ${tokens.colorNeutralStroke2}`,
    borderRadius: '6px',
    fontSize: '12px',
  },
  sourceItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    padding: '4px 0',
    color: tokens.colorNeutralForeground2,
  },
  
  assistantContent: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-start',
    minWidth: 0,
  },
});


function formatTime(isoString) {
  if (!isoString) return '';
  const date = new Date(isoString);
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}


export default function MessageBubble({ message }) {
  const styles = useStyles();
  const { activeUser, activeUserId } = useUser();
  const [sourcesOpen, setSourcesOpen] = useState(false);

  const isUser = message.role === 'user';
  const isAssistant = message.role === 'assistant';
  const sources = message.sources || [];
  const isStreaming = !!message.streaming;
  const isEmpty = !message.content;

  // ── USER MESSAGE ─────────────────────────────────────────────
  if (isUser) {
    const userName = activeUser?.display_name || activeUserId || 'You';
    return (
      <div className={mergeClasses(styles.row, styles.rowUser)}>
        <Avatar
          className={styles.avatar}
          name={userName}
          size={36}
          color="brand"
        />
        <div>
          <div className={mergeClasses(styles.bubble, styles.bubbleUser)}>
            {message.content}
          </div>
          <div className={styles.meta} style={{ textAlign: 'right' }}>
            {userName} · {formatTime(message.created_at)}
          </div>
        </div>
      </div>
    );
  }

  // ── ASSISTANT MESSAGE ────────────────────────────────────────
  return (
    <div className={styles.row}>
      <div className={styles.assistantIcon}>
        <Bot24Regular />
      </div>
      <div className={styles.assistantContent}>
        <div className={mergeClasses(
          styles.bubble,
          styles.bubbleAssistant,
          isEmpty && isStreaming && styles.bubbleEmpty,
        )}>
          {message.content}
          {isStreaming && <span className={styles.cursor} />}
        </div>
        <div className={styles.meta}>
          Assistant · {formatTime(message.created_at)}
        </div>

        {sources.length > 0 && (
          <div className={styles.sourcesContainer}>
            <Button
              appearance="subtle"
              size="small"
              className={styles.sourcesToggle}
              icon={sourcesOpen ? <ChevronUp16Regular /> : <ChevronDown16Regular />}
              iconPosition="after"
              onClick={() => setSourcesOpen(!sourcesOpen)}
            >
              📎 {sources.length} source{sources.length !== 1 ? 's' : ''}
            </Button>

            {sourcesOpen && (
              <div className={styles.sourcesList}>
                {sources.map((src, i) => {
                  const label = src.source || 'Unknown source';
                  const page = src.page ? `, page ${src.page}` : '';
                  return (
                    <div key={i} className={styles.sourceItem}>
                      <DocumentRegular />
                      <Text>[{i + 1}] {label}{page}</Text>
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