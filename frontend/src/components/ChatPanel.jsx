/**
 * ChatPanel.jsx — Right-side chat area with STREAMING responses.
 *
 * Streaming flow:
 *   1. User sends question
 *   2. Show their message immediately
 *   3. Add empty assistant message + status indicator ("Retrieving...", "Generating...")
 *   4. As tokens arrive, append to the assistant message live
 *   5. When done, attach sources and trigger sidebar refresh
 */
import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Text,
  Spinner,
  tokens,
  makeStyles,
} from '@fluentui/react-components';
import {
  ChatMultiple24Regular,
  ErrorCircle24Regular,
} from '@fluentui/react-icons';

import api from '../api';
import { useUser } from '../context/UserContext';
import { useChat } from '../context/ChatContext';
import MessageBubble from './MessageBubble';
import MessageInput from './MessageInput';


const useStyles = makeStyles({
  root: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    backgroundColor: tokens.colorNeutralBackground2,
    overflow: 'hidden',
  },
  header: {
    padding: '14px 24px',
    borderBottom: `1px solid ${tokens.colorNeutralStroke2}`,
    backgroundColor: tokens.colorNeutralBackground1,
    minHeight: '52px',
    display: 'flex',
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: '15px',
    fontWeight: 600,
  },
  messagesArea: {
    flex: 1,
    overflowY: 'auto',
    padding: '24px',
  },
  messagesInner: {
    maxWidth: '900px',
    margin: '0 auto',
  },
  emptyState: {
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    color: tokens.colorNeutralForeground3,
    gap: '12px',
    textAlign: 'center',
    padding: '40px',
  },
  emptyIcon: {
    fontSize: '64px',
    color: tokens.colorNeutralForeground4,
  },
  welcome: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    color: tokens.colorNeutralForeground3,
    gap: '16px',
    padding: '40px',
    textAlign: 'center',
  },
  // Status indicator (Retrieving... / Generating...)
  statusRow: {
    display: 'flex',
    gap: '12px',
    alignItems: 'center',
    padding: '12px 16px',
    marginBottom: '20px',
  },
  statusIcon: {
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
  statusText: {
    color: tokens.colorNeutralForeground3,
    fontStyle: 'italic',
    fontSize: '14px',
  },
  errorBox: {
    padding: '12px 16px',
    backgroundColor: tokens.colorPaletteRedBackground1,
    border: `1px solid ${tokens.colorPaletteRedBorder1}`,
    borderRadius: '8px',
    marginBottom: '16px',
    color: tokens.colorPaletteRedForeground1,
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
});


export default function ChatPanel({ onSidebarRefresh }) {
  const styles = useStyles();
  const { activeUserId, activeUser } = useUser();
  const { activeSessionId } = useChat();

  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [streamStatus, setStreamStatus] = useState(null); // "retrieving" | "generating" | null
  const [conversation, setConversation] = useState(null);
  const [error, setError] = useState(null);

  const messagesEndRef = useRef(null);

  // ── Load messages when conversation changes ──────────────────
  const loadMessages = useCallback(async () => {
    if (!activeSessionId || !activeUserId) {
      setMessages([]);
      setConversation(null);
      return;
    }
    try {
      setLoading(true);
      setError(null);
      const data = await api.getConversation(activeUserId, activeSessionId);
      setConversation(data);
      setMessages(data.turns || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [activeUserId, activeSessionId]);

  useEffect(() => {
    loadMessages();
  }, [loadMessages]);

  // ── Auto-scroll to bottom on new messages or token updates ──
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, streaming]);

  // ── Send message via STREAMING ───────────────────────────────
  const handleSend = async (question) => {
    if (streaming) return;

    setStreaming(true);
    setStreamStatus(null);
    setError(null);

    const now = new Date().toISOString();

    // 1. Add user message immediately
    const userMessage = {
      role: 'user',
      content: question,
      created_at: now,
    };

    // 2. Add empty assistant message (will be filled as tokens arrive)
    const assistantMessage = {
      role: 'assistant',
      content: '',
      sources: [],
      created_at: now,
      streaming: true, // mark for blinking cursor effect
    };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);

    // Use a buffer to accumulate tokens (avoids re-render thrashing)
    let buffer = '';

    await api.streamQuery(activeUserId, activeSessionId, question, {
      onStatus: (stage) => {
        setStreamStatus(stage);
      },

      onToken: (token) => {
        buffer += token;
        // Update the last message in the array with the new content
        setMessages((prev) => {
          const newMessages = [...prev];
          const lastIdx = newMessages.length - 1;
          if (lastIdx >= 0 && newMessages[lastIdx].role === 'assistant') {
            newMessages[lastIdx] = {
              ...newMessages[lastIdx],
              content: buffer,
            };
          }
          return newMessages;
        });
        // Clear status once we start receiving tokens
        if (streamStatus !== null) {
          setStreamStatus(null);
        }
      },

      onDone: (event) => {
        // Final update: attach sources, remove streaming flag
        setMessages((prev) => {
          const newMessages = [...prev];
          const lastIdx = newMessages.length - 1;
          if (lastIdx >= 0 && newMessages[lastIdx].role === 'assistant') {
            newMessages[lastIdx] = {
              ...newMessages[lastIdx],
              content: buffer || event.answer || '',
              sources: event.sources || [],
              streaming: false,
            };
          }
          return newMessages;
        });
        setStreaming(false);
        setStreamStatus(null);
        // Trigger sidebar refresh (in case auto-title updated)
        if (onSidebarRefresh) onSidebarRefresh();
      },

      onError: (errMsg) => {
        setError(errMsg);
        setStreaming(false);
        setStreamStatus(null);
        // Remove the empty assistant bubble
        setMessages((prev) => {
          if (prev.length > 0 && prev[prev.length - 1].role === 'assistant' && !prev[prev.length - 1].content) {
            return prev.slice(0, -1);
          }
          return prev;
        });
      },
    });
  };

  // ── Welcome state (no conversation selected) ────────────────
  if (!activeSessionId) {
    return (
      <div className={styles.root}>
        <div className={styles.welcome}>
          <ChatMultiple24Regular className={styles.emptyIcon} />
          <Text size={500} weight="semibold">
            Welcome, {activeUser?.display_name || activeUserId}
          </Text>
          <Text>Select a conversation or start a new one.</Text>
          <Text style={{ fontSize: 13, marginTop: 8 }}>
            💡 Each user sees only their own conversations and documents.
          </Text>
        </div>
      </div>
    );
  }

  // ── Active conversation ──────────────────────────────────────
  const conversationTitle = conversation?.title || '(untitled)';

  // Status text for the indicator
  const statusText = streamStatus === 'retrieving'
    ? 'Searching documents...'
    : streamStatus === 'generating'
      ? 'Generating answer...'
      : null;

  return (
    <div className={styles.root}>
      <div className={styles.header}>
        <Text className={styles.headerTitle}>{conversationTitle}</Text>
      </div>

      <div className={styles.messagesArea}>
        <div className={styles.messagesInner}>

          {loading && (
            <div className={styles.emptyState}>
              <Spinner size="medium" />
              <Text>Loading conversation...</Text>
            </div>
          )}

          {!loading && messages.length === 0 && (
            <div className={styles.emptyState}>
              <ChatMultiple24Regular className={styles.emptyIcon} />
              <Text size={400} weight="semibold">Start the conversation</Text>
              <Text>Type your first question below.</Text>
            </div>
          )}

          {!loading && messages.map((msg, i) => (
            <MessageBubble key={i} message={msg} />
          ))}

          {/* Status while waiting for tokens */}
          {streaming && statusText && (
            <div className={styles.statusRow}>
              <div className={styles.statusIcon}>
                <Spinner size="tiny" appearance="inverted" />
              </div>
              <Text className={styles.statusText}>{statusText}</Text>
            </div>
          )}

          {error && (
            <div className={styles.errorBox}>
              <ErrorCircle24Regular />
              <Text>{error}</Text>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      <MessageInput onSend={handleSend} disabled={streaming} />
    </div>
  );
}