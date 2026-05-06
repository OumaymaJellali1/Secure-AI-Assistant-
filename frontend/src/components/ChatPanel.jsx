import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Text, Spinner, makeStyles } from '@fluentui/react-components';
import { ChatMultiple24Regular, ErrorCircle24Regular } from '@fluentui/react-icons';

import api from '../api';
import { useUser } from '../context/UserContext';
import { useChat } from '../context/ChatContext';
import MessageBubble from './MessageBubble';
import MessageInput from './MessageInput';

const useStyles = makeStyles({
  root: { flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', background: '#F7F6F3' },
  header: {
    padding: '0 28px', height: '52px', borderBottom: '1px solid rgba(0,0,0,0.07)',
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    background: '#ffffff', flexShrink: 0,
  },
  headerTitle: { fontSize: '14px', fontWeight: '600', color: '#1a1a2e', fontFamily: "'DM Sans', sans-serif", letterSpacing: '-0.1px' },
  headerMeta: { fontSize: '12px', color: '#9ca3af', fontFamily: "'DM Sans', sans-serif" },
  messagesArea: { flex: 1, overflowY: 'auto', padding: '24px 28px', display: 'flex', flexDirection: 'column' },
  messagesInner: { maxWidth: '800px', width: '100%', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '4px', flex: 1 },
  emptyState: { flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '12px', textAlign: 'center', padding: '60px 24px', color: '#9ca3af' },
  emptyIcon: { fontSize: '48px', marginBottom: '4px', opacity: 0.5 },
  emptyTitle: { fontSize: '18px', fontWeight: '600', color: '#374151', fontFamily: "'DM Sans', sans-serif", margin: '0 0 4px' },
  emptySub: { fontSize: '14px', color: '#9ca3af', margin: 0, lineHeight: '1.5' },
  welcome: { flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '60px 40px', textAlign: 'center' },
  welcomeEmoji: { fontSize: '48px', marginBottom: '16px', display: 'block' },
  welcomeTitle: { fontSize: '22px', fontWeight: '700', color: '#1a1a2e', fontFamily: "'DM Sans', sans-serif", margin: '0 0 8px', letterSpacing: '-0.3px' },
  welcomeSub: { fontSize: '14px', color: '#6b7280', margin: '0 0 24px', lineHeight: '1.6', maxWidth: '320px' },
  hints: { display: 'flex', gap: '10px', flexWrap: 'wrap', justifyContent: 'center', maxWidth: '500px' },
  hint: { padding: '8px 14px', borderRadius: '20px', border: '1.5px solid #e5e7eb', background: '#ffffff', fontSize: '13px', color: '#374151', fontFamily: "'DM Sans', sans-serif", cursor: 'default' },
  statusRow: { display: 'flex', alignItems: 'center', gap: '10px', padding: '8px 0' },
  statusDot: { width: '28px', height: '28px', borderRadius: '50%', background: 'linear-gradient(135deg, #1a1a2e, #4f46e5)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 },
  statusText: { color: '#9ca3af', fontSize: '13px', fontStyle: 'italic', fontFamily: "'DM Sans', sans-serif" },
  errorBox: { display: 'flex', alignItems: 'center', gap: '8px', padding: '10px 14px', background: '#fef2f2', border: '1px solid #fecaca', borderRadius: '10px', color: '#dc2626', fontSize: '13px', marginBottom: '12px', fontFamily: "'DM Sans', sans-serif" },
});

export default function ChatPanel({ onSidebarRefresh }) {
  const styles = useStyles();
  const { activeUserId, displayName } = useUser();
  const { activeSessionId } = useChat();

  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [streamStatus, setStreamStatus] = useState(null);
  const [conversation, setConversation] = useState(null);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

  const loadMessages = useCallback(async () => {
    if (!activeSessionId) {
      setMessages([]);
      setConversation(null);
      return;
    }
    try {
      setLoading(true);
      setError(null);
      const data = await api.getConversation(activeSessionId);   // ← no userId
      setConversation(data);
      setMessages(data.turns || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [activeSessionId]);

  useEffect(() => { loadMessages(); }, [loadMessages]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streaming]);

  const handleSend = async (question) => {
    if (streaming) return;
    setStreaming(true);
    setStreamStatus(null);
    setError(null);

    const now = new Date().toISOString();
    const userMsg = { role: 'user', content: question, created_at: now };
    const assistantMsg = { role: 'assistant', content: '', sources: [], created_at: now, streaming: true };
    setMessages(prev => [...prev, userMsg, assistantMsg]);

    let buffer = '';
    await api.streamQuery(activeSessionId, question, {   // ← no userId
      onStatus: (stage) => setStreamStatus(stage),
      onToken: (token) => {
        buffer += token;
        setMessages(prev => {
          const next = [...prev];
          const li = next.length - 1;
          if (li >= 0 && next[li].role === 'assistant') {
            next[li] = { ...next[li], content: buffer };
          }
          return next;
        });
        setStreamStatus(null);
      },
      onDone: (event) => {
        setMessages(prev => {
          const next = [...prev];
          const li = next.length - 1;
          if (li >= 0 && next[li].role === 'assistant') {
            next[li] = { ...next[li], content: buffer || event.answer || '', sources: event.sources || [], streaming: false };
          }
          return next;
        });
        setStreaming(false);
        setStreamStatus(null);
        onSidebarRefresh?.();
      },
      onError: (errMsg) => {
        setError(errMsg);
        setStreaming(false);
        setStreamStatus(null);
        setMessages(prev => {
          if (prev.length > 0 && prev[prev.length - 1].role === 'assistant' && !prev[prev.length - 1].content) {
            return prev.slice(0, -1);
          }
          return prev;
        });
      },
    });
  };

  if (!activeSessionId) {
    const name = displayName || activeUserId || 'there';
    return (
      <div className={styles.root}>
        <div className={styles.welcome}>
          <span className={styles.welcomeEmoji}>👋</span>
          <h2 className={styles.welcomeTitle}>Hello, {name}!</h2>
          <p className={styles.welcomeSub}>
            Select a conversation from the sidebar, or start a new one to ask anything about your documents.
          </p>
          <div className={styles.hints}>
            {['Summarize my latest report', 'What does the contract say about...', 'Find all mentions of...'].map(h => (
              <span key={h} className={styles.hint}>"{h}"</span>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const statusText = streamStatus === 'retrieving' ? 'Searching documents...'
    : streamStatus === 'generating' ? 'Generating answer...'
    : null;

  return (
    <div className={styles.root}>
      <div className={styles.header}>
        <span className={styles.headerTitle}>{conversation?.title || 'New conversation'}</span>
        <span className={styles.headerMeta}>{messages.length} messages</span>
      </div>

      <div className={styles.messagesArea}>
        <div className={styles.messagesInner}>
          {loading && (
            <div className={styles.emptyState}>
              <Spinner size="medium" />
              <Text style={{ fontFamily: "'DM Sans', sans-serif", color: '#9ca3af' }}>Loading...</Text>
            </div>
          )}

          {!loading && messages.length === 0 && (
            <div className={styles.emptyState}>
              <ChatMultiple24Regular className={styles.emptyIcon} />
              <p className={styles.emptyTitle}>Start the conversation</p>
              <p className={styles.emptySub}>Type your question below to get answers from your documents.</p>
            </div>
          )}

          {!loading && messages.map((msg, i) => (
            <MessageBubble key={i} message={msg} />
          ))}

          {streaming && statusText && (
            <div className={styles.statusRow}>
              <div className={styles.statusDot}>
                <Spinner size="tiny" appearance="inverted" />
              </div>
              <span className={styles.statusText}>{statusText}</span>
            </div>
          )}

          {error && (
            <div className={styles.errorBox}>
              <ErrorCircle24Regular />
              <span>{error}</span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      <MessageInput onSend={handleSend} disabled={streaming} />
    </div>
  );
}