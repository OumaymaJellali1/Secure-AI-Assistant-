import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Text, Spinner, makeStyles } from '@fluentui/react-components';
import { ErrorCircle24Regular, CheckmarkCircle20Regular } from '@fluentui/react-icons';

import api from '../api';
import { useUser } from '../context/UserContext';
import { useChat } from '../context/ChatContext';
import MessageBubble from './MessageBubble';
import MessageInput from './MessageInput';

const MAX_W = '720px';
const F = "'Lexend', 'DM Sans', sans-serif";

const useStyles = makeStyles({
  root: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    background: '#F7F6F3',
    position: 'relative',
  },

  // ── Header ──────────────────────────────────────────────────────
  header: {
    padding: '0 28px',
    height: '52px',
    borderBottom: '1px solid rgba(0,0,0,0.07)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: '#ffffff',
    flexShrink: 0,
    position: 'relative',
  },
  headerTitle: {
    fontSize: '14px',
    fontWeight: '600',
    color: '#1a1a2e',
    fontFamily: F,
    letterSpacing: '-0.1px',
  },
  headerMeta: {
    position: 'absolute',
    right: '28px',
    fontSize: '12px',
    color: '#9ca3af',
    fontFamily: F,
  },

  // ── LANDING MODE ────────────────────────────────────────────────
  landingRoot: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '40px 28px 60px',
    overflow: 'hidden',
  },
  landingInner: {
    width: '100%',
    maxWidth: MAX_W,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
  landingEmoji: {
    fontSize: '52px',
    marginBottom: '20px',
    display: 'block',
    lineHeight: 1,
    animation: 'fadeUp 0.45s ease both',
  },
  landingTitle: {
    fontSize: '26px',
    fontWeight: '800',
    color: '#1a1a2e',
    fontFamily: F,
    margin: '0 0 10px',
    letterSpacing: '-0.6px',
    textAlign: 'center',
    animation: 'fadeUp 0.45s 0.06s ease both',
  },
  landingSub: {
    fontSize: '14px',
    color: '#6b7280',
    margin: '0 0 28px',
    lineHeight: '1.65',
    maxWidth: '360px',
    textAlign: 'center',
    fontFamily: F,
    animation: 'fadeUp 0.45s 0.1s ease both',
  },
  landingInputWrap: {
    width: '100%',
    animation: 'fadeUp 0.45s 0.14s ease both',
  },
  hints: {
    display: 'flex',
    gap: '8px',
    flexWrap: 'wrap',
    justifyContent: 'center',
    marginTop: '18px',
    animation: 'fadeUp 0.45s 0.2s ease both',
  },

  // ── CHAT MODE ────────────────────────────────────────────────────
  messagesArea: {
    flex: 1,
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: '0 28px',
  },
  messagesInner: {
    width: '100%',
    maxWidth: MAX_W,
    display: 'flex',
    flexDirection: 'column',
    flex: 1,
    paddingTop: '28px',
    paddingBottom: '16px',
    gap: '2px',
  },
  bottomInput: {
    flexShrink: 0,
  },

  // ── Status / error ───────────────────────────────────────────────
  statusRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    padding: '8px 0',
  },
  statusDot: {
    width: '28px',
    height: '28px',
    borderRadius: '50%',
    background: 'linear-gradient(135deg, #1a1a2e, #4f46e5)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
  },
  statusText: {
    color: '#9ca3af',
    fontSize: '13px',
    fontStyle: 'italic',
    fontFamily: F,
  },
  errorBox: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '10px 14px',
    background: '#fef2f2',
    border: '1px solid #fecaca',
    borderRadius: '10px',
    color: '#dc2626',
    fontSize: '13px',
    marginBottom: '12px',
    fontFamily: F,
  },

  // ── Upload toast ─────────────────────────────────────────────────
  uploadToast: {
    position: 'fixed',
    bottom: '100px',
    left: '50%',
    transform: 'translateX(-50%)',
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    padding: '12px 20px',
    borderRadius: '40px',
    fontSize: '13px',
    fontWeight: '600',
    fontFamily: F,
    boxShadow: '0 8px 32px rgba(0,0,0,0.15)',
    zIndex: 100,
    whiteSpace: 'nowrap',
    animation: 'toastPop 0.3s cubic-bezier(.34,1.56,.64,1)',
  },
  uploadToastProgress: { background: '#1a1a2e', color: '#fff' },
  uploadToastSuccess:  { background: '#f0fdf4', border: '1.5px solid #bbf7d0', color: '#166534' },
  uploadToastError:    { background: '#fef2f2', border: '1.5px solid #fecaca', color: '#dc2626' },
});

const HINT_PROMPTS = [
  'Summarize my latest report',
  'What does the contract say about termination?',
  'Find all mentions of the budget',
];

export default function ChatPanel({ onSidebarRefresh }) {
  const styles = useStyles();
  const { activeUserId, displayName } = useUser();
  const { activeSessionId, setActiveSessionId } = useChat();

  const [messages, setMessages]           = useState([]);
  const [loading, setLoading]             = useState(false);
  const [streaming, setStreaming]         = useState(false);
  const [streamStatus, setStreamStatus]   = useState(null);
  const [conversation, setConversation]   = useState(null);
  const [error, setError]                 = useState(null);
  const [uploadState, setUploadState]     = useState(null);
  const [prefillValue, setPrefillValue]   = useState('');

  // ── Knowledge scope state ────────────────────────────────────────
  const [queryScope, setQueryScope]           = useState('all_kb');
  const [queryDocumentId, setQueryDocumentId] = useState(null);
  const [documents, setDocuments]             = useState([]);

  const messagesEndRef = useRef(null);

  // ── Load documents list for the scope picker ─────────────────────
  const loadDocuments = useCallback(async () => {
    if (!activeUserId) return;
    try {
      const data = await api.listDocuments();
      setDocuments(data || []);
    } catch {
      // non-critical — scope picker just shows empty list
    }
  }, [activeUserId]);

  useEffect(() => { loadDocuments(); }, [loadDocuments]);

  // Reload docs whenever sidebar refreshes (new upload)
  const handleSidebarRefresh = useCallback(() => {
    loadDocuments();
    onSidebarRefresh?.();
  }, [loadDocuments, onSidebarRefresh]);

  // ── Load messages when session changes ───────────────────────────
  const loadMessages = useCallback(async () => {
    if (!activeSessionId) {
      setMessages([]);
      setConversation(null);
      return;
    }
    try {
      setLoading(true);
      setError(null);
      const data = await api.getConversation(activeSessionId);
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

  // Auto-clear upload toast
  useEffect(() => {
    if (!uploadState || uploadState.status === 'uploading') return;
    const t = setTimeout(() => setUploadState(null), 4000);
    return () => clearTimeout(t);
  }, [uploadState]);

  // ── Upload handler ───────────────────────────────────────────────
  const handleUpload = async (file) => {
    setUploadState({ status: 'uploading', filename: file.name });
    try {
      const result = await api.uploadDocument(file);
      setUploadState({
        status: 'success',
        filename: file.name,
        message: `${result.chunks || 0} chunks indexed`,
      });
      handleSidebarRefresh();
    } catch (err) {
      setUploadState({ status: 'error', filename: file.name, message: err.message });
      throw err;
    }
  };

  // ── Auto-create session if none selected ─────────────────────────
  const ensureSession = useCallback(async () => {
    if (activeSessionId) return activeSessionId;
    const newConv = await api.createConversation();
    setActiveSessionId(newConv.session_id);
    handleSidebarRefresh();
    return newConv.session_id;
  }, [activeSessionId, setActiveSessionId, handleSidebarRefresh]);

  // ── Build filters from scope selection ───────────────────────────
  const buildFilters = () => {
    if (queryScope === 'uploads_only') return { source_type: 'user_upload' };
    if (queryScope === 'single_doc' && queryDocumentId) return { document_id: queryDocumentId };
    return null; // all_kb → no filter
  };

  // ── Send handler ─────────────────────────────────────────────────
  const handleSend = async (question) => {
    if (streaming || !question.trim()) return;

    let sessionId;
    try {
      sessionId = await ensureSession();
    } catch (err) {
      setError(`Could not start conversation: ${err.message}`);
      return;
    }

    setStreaming(true);
    setStreamStatus(null);
    setError(null);

    const now = new Date().toISOString();
    setMessages(prev => [
      ...prev,
      { role: 'user',      content: question, created_at: now },
      { role: 'assistant', content: '', sources: [], created_at: now, streaming: true },
    ]);

    const filters = buildFilters();

    let buffer = '';
    await api.streamQuery(sessionId, question, {
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
            next[li] = {
              ...next[li],
              content: buffer || event.answer || '',
              sources: event.sources || [],
              streaming: false,
            };
          }
          return next;
        });
        setStreaming(false);
        setStreamStatus(null);
        handleSidebarRefresh();
      },
      onError: (errMsg) => {
        setError(errMsg);
        setStreaming(false);
        setStreamStatus(null);
        setMessages(prev =>
          prev.length > 0 && prev[prev.length - 1].role === 'assistant' && !prev[prev.length - 1].content
            ? prev.slice(0, -1)
            : prev
        );
      },
    }, filters);
  };

  // ── Shared MessageInput props ─────────────────────────────────────
  const inputProps = {
    onSend: handleSend,
    onUpload: handleUpload,
    disabled: streaming,
    scope: queryScope,
    documentId: queryDocumentId,
    documents,
    onScopeChange: ({ scope, documentId }) => {
      setQueryScope(scope);
      setQueryDocumentId(documentId ?? null);
    },
    prefill: prefillValue,
    onPrefillConsumed: () => setPrefillValue(''),
  };

  const name      = displayName || activeUserId || 'there';
  const isLanding = messages.length === 0 && !loading;

  const statusText =
    streamStatus === 'retrieving' ? 'Searching documents...'
    : streamStatus === 'generating' ? 'Generating answer...'
    : null;

  return (
    <div className={styles.root}>
      <style>{`
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(18px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes toastPop {
          from { opacity: 0; transform: translateX(-50%) translateY(8px) scale(.95); }
          to   { opacity: 1; transform: translateX(-50%) translateY(0) scale(1); }
        }
        .hint-pill {
          padding: 8px 15px;
          border-radius: 20px;
          border: 1.5px solid #e5e7eb;
          background: #ffffff;
          font-size: 13px;
          color: #4b5563;
          font-family: 'Lexend', 'DM Sans', sans-serif;
          cursor: pointer;
          transition: all 0.15s;
          user-select: none;
          display: inline-block;
        }
        .hint-pill:hover {
          border-color: #c7d2fe;
          background: #eef2ff;
          color: #4338ca;
          transform: translateY(-1px);
        }
      `}</style>

      {/* ── Header ───────────────────────────────────────────────── */}
      <div className={styles.header}>
        <span className={styles.headerTitle}>
          {isLanding && !activeSessionId
            ? 'RAG Assistant'
            : conversation?.title || 'New conversation'}
        </span>
        {!isLanding && (
          <span className={styles.headerMeta}>{messages.length} messages</span>
        )}
      </div>

      {/* ══════════════════════════════════════════════════════════
          LANDING MODE — input centered on the page
      ══════════════════════════════════════════════════════════ */}
      {isLanding && (
        <div className={styles.landingRoot}>
          <div className={styles.landingInner}>

            <span className={styles.landingEmoji}>👋</span>
            <h2 className={styles.landingTitle}>Hello, {name}!</h2>
            <p className={styles.landingSub}>
              Ask anything about your documents — choose a scope below, or attach one with&nbsp;📎.
            </p>

            {/* Input + scope picker centered */}
            <div className={styles.landingInputWrap}>
              <MessageInput {...inputProps} />
            </div>

            {/* Hint chips */}
            <div className={styles.hints}>
              {HINT_PROMPTS.map(h => (
                <span key={h} className="hint-pill" onClick={() => setPrefillValue(h)}>
                  {h}
                </span>
              ))}
            </div>

            {error && (
              <div className={styles.errorBox} style={{ marginTop: '20px', width: '100%' }}>
                <ErrorCircle24Regular />
                <span>{error}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ══════════════════════════════════════════════════════════
          CHAT MODE — messages scroll, input at bottom
      ══════════════════════════════════════════════════════════ */}
      {!isLanding && (
        <>
          <div className={styles.messagesArea}>
            <div className={styles.messagesInner}>

              {loading && (
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '12px' }}>
                  <Spinner size="medium" />
                  <Text style={{ fontFamily: F, color: '#9ca3af' }}>Loading…</Text>
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

          <div className={styles.bottomInput}>
            <MessageInput {...inputProps} prefill={undefined} onPrefillConsumed={undefined} />
          </div>
        </>
      )}

      {/* ── Upload toast ──────────────────────────────────────────── */}
      {uploadState && (
        <div className={`${styles.uploadToast} ${
          uploadState.status === 'uploading' ? styles.uploadToastProgress
          : uploadState.status === 'success'  ? styles.uploadToastSuccess
          : styles.uploadToastError
        }`}>
          {uploadState.status === 'uploading' && <Spinner size="tiny" appearance="inverted" />}
          {uploadState.status === 'success'   && <CheckmarkCircle20Regular style={{ flexShrink: 0 }} />}
          {uploadState.status === 'error'     && <span>⚠️</span>}
          <span>
            {uploadState.status === 'uploading'
              ? `Indexing ${uploadState.filename}…`
              : uploadState.status === 'success'
              ? `✓ ${uploadState.filename} · ${uploadState.message}`
              : `${uploadState.filename}: ${uploadState.message}`}
          </span>
        </div>
      )}
    </div>
  );
}