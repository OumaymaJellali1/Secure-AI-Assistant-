import React, { useState, useRef, useEffect } from 'react';
import { makeStyles } from '@fluentui/react-components';
import { Send24Regular, Attach24Regular, DocumentRegular, Dismiss16Regular } from '@fluentui/react-icons';
import KnowledgeScopePicker from './KnowledgeScopePicker';

const ACCEPTED = '.pdf,.docx,.txt,.md,.csv,.xlsx,.pptx,.eml,.jpg,.jpeg,.png';
const F = "'Lexend', 'DM Sans', sans-serif";

const useStyles = makeStyles({
  container: {
    padding: '10px 28px 20px',
    background: '#F7F6F3',
    flexShrink: 0,
  },
  inner: {
    maxWidth: '720px',
    margin: '0 auto',
  },

  // ── File chips above input ───────────────────────────────────────
  attachedFiles: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '6px',
    marginBottom: '8px',
  },
  fileChip: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    padding: '5px 8px 5px 10px',
    background: '#eef2ff',
    border: '1.5px solid #c7d2fe',
    borderRadius: '20px',
    fontSize: '12px',
    color: '#4338ca',
    fontFamily: F,
    fontWeight: '500',
    maxWidth: '200px',
  },
  fileChipName: {
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    maxWidth: '130px',
  },
  fileChipRemove: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '16px',
    height: '16px',
    borderRadius: '50%',
    border: 'none',
    background: 'rgba(67,56,202,0.15)',
    color: '#4338ca',
    cursor: 'pointer',
    padding: 0,
    flexShrink: 0,
    transition: 'background 0.15s',
    ':hover': { background: 'rgba(67,56,202,0.3)' },
  },
  uploadingChip: {
    background: '#fffbeb',
    border: '1.5px solid #fde68a',
    color: '#92400e',
  },

  // ── Main input box ───────────────────────────────────────────────
  box: {
    display: 'flex',
    flexDirection: 'column',
    background: '#ffffff',
    border: '1.5px solid #e5e7eb',
    borderRadius: '16px',
    padding: '10px 10px 8px 14px',
    transition: 'border-color 0.15s, box-shadow 0.15s',
    ':focus-within': {
      borderColor: '#1a1a2e',
      boxShadow: '0 0 0 3px rgba(26,26,46,0.06)',
    },
  },

  // Top row: textarea + send button
  inputRow: {
    display: 'flex',
    alignItems: 'flex-end',
    gap: '8px',
  },
  textarea: {
    flex: 1,
    border: 'none',
    background: 'transparent',
    resize: 'none',
    minHeight: '22px',
    maxHeight: '180px',
    fontSize: '14px',
    fontFamily: F,
    color: '#1a1a2e',
    outline: 'none',
    padding: '2px 0',
    lineHeight: '1.55',
    overflowY: 'auto',
    '::placeholder': { color: '#c4c9d4' },
  },
  sendBtn: {
    width: '36px',
    height: '36px',
    borderRadius: '10px',
    border: 'none',
    background: '#1a1a2e',
    color: '#ffffff',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    cursor: 'pointer',
    flexShrink: 0,
    transition: 'opacity 0.15s, transform 0.1s',
    ':disabled': { opacity: 0.35, cursor: 'not-allowed' },
    ':hover:not(:disabled)': { opacity: 0.85 },
    ':active:not(:disabled)': { transform: 'scale(0.94)' },
  },

  // Bottom toolbar row: attach + scope picker
  toolbar: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginTop: '8px',
    paddingTop: '7px',
    borderTop: '1px solid rgba(0,0,0,0.05)',
  },
  attachBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: '5px',
    padding: '4px 10px 4px 8px',
    borderRadius: '20px',
    border: '1.5px solid #e5e7eb',
    background: '#f9fafb',
    color: '#6b7280',
    fontSize: '12px',
    fontWeight: '600',
    fontFamily: F,
    cursor: 'pointer',
    flexShrink: 0,
    transition: 'all 0.15s',
    ':hover': { background: '#eef2ff', borderColor: '#c7d2fe', color: '#4f46e5' },
    ':disabled': { opacity: 0.4, cursor: 'not-allowed' },
  },

  hiddenInput: { display: 'none' },

  hint: {
    textAlign: 'center',
    fontSize: '11px',
    color: '#c4c9d4',
    marginTop: '7px',
    fontFamily: F,
    letterSpacing: '0.1px',
  },
});

export default function MessageInput({
  onSend,
  onUpload,
  disabled,
  // scope props — lifted up to ChatPanel
  scope,
  documentId,
  documents,
  onScopeChange,
  // optional prefill from hint chips
  prefill,
  onPrefillConsumed,
}) {
  const styles = useStyles();
  const [value, setValue] = useState('');
  const [pendingFiles, setPendingFiles] = useState([]);
  const [uploadingFiles, setUploadingFiles] = useState(new Set());
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);

  // Apply prefill value when parent sets it
  useEffect(() => {
    if (!prefill) return;
    setValue(prefill);
    onPrefillConsumed?.();
    textareaRef.current?.focus();
  }, [prefill]);

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 180) + 'px';
  }, [value]);

  const handleSend = () => {
    const text = value.trim();
    if ((!text && pendingFiles.length === 0) || disabled) return;
    onSend(text);
    setValue('');
    setPendingFiles([]);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileChange = async (e) => {
    const files = Array.from(e.target.files || []);
    e.target.value = '';
    if (!files.length) return;

    for (const file of files) {
      const id = `${file.name}-${Date.now()}-${Math.random()}`;
      setPendingFiles(prev => [...prev, { id, name: file.name, file, done: false }]);
      setUploadingFiles(prev => new Set(prev).add(id));
      try {
        if (onUpload) await onUpload(file);
        setPendingFiles(prev => prev.map(f => f.id === id ? { ...f, done: true } : f));
      } catch {
        setPendingFiles(prev => prev.filter(f => f.id !== id));
      } finally {
        setUploadingFiles(prev => { const n = new Set(prev); n.delete(id); return n; });
      }
    }
  };

  const removeFile = (id) => setPendingFiles(prev => prev.filter(f => f.id !== id));

  const canSend = (value.trim().length > 0 || pendingFiles.some(f => f.done)) && !disabled;

  return (
    <div className={styles.container}>
      <div className={styles.inner}>

        {/* File chips */}
        {pendingFiles.length > 0 && (
          <div className={styles.attachedFiles}>
            {pendingFiles.map(f => {
              const isUploading = uploadingFiles.has(f.id);
              return (
                <div key={f.id} className={`${styles.fileChip} ${isUploading ? styles.uploadingChip : ''}`}>
                  <DocumentRegular style={{ fontSize: 13, flexShrink: 0 }} />
                  <span className={styles.fileChipName} title={f.name}>
                    {isUploading ? 'Uploading…' : f.name}
                  </span>
                  {!isUploading && (
                    <button className={styles.fileChipRemove} onClick={() => removeFile(f.id)} title="Remove">
                      <Dismiss16Regular style={{ fontSize: 10 }} />
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* Input box */}
        <div className={styles.box}>

          {/* Top row: textarea + send */}
          <div className={styles.inputRow}>
            <textarea
              ref={textareaRef}
              className={styles.textarea}
              value={value}
              onChange={e => setValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={disabled ? 'Generating response...' : 'Ask anything about your documents...'}
              disabled={disabled}
              rows={1}
            />
            <button
              className={styles.sendBtn}
              onClick={handleSend}
              disabled={!canSend}
              type="button"
            >
              <Send24Regular style={{ fontSize: 16 }} />
            </button>
          </div>

          {/* Bottom toolbar: attach + scope picker */}
          <div className={styles.toolbar}>
            {/* Paperclip button */}
            <button
              className={styles.attachBtn}
              onClick={() => fileInputRef.current?.click()}
              disabled={disabled}
              title="Attach a document"
              type="button"
            >
              <Attach24Regular style={{ fontSize: 14 }} />
              Attach
            </button>

            <input
              type="file"
              ref={fileInputRef}
              className={styles.hiddenInput}
              accept={ACCEPTED}
              multiple
              onChange={handleFileChange}
            />

            {/* Knowledge scope picker */}
            <KnowledgeScopePicker
              scope={scope}
              documentId={documentId}
              documents={documents || []}
              onChange={onScopeChange}
            />
          </div>
        </div>

        <div className={styles.hint}>
          Enter to send · Shift+Enter for new line
        </div>
      </div>
    </div>
  );
}