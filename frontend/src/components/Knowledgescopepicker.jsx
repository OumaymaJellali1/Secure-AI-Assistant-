/**
 * KnowledgeScopePicker.jsx
 *
 * A compact inline control that lets the user choose what knowledge sources
 * the RAG pipeline searches before answering.
 *
 * Props:
 *   scope          : "all_kb" | "uploads_only" | "single_doc"
 *   documentId     : string | null   (the selected doc when scope="single_doc")
 *   documents      : DocumentOut[]   (list from api.listDocuments())
 *   onChange       : ({ scope, documentId }) => void
 *
 * Usage (inside ChatPanel, above / beside the send button):
 *
 *   <KnowledgeScopePicker
 *     scope={queryScope}
 *     documentId={queryDocumentId}
 *     documents={documents}
 *     onChange={({ scope, documentId }) => {
 *       setQueryScope(scope);
 *       setQueryDocumentId(documentId);
 *     }}
 *   />
 */

import React, { useState, useRef, useEffect } from 'react';
import { makeStyles, tokens } from '@fluentui/react-components';

// ── Scope option definitions ──────────────────────────────────────
const SCOPE_OPTIONS = [
  {
    value: 'all_kb',
    label: 'All knowledge',
    shortLabel: 'All knowledge',
    icon: '🌐',
    description: 'Your uploads + everything stored in the vector database',
  },
  {
    value: 'uploads_only',
    label: 'My uploads',
    shortLabel: 'My uploads',
    icon: '📁',
    description: 'Only documents you have uploaded',
  },
  {
    value: 'single_doc',
    label: 'One document',
    shortLabel: 'One document',
    icon: '📄',
    description: 'Restrict answers to a single document',
  },
];

// ── Styles ────────────────────────────────────────────────────────
const useStyles = makeStyles({
  wrapper: {
    position: 'relative',
    display: 'inline-flex',
    alignItems: 'center',
    fontFamily: "'Lexend', 'Segoe UI', sans-serif",
  },

  // The pill trigger button
  trigger: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
    padding: '5px 10px 5px 8px',
    borderRadius: '20px',
    border: '1.5px solid rgba(79,70,229,0.25)',
    background: 'rgba(79,70,229,0.05)',
    cursor: 'pointer',
    fontSize: '12.5px',
    fontWeight: '600',
    color: '#4f46e5',
    fontFamily: "'Lexend', 'Segoe UI', sans-serif",
    transition: 'all 0.15s ease',
    userSelect: 'none',
    whiteSpace: 'nowrap',
    ':hover': {
      background: 'rgba(79,70,229,0.10)',
      borderColor: 'rgba(79,70,229,0.45)',
    },
  },
  triggerIcon: {
    fontSize: '13px',
    lineHeight: 1,
  },
  triggerChevron: {
    fontSize: '9px',
    opacity: 0.6,
    marginLeft: '1px',
    transition: 'transform 0.15s ease',
  },
  triggerChevronOpen: {
    transform: 'rotate(180deg)',
  },

  // Dropdown panel
  dropdown: {
    position: 'absolute',
    bottom: 'calc(100% + 8px)',
    left: 0,
    zIndex: 1000,
    background: '#ffffff',
    border: '1px solid rgba(0,0,0,0.1)',
    borderRadius: '14px',
    boxShadow: '0 8px 32px rgba(0,0,0,0.12), 0 2px 8px rgba(0,0,0,0.06)',
    padding: '6px',
    minWidth: '260px',
    animation: 'scopeDropIn 0.15s ease',
  },

  dropdownTitle: {
    fontSize: '10.5px',
    fontWeight: '700',
    color: '#9ca3af',
    letterSpacing: '0.6px',
    textTransform: 'uppercase',
    padding: '6px 10px 4px',
    fontFamily: "'Lexend', 'Segoe UI', sans-serif",
  },

  // Scope row
  scopeRow: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: '10px',
    padding: '8px 10px',
    borderRadius: '9px',
    cursor: 'pointer',
    transition: 'background 0.1s ease',
    ':hover': {
      background: '#f5f3ff',
    },
  },
  scopeRowActive: {
    background: '#ede9fe',
  },
  scopeRowIcon: {
    fontSize: '16px',
    lineHeight: 1,
    marginTop: '1px',
    flexShrink: 0,
  },
  scopeRowText: {
    flex: 1,
  },
  scopeRowLabel: {
    fontSize: '13px',
    fontWeight: '600',
    color: '#1a1a2e',
    fontFamily: "'Lexend', 'Segoe UI', sans-serif",
    lineHeight: 1.3,
  },
  scopeRowLabelActive: {
    color: '#4f46e5',
  },
  scopeRowDesc: {
    fontSize: '11.5px',
    color: '#6b7280',
    fontFamily: "'Lexend', 'Segoe UI', sans-serif",
    lineHeight: 1.4,
    marginTop: '1px',
  },
  scopeRowCheck: {
    fontSize: '13px',
    color: '#4f46e5',
    flexShrink: 0,
    marginTop: '2px',
  },

  // Document selector (appears when single_doc is active)
  docSelectWrap: {
    padding: '6px 10px 8px',
    borderTop: '1px solid rgba(0,0,0,0.06)',
    marginTop: '2px',
  },
  docSelectLabel: {
    fontSize: '10.5px',
    fontWeight: '700',
    color: '#9ca3af',
    letterSpacing: '0.6px',
    textTransform: 'uppercase',
    marginBottom: '5px',
    fontFamily: "'Lexend', 'Segoe UI', sans-serif",
  },
  docSelect: {
    width: '100%',
    padding: '7px 10px',
    borderRadius: '8px',
    border: '1.5px solid rgba(79,70,229,0.3)',
    background: '#f9f8ff',
    fontSize: '12.5px',
    color: '#1a1a2e',
    fontFamily: "'Lexend', 'Segoe UI', sans-serif",
    fontWeight: '500',
    outline: 'none',
    cursor: 'pointer',
    appearance: 'auto',
    ':focus': {
      borderColor: '#4f46e5',
    },
  },
  docSelectEmpty: {
    color: '#9ca3af',
    fontStyle: 'italic',
  },
});

// ── Component ─────────────────────────────────────────────────────
export default function KnowledgeScopePicker({
  scope = 'all_kb',
  documentId = null,
  documents = [],
  onChange,
}) {
  const styles = useStyles();
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    if (!open) return;
    const handler = (e) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  const activeOption = SCOPE_OPTIONS.find(o => o.value === scope) || SCOPE_OPTIONS[0];

  const handleSelectScope = (value) => {
    if (value === 'single_doc') {
      // Auto-select first doc if none selected yet
      const firstDoc = documents[0]?.document_id ?? null;
      onChange?.({ scope: value, documentId: documentId ?? firstDoc });
    } else {
      onChange?.({ scope: value, documentId: null });
      setOpen(false);
    }
  };

  const handleDocChange = (e) => {
    onChange?.({ scope: 'single_doc', documentId: e.target.value || null });
  };

  return (
    <>
      {/* Keyframe animation injected once */}
      <style>{`
        @keyframes scopeDropIn {
          from { opacity: 0; transform: translateY(4px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>

      <div ref={wrapperRef} className={styles.wrapper}>
        {/* Pill trigger */}
        <button
          className={styles.trigger}
          onClick={() => setOpen(v => !v)}
          title="Choose knowledge source"
          type="button"
        >
          <span className={styles.triggerIcon}>{activeOption.icon}</span>
          {activeOption.shortLabel}
          {scope === 'single_doc' && documentId && (
            <span style={{ opacity: 0.65, fontWeight: 400, maxWidth: 90, overflow: 'hidden', textOverflow: 'ellipsis', display: 'inline-block' }}>
              &nbsp;·&nbsp;{documents.find(d => d.document_id === documentId)?.filename ?? documentId}
            </span>
          )}
          <span className={`${styles.triggerChevron}${open ? ` ${styles.triggerChevronOpen}` : ''}`}>▼</span>
        </button>

        {/* Dropdown */}
        {open && (
          <div className={styles.dropdown}>
            <div className={styles.dropdownTitle}>Search scope</div>

            {SCOPE_OPTIONS.map(opt => {
              const isActive = scope === opt.value;
              return (
                <div
                  key={opt.value}
                  className={`${styles.scopeRow}${isActive ? ` ${styles.scopeRowActive}` : ''}`}
                  onClick={() => handleSelectScope(opt.value)}
                >
                  <span className={styles.scopeRowIcon}>{opt.icon}</span>
                  <div className={styles.scopeRowText}>
                    <div className={`${styles.scopeRowLabel}${isActive ? ` ${styles.scopeRowLabelActive}` : ''}`}>
                      {opt.label}
                    </div>
                    <div className={styles.scopeRowDesc}>{opt.description}</div>
                  </div>
                  {isActive && <span className={styles.scopeRowCheck}>✓</span>}
                </div>
              );
            })}

            {/* Document selector — only shown when single_doc is active */}
            {scope === 'single_doc' && (
              <div className={styles.docSelectWrap}>
                <div className={styles.docSelectLabel}>Select document</div>
                {documents.length === 0 ? (
                  <div className={`${styles.docSelect} ${styles.docSelectEmpty}`}
                    style={{ padding: '7px 10px', fontSize: '12px', color: '#9ca3af', fontStyle: 'italic' }}>
                    No documents uploaded yet
                  </div>
                ) : (
                  <select
                    className={styles.docSelect}
                    value={documentId ?? ''}
                    onChange={handleDocChange}
                  >
                    <option value="" disabled>— choose a document —</option>
                    {documents.map(doc => (
                      <option key={doc.document_id} value={doc.document_id}>
                        {doc.filename}
                      </option>
                    ))}
                  </select>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </>
  );
}