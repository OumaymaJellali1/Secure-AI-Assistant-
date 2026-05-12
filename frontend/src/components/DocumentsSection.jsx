import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Spinner, Menu, MenuTrigger, MenuPopover, MenuList, MenuItem,
  makeStyles, mergeClasses,
} from '@fluentui/react-components';
import {
  DocumentRegular, DocumentAdd20Regular, DocumentMultipleRegular,
  MoreHorizontal20Regular, DeleteRegular, CheckmarkCircle20Regular,
  DismissCircle20Regular, ChevronUp16Regular, ChevronDown16Regular,
} from '@fluentui/react-icons';
import api from '../api';
import { useUser } from '../context/UserContext';

const ACCEPTED = '.pdf,.docx,.txt,.md,.csv,.xlsx,.pptx,.eml';

const F = "'Lexend', 'DM Sans', sans-serif";

const useStyles = makeStyles({
  section: { marginBottom: '4px' },
  sectionHeader: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '6px 4px 6px', marginBottom: '4px',
  },
  sectionLabel: {
    display: 'flex', alignItems: 'center', gap: '6px',
    color: '#9ca3af', fontSize: '12px', fontWeight: '700',
    textTransform: 'uppercase', letterSpacing: '0.7px', fontFamily: F,
    cursor: 'pointer', userSelect: 'none',
    ':hover': { color: '#6b7280' },
  },
  collapseChevron: {
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    width: '16px', height: '16px', flexShrink: 0,
    transition: 'transform 0.2s ease',
  },
  headerRight: {
    display: 'flex', alignItems: 'center', gap: '6px',
  },
  countBadge: {
    fontSize: '11px', color: '#9ca3af', background: '#f3f4f6',
    borderRadius: '8px', padding: '1px 6px', fontFamily: F, fontWeight: '600',
  },
  uploadBtn: {
    display: 'flex', alignItems: 'center', gap: '7px',
    border: '1.5px solid #e0e7ff', background: 'linear-gradient(135deg, #eef2ff, #f5f3ff)',
    color: '#4f46e5', fontSize: '13px', fontWeight: '600',
    cursor: 'pointer', fontFamily: F,
    padding: '8px 14px', borderRadius: '10px',
    transition: 'all 0.2s', boxShadow: '0 1px 4px rgba(99,102,241,.15)',
  },
  hiddenInput: { display: 'none' },
  collapsibleBody: {
    overflow: 'hidden',
    transition: 'max-height 0.25s ease, opacity 0.2s ease',
  },
  toast: {
    padding: '10px 14px', borderRadius: '12px', marginBottom: '8px',
    display: 'flex', alignItems: 'center', gap: '10px',
    fontSize: '13px', fontWeight: '500', fontFamily: F,
    boxShadow: '0 4px 16px rgba(0,0,0,.1)',
    animation: 'toastSlideIn .3s cubic-bezier(.34,1.56,.64,1)',
  },
  toastProgress: {
    background: '#eff6ff', border: '1.5px solid #bfdbfe', color: '#1d4ed8',
  },
  toastSuccess: {
    background: '#f0fdf4', border: '1.5px solid #bbf7d0', color: '#166534',
  },
  toastError: {
    background: '#fef2f2', border: '1.5px solid #fecaca', color: '#dc2626',
  },
  toastMsg: {
    flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  },
  progressBar: {
    height: '3px', background: '#bfdbfe', borderRadius: '2px',
    overflow: 'hidden', marginTop: '6px',
  },
  progressFill: {
    height: '100%', background: 'linear-gradient(90deg, #3b82f6, #6366f1)',
    borderRadius: '2px', animation: 'progressPulse 1.2s ease-in-out infinite',
  },
  item: {
    display: 'flex', alignItems: 'center', gap: '8px',
    padding: '7px 8px', borderRadius: '9px',
    cursor: 'default', transition: 'background 0.12s', minHeight: '32px',
    ':hover': { background: '#f3f4f6' },
  },
  itemIcon: { fontSize: '14px', color: '#818cf8', flexShrink: 0 },
  itemTitle: {
    flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
    fontSize: '13px', color: '#374151', fontFamily: F, fontWeight: '400',
  },
  itemMeta: {
    fontSize: '11px', color: '#c4c9d4', fontFamily: F, flexShrink: 0,
    background: '#f3f4f6', padding: '2px 6px', borderRadius: '4px',
  },
  itemMenu: {
    flexShrink: 0, opacity: 0, transition: 'opacity 0.1s',
    minWidth: 'auto', height: '22px', padding: '0 3px',
  },
  itemMenuVisible: { opacity: 1 },
  emptyMsg: {
    color: '#c4c9d4', fontSize: '13px', textAlign: 'center',
    padding: '12px 8px', fontStyle: 'italic', fontFamily: F,
  },
  loadingBox: { display: 'flex', justifyContent: 'center', padding: '12px' },
  dropZone: {
    border: '2px dashed #e0e7ff', borderRadius: '12px',
    padding: '16px', textAlign: 'center', marginBottom: '8px',
    background: '#fafbff', cursor: 'pointer', transition: 'all .2s',
    color: '#818cf8', fontSize: '13px', fontFamily: F,
  },
  dropZoneActive: {
    borderColor: '#6366f1', background: '#eef2ff', color: '#4f46e5',
  },
});

function DocItem({ doc, onDelete }) {
  const styles = useStyles();
  const [hovered, setHovered] = useState(false);

  return (
    <div className={styles.item}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      title={doc.filename}>
      <DocumentRegular className={styles.itemIcon} />
      <span className={styles.itemTitle}>{doc.filename}</span>
      <span className={styles.itemMeta}>{doc.chunks ?? 0}c</span>
      <Menu>
        <MenuTrigger disableButtonEnhancement>
          <button
            className={mergeClasses(styles.itemMenu, hovered && styles.itemMenuVisible)}
            style={{ border: 'none', background: 'transparent', cursor: 'pointer', display: 'flex', alignItems: 'center', borderRadius: '5px' }}>
            <MoreHorizontal20Regular style={{ fontSize: 14, color: '#9ca3af' }} />
          </button>
        </MenuTrigger>
        <MenuPopover>
          <MenuList>
            <MenuItem icon={<DeleteRegular />} onClick={() => onDelete(doc)}>Delete</MenuItem>
          </MenuList>
        </MenuPopover>
      </Menu>
    </div>
  );
}

export default function DocumentsSection({ refreshKey }) {
  const styles = useStyles();
  const { activeUserId } = useUser();
  const fileRef = useRef(null);

  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadFilename, setUploadFilename] = useState('');
  const [uploadSuccess, setUploadSuccess] = useState(null);
  const [uploadError, setUploadError] = useState(null);
  const [internalRefresh, setInternalRefresh] = useState(0);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isExpanded, setIsExpanded] = useState(true);

  const loadDocuments = useCallback(async () => {
    if (!activeUserId) return;
    try {
      setLoading(true);
      setError(null);
      const data = await api.listDocuments(activeUserId);
      setDocuments(data || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [activeUserId]);

  useEffect(() => { loadDocuments(); }, [loadDocuments, refreshKey, internalRefresh]);
  useEffect(() => {
    if (!uploadSuccess) return;
    const t = setTimeout(() => setUploadSuccess(null), 5000);
    return () => clearTimeout(t);
  }, [uploadSuccess]);
  useEffect(() => {
    if (!uploadError) return;
    const t = setTimeout(() => setUploadError(null), 6000);
    return () => clearTimeout(t);
  }, [uploadError]);

  const processFile = async (file) => {
    if (!file) return;
    setUploading(true);
    setUploadFilename(file.name);
    setUploadError(null);
    setUploadSuccess(null);
    try {
      const result = await api.uploadDocument(file);
      setUploadSuccess({ filename: file.name, chunks: result.chunks || 0 });
      setInternalRefresh(k => k + 1);
      // Auto-expand when a file is uploaded
      setIsExpanded(true);
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
      setUploadFilename('');
    }
  };

  const handleFileSelected = (e) => {
    processFile(e.target.files?.[0]);
    e.target.value = '';
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    processFile(e.dataTransfer.files?.[0]);
  };

  const handleDelete = async (doc) => {
    if (!confirm(`Delete "${doc.filename}"? This cannot be undone.`)) return;
    try {
      await api.deleteDocument(doc.document_id);
      setInternalRefresh(k => k + 1);
    } catch (err) {
      alert(`Failed: ${err.message}`);
    }
  };

  return (
    <div className={styles.section}>
      <style>{`
        @keyframes toastSlideIn { from{opacity:0;transform:translateY(-8px)} to{opacity:1;transform:translateY(0)} }
        @keyframes progressPulse { 0%{opacity:1;width:30%} 50%{opacity:.8;width:70%} 100%{opacity:1;width:30%} }
        .upload-btn-hover:hover { background: linear-gradient(135deg, #e0e7ff, #ede9fe) !important; border-color: #818cf8 !important; box-shadow: 0 2px 8px rgba(99,102,241,.25) !important; transform: translateY(-1px); }
      `}</style>

      <div className={styles.sectionHeader}>
        {/* Clickable label area to toggle collapse */}
        <div
          className={styles.sectionLabel}
          onClick={() => setIsExpanded(prev => !prev)}
          title={isExpanded ? 'Collapse documents' : 'Expand documents'}
        >
          <DocumentMultipleRegular style={{ fontSize: 13 }} />
          Documents
          {documents.length > 0 && (
            <span className={styles.countBadge}>{documents.length}</span>
          )}
          <span className={styles.collapseChevron} style={{ transform: isExpanded ? 'rotate(0deg)' : 'rotate(-90deg)' }}>
            <ChevronDown16Regular style={{ fontSize: 11 }} />
          </span>
        </div>

        {/* Upload button always visible */}
        <div className={styles.headerRight}>
          <button
            className={`${styles.uploadBtn} upload-btn-hover`}
            onClick={() => fileRef.current?.click()}
            disabled={uploading}
            title="Upload a document">
            {uploading
              ? <Spinner size="tiny" />
              : <DocumentAdd20Regular style={{ fontSize: 16 }} />}
            Upload
          </button>
        </div>
      </div>

      <input type="file" ref={fileRef} className={styles.hiddenInput} accept={ACCEPTED} onChange={handleFileSelected} />

      {/* Upload toasts always visible (outside collapse) */}
      {uploading && (
        <div className={`${styles.toast} ${styles.toastProgress}`}>
          <Spinner size="tiny" />
          <div className={styles.toastMsg}>
            <div>Indexing {uploadFilename}…</div>
            <div className={styles.progressBar}><div className={styles.progressFill} /></div>
          </div>
        </div>
      )}
      {uploadSuccess && (
        <div className={`${styles.toast} ${styles.toastSuccess}`}>
          <CheckmarkCircle20Regular style={{ flexShrink: 0 }} />
          <span className={styles.toastMsg}>✓ {uploadSuccess.filename} — {uploadSuccess.chunks} chunks indexed</span>
        </div>
      )}
      {uploadError && (
        <div className={`${styles.toast} ${styles.toastError}`}>
          <DismissCircle20Regular style={{ flexShrink: 0 }} />
          <span className={styles.toastMsg}>{uploadError}</span>
        </div>
      )}

      {/* Collapsible body */}
      <div
        className={styles.collapsibleBody}
        style={{
          maxHeight: isExpanded ? '1000px' : '0px',
          opacity: isExpanded ? 1 : 0,
          pointerEvents: isExpanded ? 'auto' : 'none',
        }}
      >
        {/* Drop zone (shown when no docs) */}
        {documents.length === 0 && !loading && (
          <div
            className={mergeClasses(styles.dropZone, isDragOver && styles.dropZoneActive)}
            onClick={() => fileRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}>
            <div style={{ fontSize: 22, marginBottom: 4 }}>📎</div>
            <div style={{ fontWeight: 600 }}>Drop a file here</div>
            <div style={{ fontSize: 12, opacity: .7, marginTop: 2 }}>PDF, DOCX, TXT, PPTX, XLSX…</div>
          </div>
        )}

        {loading && <div className={styles.loadingBox}><Spinner size="tiny" /></div>}
        {error && <div className={styles.toastError} style={{ padding: '8px 12px', borderRadius: 8, fontSize: 12, fontFamily: F }}>{error}</div>}

        {!loading && !error && documents.length > 0 && documents.map(doc => (
          <DocItem key={doc.document_id} doc={doc} onDelete={handleDelete} />
        ))}
      </div>
    </div>
  );
}
