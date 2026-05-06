import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Spinner, Menu, MenuTrigger, MenuPopover, MenuList, MenuItem,
  makeStyles, mergeClasses,
} from '@fluentui/react-components';
import {
  DocumentRegular, DocumentAdd20Regular, DocumentMultipleRegular,
  MoreHorizontal20Regular, DeleteRegular,
} from '@fluentui/react-icons';
import api from '../api';
import { useUser } from '../context/UserContext';

const ACCEPTED = '.pdf,.docx,.txt,.md,.csv,.xlsx,.pptx,.eml';

const useStyles = makeStyles({
  section: {
    marginBottom: '4px',
  },
  sectionHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '6px 4px 4px',
    marginBottom: '2px',
  },
  sectionLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    color: '#9ca3af',
    fontSize: '11px',
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: '0.6px',
    fontFamily: "'DM Sans', sans-serif",
  },
  uploadBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: '5px',
    border: 'none',
    background: 'transparent',
    color: '#6b7280',
    fontSize: '12px',
    cursor: 'pointer',
    fontFamily: "'DM Sans', sans-serif",
    padding: '2px 6px',
    borderRadius: '6px',
    transition: 'all 0.15s',
    ':hover': {
      background: '#f3f4f6',
      color: '#374151',
    },
  },
  hiddenInput: { display: 'none' },
  progressBox: {
    padding: '7px 10px',
    background: '#eff6ff',
    border: '1px solid #bfdbfe',
    borderRadius: '8px',
    marginBottom: '6px',
    display: 'flex',
    alignItems: 'center',
    gap: '7px',
    fontSize: '12px',
    color: '#1d4ed8',
    fontFamily: "'DM Sans', sans-serif",
  },
  successBox: {
    padding: '7px 10px',
    background: '#f0fdf4',
    border: '1px solid #bbf7d0',
    borderRadius: '8px',
    marginBottom: '6px',
    fontSize: '12px',
    color: '#166534',
    fontFamily: "'DM Sans', sans-serif",
  },
  errorBox: {
    padding: '7px 10px',
    background: '#fef2f2',
    border: '1px solid #fecaca',
    borderRadius: '8px',
    marginBottom: '6px',
    fontSize: '12px',
    color: '#dc2626',
    fontFamily: "'DM Sans', sans-serif",
  },
  item: {
    display: 'flex',
    alignItems: 'center',
    gap: '7px',
    padding: '6px 8px',
    borderRadius: '8px',
    cursor: 'default',
    transition: 'background 0.12s',
    minHeight: '30px',
    ':hover': {
      background: '#f3f4f6',
    },
  },
  itemIcon: {
    fontSize: '13px',
    color: '#9ca3af',
    flexShrink: 0,
  },
  itemTitle: {
    flex: 1,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    fontSize: '12px',
    color: '#374151',
    fontFamily: "'DM Sans', sans-serif",
  },
  itemMeta: {
    fontSize: '10px',
    color: '#c4c9d4',
    fontFamily: "'DM Sans', sans-serif",
    flexShrink: 0,
  },
  itemMenu: {
    flexShrink: 0,
    opacity: 0,
    transition: 'opacity 0.1s',
    minWidth: 'auto',
    height: '20px',
    padding: '0 3px',
  },
  itemMenuVisible: {
    opacity: 1,
  },
  emptyMsg: {
    color: '#c4c9d4',
    fontSize: '12px',
    textAlign: 'center',
    padding: '10px 8px',
    fontStyle: 'italic',
    fontFamily: "'DM Sans', sans-serif",
  },
  loadingBox: {
    display: 'flex',
    justifyContent: 'center',
    padding: '10px',
  },
});

function DocItem({ doc, onDelete }) {
  const styles = useStyles();
  const [hovered, setHovered] = useState(false);

  return (
    <div
      className={styles.item}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      title={doc.filename}
    >
      <DocumentRegular className={styles.itemIcon} />
      <span className={styles.itemTitle}>{doc.filename}</span>
      <span className={styles.itemMeta}>{doc.chunks}c</span>
      <Menu>
        <MenuTrigger disableButtonEnhancement>
          <button
            className={mergeClasses(styles.itemMenu, hovered && styles.itemMenuVisible)}
            style={{ border: 'none', background: 'transparent', cursor: 'pointer', display: 'flex', alignItems: 'center', borderRadius: '5px' }}
          >
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

  const handleFileSelected = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = '';
    setUploading(true);
    setUploadFilename(file.name);
    setUploadError(null);
    setUploadSuccess(null);
    try {
      const result = await api.uploadDocument(activeUserId, file);
      setUploadSuccess({ filename: file.name, chunks: result.chunks || 0 });
      setInternalRefresh(k => k + 1);
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
      setUploadFilename('');
    }
  };

  const handleDelete = async (doc) => {
    if (!confirm(`Delete "${doc.filename}"?`)) return;
    try {
      await api.deleteDocument(activeUserId, doc.document_id);
      setInternalRefresh(k => k + 1);
    } catch (err) {
      alert(`Failed: ${err.message}`);
    }
  };

  return (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <div className={styles.sectionLabel}>
          <DocumentMultipleRegular style={{ fontSize: 13 }} />
          Documents
        </div>
        <button
          className={styles.uploadBtn}
          onClick={() => fileRef.current?.click()}
          disabled={uploading}
        >
          {uploading ? <Spinner size="tiny" /> : <DocumentAdd20Regular style={{ fontSize: 14 }} />}
          Upload
        </button>
      </div>

      <input type="file" ref={fileRef} className={styles.hiddenInput} accept={ACCEPTED} onChange={handleFileSelected} />

      {uploading && (
        <div className={styles.progressBox}>
          <Spinner size="tiny" />
          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            Indexing {uploadFilename}...
          </span>
        </div>
      )}
      {uploadSuccess && (
        <div className={styles.successBox}>
          ✓ {uploadSuccess.filename} ({uploadSuccess.chunks} chunks)
        </div>
      )}
      {uploadError && <div className={styles.errorBox}>✗ {uploadError}</div>}

      {loading && <div className={styles.loadingBox}><Spinner size="tiny" /></div>}
      {error && <div className={styles.errorBox}>{error}</div>}

      {!loading && !error && documents.length === 0 && (
        <div className={styles.emptyMsg}>No documents yet</div>
      )}

      {!loading && !error && documents.map(doc => (
        <DocItem key={doc.document_id} doc={doc} onDelete={handleDelete} />
      ))}
    </div>
  );
}
