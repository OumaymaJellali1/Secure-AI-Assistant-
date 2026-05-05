/**
 * DocumentsSection.jsx — Sidebar section for the user's uploaded documents.
 *
 * Features:
 *   • "Upload" button → opens hidden file picker
 *   • Progress indicator during upload (shows file is being processed)
 *   • Lists user's own documents (from document_permissions table)
 *   • Hover → ⋯ menu with Delete option
 *   • Auto-refreshes when user switches or after upload/delete
 *
 * Multi-user demo: each user only sees THEIR uploads.
 * Bob can't see Alice's docs in this list.
 */
import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Text,
  Button,
  Menu,
  MenuTrigger,
  MenuPopover,
  MenuList,
  MenuItem,
  Spinner,
  tokens,
  makeStyles,
  mergeClasses,
} from '@fluentui/react-components';
import {
  DocumentRegular,
  DocumentAdd20Regular,
  DocumentMultipleRegular,
  MoreHorizontal20Regular,
  DeleteRegular,
  Checkmark16Filled,
} from '@fluentui/react-icons';

import api from '../api';
import { useUser } from '../context/UserContext';


// Allowed extensions (must match backend ALLOWED_EXTENSIONS)
const ACCEPTED_EXTENSIONS = '.pdf,.docx,.txt,.md,.csv,.xlsx,.pptx,.eml';


const useStyles = makeStyles({
  section: {
    marginBottom: '8px',
  },
  sectionHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    padding: '8px 4px 4px',
    color: tokens.colorNeutralForeground3,
    fontSize: '11px',
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  uploadButton: {
    width: '100%',
    justifyContent: 'flex-start',
    backgroundColor: tokens.colorNeutralBackground1,
    border: `1px dashed ${tokens.colorNeutralStroke1}`,
    marginBottom: '8px',
    color: tokens.colorBrandForeground1,
    ':hover': {
      backgroundColor: tokens.colorBrandBackground2,
      borderColor: tokens.colorBrandStroke1,
    },
  },
  uploadButtonDisabled: {
    color: tokens.colorNeutralForeground3,
    ':hover': {
      backgroundColor: tokens.colorNeutralBackground1,
    },
  },
  hiddenInput: {
    display: 'none',
  },

  // Upload progress
  progressBox: {
    padding: '10px 12px',
    backgroundColor: tokens.colorBrandBackground2,
    border: `1px solid ${tokens.colorBrandStroke2}`,
    borderRadius: '6px',
    marginBottom: '8px',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  progressText: {
    fontSize: '12px',
    color: tokens.colorBrandForeground1,
    flex: 1,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  successBox: {
    padding: '8px 12px',
    backgroundColor: tokens.colorPaletteGreenBackground2,
    border: `1px solid ${tokens.colorPaletteGreenBorder1}`,
    borderRadius: '6px',
    marginBottom: '8px',
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    fontSize: '12px',
    color: tokens.colorPaletteGreenForeground1,
  },
  errorBox: {
    padding: '8px 12px',
    backgroundColor: tokens.colorPaletteRedBackground2,
    border: `1px solid ${tokens.colorPaletteRedBorder1}`,
    borderRadius: '6px',
    marginBottom: '8px',
    fontSize: '12px',
    color: tokens.colorPaletteRedForeground1,
  },

  // Document item
  item: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '6px 10px',
    borderRadius: '6px',
    cursor: 'default',
    color: tokens.colorNeutralForeground1,
    transitionProperty: 'background-color',
    transitionDuration: '0.1s',
    minHeight: '32px',
    ':hover': {
      backgroundColor: tokens.colorNeutralBackground1Hover,
    },
  },
  itemIcon: {
    fontSize: '14px',
    color: tokens.colorNeutralForeground3,
    flexShrink: 0,
  },
  itemTitle: {
    flex: 1,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    fontSize: '13px',
  },
  itemMeta: {
    fontSize: '11px',
    color: tokens.colorNeutralForeground3,
    flexShrink: 0,
  },
  itemMenu: {
    flexShrink: 0,
    opacity: 0,
    transitionProperty: 'opacity',
    transitionDuration: '0.1s',
  },
  itemMenuVisible: {
    opacity: 1,
  },

  // Empty / loading
  emptyMsg: {
    color: tokens.colorNeutralForeground3,
    fontSize: '12px',
    textAlign: 'center',
    padding: '12px 8px',
    fontStyle: 'italic',
  },
  loadingBox: {
    display: 'flex',
    justifyContent: 'center',
    padding: '12px',
  },
});


// ── INDIVIDUAL DOCUMENT ITEM ───────────────────────────────────────
function DocumentItem({ doc, onDelete }) {
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
          <Button
            className={mergeClasses(styles.itemMenu, hovered && styles.itemMenuVisible)}
            appearance="subtle"
            size="small"
            icon={<MoreHorizontal20Regular />}
          />
        </MenuTrigger>
        <MenuPopover>
          <MenuList>
            <MenuItem
              icon={<DeleteRegular />}
              onClick={() => onDelete(doc)}
            >
              Delete
            </MenuItem>
          </MenuList>
        </MenuPopover>
      </Menu>
    </div>
  );
}


// ── MAIN COMPONENT ─────────────────────────────────────────────────
export default function DocumentsSection({ refreshKey }) {
  const styles = useStyles();
  const { activeUserId } = useUser();
  const fileInputRef = useRef(null);

  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Upload state
  const [uploading, setUploading] = useState(false);
  const [uploadingFilename, setUploadingFilename] = useState('');
  const [uploadSuccess, setUploadSuccess] = useState(null); // { filename, chunks }
  const [uploadError, setUploadError] = useState(null);

  // Internal refresh trigger (after upload/delete)
  const [internalRefresh, setInternalRefresh] = useState(0);
  const triggerRefresh = () => setInternalRefresh(k => k + 1);

  // ── Load documents ──────────────────────────────────────────────
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

  useEffect(() => {
    loadDocuments();
  }, [loadDocuments, refreshKey, internalRefresh]);

  // Auto-clear success message after 5 seconds
  useEffect(() => {
    if (!uploadSuccess) return;
    const timer = setTimeout(() => setUploadSuccess(null), 5000);
    return () => clearTimeout(timer);
  }, [uploadSuccess]);

  // ── Upload handler ──────────────────────────────────────────────
  const handleUploadClick = () => {
    if (uploading) return;
    fileInputRef.current?.click();
  };

  const handleFileSelected = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    // Reset input so the same file can be re-selected later
    e.target.value = '';

    setUploading(true);
    setUploadingFilename(file.name);
    setUploadError(null);
    setUploadSuccess(null);

    try {
      const result = await api.uploadDocument(activeUserId, file);
      setUploadSuccess({
        filename: file.name,
        chunks: result.chunks || 0,
      });
      triggerRefresh();
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
      setUploadingFilename('');
    }
  };

  // ── Delete handler ──────────────────────────────────────────────
  const handleDelete = async (doc) => {
    if (!confirm(`Delete "${doc.filename}"? This cannot be undone.`)) return;
    
    try {
      await api.deleteDocument(activeUserId, doc.document_id);
      triggerRefresh();
    } catch (err) {
      alert(`Failed to delete: ${err.message}`);
    }
  };

  // ── Render ──────────────────────────────────────────────────────
  return (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <DocumentMultipleRegular style={{ fontSize: 14 }} />
        Your documents
      </div>

      {/* Hidden file input */}
      <input
        type="file"
        ref={fileInputRef}
        className={styles.hiddenInput}
        accept={ACCEPTED_EXTENSIONS}
        onChange={handleFileSelected}
      />

      {/* Upload button */}
      <Button
        className={mergeClasses(styles.uploadButton, uploading && styles.uploadButtonDisabled)}
        icon={uploading ? <Spinner size="tiny" /> : <DocumentAdd20Regular />}
        onClick={handleUploadClick}
        disabled={uploading}
      >
        {uploading ? 'Indexing...' : 'Upload document'}
      </Button>

      {/* Upload progress */}
      {uploading && (
        <div className={styles.progressBox}>
          <Spinner size="tiny" />
          <Text className={styles.progressText} title={uploadingFilename}>
            Processing {uploadingFilename}
          </Text>
        </div>
      )}

      {/* Upload success */}
      {uploadSuccess && (
        <div className={styles.successBox}>
          <Checkmark16Filled />
          <Text>Uploaded {uploadSuccess.filename} ({uploadSuccess.chunks} chunks)</Text>
        </div>
      )}

      {/* Upload error */}
      {uploadError && (
        <div className={styles.errorBox}>
          ❌ {uploadError}
        </div>
      )}

      {/* Loading state */}
      {loading && (
        <div className={styles.loadingBox}>
          <Spinner size="tiny" />
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className={styles.errorBox}>{error}</div>
      )}

      {/* Empty state */}
      {!loading && !error && documents.length === 0 && (
        <Text className={styles.emptyMsg}>
          No documents yet.<br />
          Click "Upload" to add one.
        </Text>
      )}

      {/* Document list */}
      {!loading && !error && documents.length > 0 && (
        <div>
          {documents.map((doc) => (
            <DocumentItem
              key={doc.document_id}
              doc={doc}
              onDelete={handleDelete}
            />
          ))}
        </div>
      )}
    </div>
  );
}