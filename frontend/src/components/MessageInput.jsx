/**
 * MessageInput.jsx — Text input + send button at bottom of chat panel.
 *
 * Features:
 *   • Multi-line auto-grow textarea
 *   • Enter to send, Shift+Enter for new line
 *   • Disabled while waiting for response
 *   • Send button has loading spinner
 */
import React, { useState, useRef, useEffect } from 'react';
import {
  Button,
  Textarea,
  Spinner,
  tokens,
  makeStyles,
} from '@fluentui/react-components';
import { Send24Regular } from '@fluentui/react-icons';


const useStyles = makeStyles({
  container: {
    padding: '16px 24px 20px',
    backgroundColor: tokens.colorNeutralBackground2,
  },
  inputRow: {
    display: 'flex',
    alignItems: 'flex-end',
    gap: '8px',
    backgroundColor: tokens.colorNeutralBackground1,
    border: `1px solid ${tokens.colorNeutralStroke1}`,
    borderRadius: '12px',
    padding: '8px 8px 8px 16px',
    transitionProperty: 'border-color',
    transitionDuration: '0.15s',
    ':focus-within': {
      borderColor: tokens.colorBrandStroke1,
    },
  },
  textarea: {
    flex: 1,
    border: 'none',
    backgroundColor: 'transparent',
    resize: 'none',
    minHeight: '24px',
    maxHeight: '200px',
    fontSize: '14px',
    fontFamily: 'inherit',
    outline: 'none',
    padding: '8px 0',
    color: tokens.colorNeutralForeground1,
    overflowY: 'auto',
    '&::placeholder': {
      color: tokens.colorNeutralForeground4,
    },
  },
  sendButton: {
    flexShrink: 0,
    minWidth: '40px',
    height: '40px',
  },
  hint: {
    fontSize: '11px',
    color: tokens.colorNeutralForeground4,
    marginTop: '6px',
    paddingLeft: '4px',
  },
});


export default function MessageInput({ onSend, disabled }) {
  const styles = useStyles();
  const [value, setValue] = useState('');
  const textareaRef = useRef(null);

  // Auto-grow the textarea as user types
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 200) + 'px';
  }, [value]);

  const handleSend = () => {
    const text = value.trim();
    if (!text || disabled) return;
    onSend(text);
    setValue('');
  };

  const handleKeyDown = (e) => {
    // Enter = send, Shift+Enter = new line
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.inputRow}>
        <textarea
          ref={textareaRef}
          className={styles.textarea}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={disabled ? 'Waiting for response...' : 'Type a message... (Enter to send, Shift+Enter for new line)'}
          disabled={disabled}
          rows={1}
        />
        <Button
          className={styles.sendButton}
          appearance="primary"
          icon={disabled ? <Spinner size="tiny" appearance="inverted" /> : <Send24Regular />}
          onClick={handleSend}
          disabled={disabled || !value.trim()}
          title="Send (Enter)"
        />
      </div>
    </div>
  );
}