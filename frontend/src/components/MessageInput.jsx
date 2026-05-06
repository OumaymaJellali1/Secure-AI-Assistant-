import React, { useState, useRef, useEffect } from 'react';
import { makeStyles } from '@fluentui/react-components';
import { Send24Regular } from '@fluentui/react-icons';

const useStyles = makeStyles({
  container: {
    padding: '12px 28px 18px',
    background: '#F7F6F3',
    flexShrink: 0,
  },
  inner: {
    maxWidth: '800px',
    margin: '0 auto',
  },
  box: {
    display: 'flex',
    alignItems: 'flex-end',
    gap: '10px',
    background: '#ffffff',
    border: '1.5px solid #e5e7eb',
    borderRadius: '14px',
    padding: '10px 10px 10px 16px',
    transition: 'border-color 0.15s, box-shadow 0.15s',
    ':focus-within': {
      borderColor: '#1a1a2e',
      boxShadow: '0 0 0 3px rgba(26,26,46,0.06)',
    },
  },
  textarea: {
    flex: 1,
    border: 'none',
    background: 'transparent',
    resize: 'none',
    minHeight: '22px',
    maxHeight: '180px',
    fontSize: '14px',
    fontFamily: "'DM Sans', 'Segoe UI', sans-serif",
    color: '#1a1a2e',
    outline: 'none',
    padding: '4px 0',
    lineHeight: '1.5',
    overflowY: 'auto',
    '::placeholder': {
      color: '#c4c9d4',
    },
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
    fontSize: '16px',
    ':disabled': {
      opacity: 0.4,
      cursor: 'not-allowed',
    },
    ':hover:not(:disabled)': {
      opacity: 0.85,
    },
    ':active:not(:disabled)': {
      transform: 'scale(0.95)',
    },
  },
  hint: {
    textAlign: 'center',
    fontSize: '11px',
    color: '#c4c9d4',
    marginTop: '6px',
    fontFamily: "'DM Sans', sans-serif",
  },
});

export default function MessageInput({ onSend, disabled }) {
  const styles = useStyles();
  const [value, setValue] = useState('');
  const textareaRef = useRef(null);

  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 180) + 'px';
  }, [value]);

  const handleSend = () => {
    const text = value.trim();
    if (!text || disabled) return;
    onSend(text);
    setValue('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.inner}>
        <div className={styles.box}>
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
            disabled={disabled || !value.trim()}
          >
            <Send24Regular style={{ fontSize: 16 }} />
          </button>
        </div>
        <div className={styles.hint}>Enter to send · Shift+Enter for new line</div>
      </div>
    </div>
  );
}
