import React, { useState, useEffect } from 'react';
import { Spinner } from '@fluentui/react-components';
import { useUser } from '../context/UserContext';
import api from '../api';

const styles = {
  page: {
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #fda085 100%)',
    backgroundSize: '400% 400%',
    animation: 'gradientShift 12s ease infinite',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontFamily: "'DM Sans', 'Segoe UI', sans-serif",
    position: 'relative',
    overflow: 'hidden',
  },
  blobs: {
    position: 'absolute',
    inset: 0,
    pointerEvents: 'none',
    overflow: 'hidden',
  },
  blob1: {
    position: 'absolute',
    width: '500px',
    height: '500px',
    borderRadius: '50%',
    background: 'rgba(255,255,255,0.08)',
    top: '-100px',
    left: '-150px',
    animation: 'blobFloat 8s ease-in-out infinite',
  },
  blob2: {
    position: 'absolute',
    width: '350px',
    height: '350px',
    borderRadius: '50%',
    background: 'rgba(255,255,255,0.06)',
    bottom: '-80px',
    right: '-100px',
    animation: 'blobFloat 10s ease-in-out infinite reverse',
  },
  blob3: {
    position: 'absolute',
    width: '200px',
    height: '200px',
    borderRadius: '50%',
    background: 'rgba(255,255,255,0.05)',
    top: '40%',
    right: '15%',
    animation: 'blobFloat 7s ease-in-out infinite 2s',
  },
  card: {
    background: 'rgba(255,255,255,0.97)',
    backdropFilter: 'blur(20px)',
    borderRadius: '28px',
    padding: '52px 48px',
    width: '100%',
    maxWidth: '460px',
    boxShadow: '0 32px 80px rgba(0,0,0,0.2), 0 0 0 1px rgba(255,255,255,0.3)',
    position: 'relative',
    zIndex: 1,
    animation: 'cardIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) both',
  },
  logoRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '14px',
    marginBottom: '32px',
  },
  logoIcon: {
    width: '48px',
    height: '48px',
    borderRadius: '14px',
    background: 'linear-gradient(135deg, #667eea, #764ba2)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '22px',
    boxShadow: '0 8px 20px rgba(102,126,234,0.4)',
    flexShrink: 0,
  },
  logoTextGroup: {},
  logoTitle: {
    fontSize: '20px',
    fontWeight: 700,
    color: '#1a1a2e',
    letterSpacing: '-0.3px',
    lineHeight: 1.2,
  },
  logoSub: {
    fontSize: '12px',
    color: '#8b8fa8',
    fontWeight: 500,
    letterSpacing: '0.5px',
    textTransform: 'uppercase',
  },
  heading: {
    fontSize: '28px',
    fontWeight: 800,
    color: '#0f0f23',
    marginBottom: '6px',
    letterSpacing: '-0.5px',
  },
  subheading: {
    fontSize: '15px',
    color: '#6b7280',
    marginBottom: '36px',
    fontWeight: 400,
  },
  label: {
    display: 'block',
    fontSize: '13px',
    fontWeight: 600,
    color: '#374151',
    marginBottom: '8px',
    letterSpacing: '0.1px',
  },
  userGrid: {
    display: 'grid',
    gap: '10px',
    marginBottom: '28px',
  },
  userCard: {
    display: 'flex',
    alignItems: 'center',
    gap: '14px',
    padding: '14px 18px',
    borderRadius: '14px',
    border: '2px solid #e5e7eb',
    cursor: 'pointer',
    transition: 'all 0.18s ease',
    background: '#fff',
    outline: 'none',
    textAlign: 'left',
    width: '100%',
  },
  userCardSelected: {
    borderColor: '#667eea',
    background: 'linear-gradient(135deg, rgba(102,126,234,0.06), rgba(118,75,162,0.06))',
    boxShadow: '0 0 0 3px rgba(102,126,234,0.15)',
  },
  userCardHover: {
    borderColor: '#a78bfa',
    background: '#faf5ff',
  },
  avatarCircle: {
    width: '42px',
    height: '42px',
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '16px',
    fontWeight: 700,
    color: '#fff',
    flexShrink: 0,
  },
  userInfo: {
    flex: 1,
  },
  userName: {
    fontSize: '14px',
    fontWeight: 600,
    color: '#111827',
    marginBottom: '2px',
  },
  userEmail: {
    fontSize: '12px',
    color: '#9ca3af',
  },
  checkIcon: {
    width: '22px',
    height: '22px',
    borderRadius: '50%',
    background: 'linear-gradient(135deg, #667eea, #764ba2)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#fff',
    fontSize: '12px',
    flexShrink: 0,
  },
  loginBtn: {
    width: '100%',
    padding: '15px',
    borderRadius: '14px',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: '#fff',
    fontSize: '15px',
    fontWeight: 700,
    border: 'none',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    boxShadow: '0 8px 25px rgba(102,126,234,0.45)',
    letterSpacing: '0.2px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '10px',
  },
  loginBtnDisabled: {
    background: '#e5e7eb',
    color: '#9ca3af',
    boxShadow: 'none',
    cursor: 'not-allowed',
  },
  footer: {
    marginTop: '24px',
    textAlign: 'center',
    fontSize: '12px',
    color: '#d1d5db',
  },
  footerStrong: {
    color: '#9ca3af',
    fontWeight: 600,
  },
  errorBox: {
    background: '#fef2f2',
    border: '1px solid #fecaca',
    borderRadius: '10px',
    padding: '12px 16px',
    color: '#dc2626',
    fontSize: '13px',
    marginBottom: '20px',
  },
};

const AVATAR_GRADIENTS = [
  'linear-gradient(135deg, #667eea, #764ba2)',
  'linear-gradient(135deg, #f093fb, #f5576c)',
  'linear-gradient(135deg, #4facfe, #00f2fe)',
  'linear-gradient(135deg, #43e97b, #38f9d7)',
  'linear-gradient(135deg, #fa709a, #fee140)',
];

function getInitials(name) {
  return name ? name.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2) : '?';
}

export default function LoginPage({ onLogin }) {
  const [users, setUsers]       = useState([]);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState(null);
  const [selected, setSelected] = useState(null);
  const [hoveredId, setHoveredId]   = useState(null);
  const [loggingIn, setLoggingIn]   = useState(false);
  const [loginError, setLoginError] = useState(null);

  // Load users list from backend on mount
  useEffect(() => {
    api.get('/auth/users')
      .then(res => {
        setUsers(res.data);
        if (res.data.length > 0) setSelected(res.data[0].id);
      })
      .catch(e => setError(e.response?.data?.detail || e.message))
      .finally(() => setLoading(false));
  }, []);

  const handleLogin = async () => {
    if (!selected) return;
    setLoggingIn(true);
    setLoginError(null);
    try {
      const res = await api.post('/auth/login', { user_id: selected });
      // res.data = { user_id, email, display_name, token }
      onLogin(res.data);
    } catch (e) {
      setLoginError(e.response?.data?.detail || e.message);
      setLoggingIn(false);
    }
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
        @keyframes gradientShift {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        @keyframes blobFloat {
          0%, 100% { transform: translateY(0) scale(1); }
          50% { transform: translateY(-30px) scale(1.05); }
        }
        @keyframes cardIn {
          0% { opacity: 0; transform: translateY(40px) scale(0.95); }
          100% { opacity: 1; transform: translateY(0) scale(1); }
        }
        .login-btn-active:hover {
          transform: translateY(-2px);
          box-shadow: 0 14px 35px rgba(102,126,234,0.55) !important;
        }
        .login-btn-active:active {
          transform: translateY(0);
        }
      `}</style>

      <div style={styles.page}>
        <div style={styles.blobs}>
          <div style={styles.blob1} />
          <div style={styles.blob2} />
          <div style={styles.blob3} />
        </div>

        <div style={styles.card}>
          <div style={styles.logoRow}>
            <div style={styles.logoIcon}>🧠</div>
            <div style={styles.logoTextGroup}>
              <div style={styles.logoTitle}>RAG Assistant</div>
              <div style={styles.logoSub}>AI Knowledge Platform</div>
            </div>
          </div>

          <h1 style={styles.heading}>Welcome back</h1>
          <p style={styles.subheading}>Select your account to continue</p>

          {loginError && <div style={styles.errorBox}>⚠️ {loginError}</div>}
          {error && <div style={styles.errorBox}>⚠️ Could not load users: {error}</div>}

          <span style={styles.label}>Choose account</span>

          {loading ? (
            <div style={{ display: 'flex', justifyContent: 'center', padding: '24px' }}>
              <Spinner size="medium" />
            </div>
          ) : (
            <div style={styles.userGrid}>
              {users.map((user, i) => {
                const isSelected = selected === user.id;
                const isHovered  = hoveredId === user.id;
                const name = user.display_name || user.id;
                return (
                  <button
                    key={user.id}
                    style={{
                      ...styles.userCard,
                      ...(isSelected ? styles.userCardSelected : {}),
                      ...(isHovered && !isSelected ? styles.userCardHover : {}),
                    }}
                    onClick={() => setSelected(user.id)}
                    onMouseEnter={() => setHoveredId(user.id)}
                    onMouseLeave={() => setHoveredId(null)}
                  >
                    <div style={{
                      ...styles.avatarCircle,
                      background: AVATAR_GRADIENTS[i % AVATAR_GRADIENTS.length],
                    }}>
                      {getInitials(name)}
                    </div>
                    <div style={styles.userInfo}>
                      <div style={styles.userName}>{name}</div>
                      {user.email && <div style={styles.userEmail}>{user.email}</div>}
                    </div>
                    {isSelected && (
                      <div style={styles.checkIcon}>✓</div>
                    )}
                  </button>
                );
              })}
            </div>
          )}

          <button
            style={{
              ...styles.loginBtn,
              ...((!selected || loggingIn) ? styles.loginBtnDisabled : {}),
            }}
            className={selected && !loggingIn ? 'login-btn-active' : ''}
            onClick={handleLogin}
            disabled={!selected || loggingIn}
          >
            {loggingIn ? (
              <Spinner size="tiny" appearance="inverted" />
            ) : (
              <>
                <span>Sign In</span>
                <span style={{ fontSize: '18px' }}>→</span>
              </>
            )}
          </button>

          <div style={styles.footer}>
            <span style={styles.footerStrong}>Dev Mode</span> · Multi-user RAG Platform
          </div>
        </div>
      </div>
    </>
  );
}