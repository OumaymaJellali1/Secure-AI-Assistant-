import React, { useState, useEffect } from 'react';

/* ── FONT INJECTION ─────────────────────────────────────────────────────── */
if (!document.getElementById('lexend-font')) {
  const link = document.createElement('link');
  link.id = 'lexend-font';
  link.rel = 'stylesheet';
  link.href = 'https://fonts.googleapis.com/css2?family=Lexend:wght@300;400;500;600;700;800;900&family=Lexend+Deca:wght@400;500;600&display=swap';
  document.head.appendChild(link);
}

const inject = (id, css) => {
  if (!document.getElementById(id)) {
    const s = document.createElement('style');
    s.id = id;
    s.textContent = css;
    document.head.appendChild(s);
  }
};

inject('auth-animations', `
  @keyframes floatA { 0%,100%{transform:translateY(0) rotate(0deg)} 50%{transform:translateY(-28px) rotate(3deg)} }
  @keyframes floatB { 0%,100%{transform:translateY(0) rotate(0deg)} 50%{transform:translateY(22px) rotate(-2deg)} }
  @keyframes floatC { 0%,100%{transform:translateX(0)} 50%{transform:translateX(18px)} }
  @keyframes cardReveal { 0%{opacity:0;transform:translateY(32px) scale(.97)} 100%{opacity:1;transform:translateY(0) scale(1)} }
  @keyframes inputPop { 0%{transform:scale(.98)} 60%{transform:scale(1.01)} 100%{transform:scale(1)} }
  @keyframes spinArc { to{transform:rotate(360deg)} }
  @keyframes errorShake { 0%,100%{transform:translateX(0)} 20%,60%{transform:translateX(-6px)} 40%,80%{transform:translateX(6px)} }
  @keyframes successBounce { 0%{transform:scale(.8);opacity:0} 60%{transform:scale(1.08)} 100%{transform:scale(1);opacity:1} }
  @keyframes pulseGlow { 0%,100%{box-shadow:0 0 0 0 rgba(99,102,241,.4)} 50%{box-shadow:0 0 0 12px rgba(99,102,241,0)} }
  @keyframes particleDrift {
    0% { transform: translateY(100vh) translateX(0) rotate(0deg); opacity: 0; }
    10% { opacity: 1; }
    90% { opacity: 1; }
    100% { transform: translateY(-10vh) translateX(40px) rotate(720deg); opacity: 0; }
  }
  .auth-input:focus { animation: inputPop .2s ease; }
  .auth-btn-active:hover { transform: translateY(-2px) !important; }
  .auth-btn-active:active { transform: translateY(0px) !important; }
`);

const F = "'Lexend', 'Segoe UI', sans-serif";

// onLogin receives the full user object { user_id, email, display_name, token }
export default function AuthPage({ onLogin }) {
  const [mode, setMode] = useState('signin');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [focused, setFocused] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [particles, setParticles] = useState([]);

  useEffect(() => {
    setParticles(Array.from({ length: 12 }, (_, i) => ({
      id: i,
      left: `${Math.random() * 100}%`,
      size: `${6 + Math.random() * 10}px`,
      delay: `${Math.random() * 8}s`,
      duration: `${8 + Math.random() * 8}s`,
      color: ['rgba(99,102,241,.6)', 'rgba(139,92,246,.5)', 'rgba(59,130,246,.5)', 'rgba(16,185,129,.4)'][Math.floor(Math.random() * 4)],
    })));
  }, []);

  const callApi = async (endpoint, body) => {
    const res = await fetch(`/api${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `Error ${res.status}`);
    return data;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(''); setSuccess('');

    if (!email.trim()) { setError('Please enter your email address.'); return; }
    if (!password.trim()) { setError('Please enter your password.'); return; }
    if (mode === 'signup' && !name.trim()) { setError('Please enter your full name.'); return; }

    setLoading(true);
    try {
      if (mode === 'signup') {
        const data = await callApi('/auth/register', { email, password, display_name: name });
        // data = { user_id, email, display_name, token }
        setSuccess(`Welcome, ${data.display_name}! Signing you in…`);
        setTimeout(() => onLogin(data), 1200);
      } else {
        const data = await callApi('/auth/login', { email, password });
        // data = { user_id, email, display_name, token }
        onLogin(data);
      }
    } catch (err) {
      setError(err.message || 'Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Guest login — calls backend with dev_alice user ID
  const handleGuest = async () => {
    setLoading(true);
    try {
      const data = await callApi('/auth/login', { email: 'dev_alice', password: 'guest' });
      onLogin(data);
    } catch (err) {
      setError('Guest login failed. Please try signing in with an account.');
    } finally {
      setLoading(false);
    }
  };

  const features = [
    { icon: '📄', text: 'Upload PDFs, DOCX, emails & more' },
    { icon: '🔍', text: 'Hybrid dense & sparse retrieval' },
    { icon: '💬', text: 'Persistent multi-turn memory' },
    { icon: '🎯', text: 'Inline source citations on every answer' },
  ];

  const inp = (id) => ({
    className: 'auth-input',
    style: {
      width: '100%', padding: '15px 18px', borderRadius: '14px',
      border: `2px solid ${focused === id ? '#6366f1' : '#e5e7eb'}`,
      fontSize: '16px', fontFamily: F, color: '#111827',
      background: focused === id ? '#fafbff' : '#fff',
      outline: 'none', transition: 'all .2s', boxSizing: 'border-box',
      boxShadow: focused === id ? '0 0 0 4px rgba(99,102,241,.12)' : 'none',
    },
    onFocus: () => setFocused(id),
    onBlur: () => setFocused(''),
  });

  return (
    <div style={{ minHeight: '100vh', display: 'flex', fontFamily: F, background: '#F7F6F3', overflow: 'hidden' }}>

      {/* LEFT PANEL */}
      <div style={{
        width: '48%', flexShrink: 0, position: 'relative', overflow: 'hidden',
        background: 'linear-gradient(150deg, #0d0221 0%, #1a0533 30%, #0d1b4b 65%, #051640 100%)',
        display: 'flex', flexDirection: 'column', justifyContent: 'space-between',
        padding: '52px 56px',
      }}>
        <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
          <div style={{ position: 'absolute', width: 420, height: 420, borderRadius: '50%', top: -100, right: -80, background: 'radial-gradient(circle, rgba(99,102,241,.2) 0%, transparent 70%)', animation: 'floatA 9s ease-in-out infinite' }} />
          <div style={{ position: 'absolute', width: 300, height: 300, borderRadius: '50%', bottom: 40, left: -60, background: 'radial-gradient(circle, rgba(139,92,246,.18) 0%, transparent 70%)', animation: 'floatB 11s ease-in-out infinite' }} />
          <div style={{ position: 'absolute', width: 180, height: 180, borderRadius: '50%', top: '45%', right: '12%', background: 'radial-gradient(circle, rgba(59,130,246,.15) 0%, transparent 70%)', animation: 'floatC 7s ease-in-out infinite' }} />
          <div style={{ position: 'absolute', inset: 0, backgroundImage: 'linear-gradient(rgba(255,255,255,.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.03) 1px, transparent 1px)', backgroundSize: '48px 48px' }} />
          {particles.map(p => (
            <div key={p.id} style={{
              position: 'absolute', bottom: 0, left: p.left,
              width: p.size, height: p.size, borderRadius: '50%',
              background: p.color, animation: `particleDrift ${p.duration} ${p.delay} infinite linear`,
            }} />
          ))}
        </div>

        {/* Logo */}
        <div style={{ position: 'relative', zIndex: 1, display: 'flex', alignItems: 'center', gap: 14 }}>
          <div style={{
            width: 50, height: 50, borderRadius: 14,
            background: 'rgba(255,255,255,.1)', border: '1px solid rgba(255,255,255,.2)',
            backdropFilter: 'blur(8px)', display: 'flex', alignItems: 'center',
            justifyContent: 'center', fontSize: 24,
          }}>🧠</div>
          <div>
            <div style={{ color: '#fff', fontSize: 20, fontWeight: 700, letterSpacing: '-0.3px' }}>RAG Assistant</div>
            <div style={{ color: 'rgba(255,255,255,.45)', fontSize: 12, fontWeight: 500, letterSpacing: '1.5px', textTransform: 'uppercase' }}>AI Knowledge Platform</div>
          </div>
        </div>

        {/* Hero */}
        <div style={{ position: 'relative', zIndex: 1 }}>
          <h1 style={{ color: '#fff', fontSize: 46, fontWeight: 800, lineHeight: 1.1, letterSpacing: '-1.5px', margin: '0 0 20px', fontFamily: F }}>
            Your documents,<br />
            <span style={{ background: 'linear-gradient(90deg, #818cf8, #a78bfa, #60a5fa)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              instantly answered.
            </span>
          </h1>
          <p style={{ color: 'rgba(255,255,255,.6)', fontSize: 17, lineHeight: 1.7, margin: '0 0 40px', fontWeight: 400 }}>
            Upload any file, ask any question. Advanced retrieval-augmented generation with precise source citations.
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {features.map(({ icon, text }) => (
              <div key={text} style={{ display: 'flex', alignItems: 'center', gap: 16, color: 'rgba(255,255,255,.78)', fontSize: 15, fontWeight: 400 }}>
                <div style={{
                  width: 38, height: 38, borderRadius: 10, flexShrink: 0,
                  background: 'rgba(255,255,255,.08)', border: '1px solid rgba(255,255,255,.14)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 18,
                }}>{icon}</div>
                {text}
              </div>
            ))}
          </div>
        </div>

        <div style={{ position: 'relative', zIndex: 1, color: 'rgba(255,255,255,.25)', fontSize: 13 }}>
          © 2025 RAG Assistant · All rights reserved
        </div>
      </div>

      {/* RIGHT PANEL */}
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '48px 60px', overflowY: 'auto' }}>
        <div style={{ width: '100%', maxWidth: 460, animation: 'cardReveal .55s cubic-bezier(.34,1.56,.64,1) both' }}>

          <h2 style={{ fontSize: 32, fontWeight: 800, color: '#0f0f23', margin: '0 0 6px', letterSpacing: '-0.8px', fontFamily: F }}>
            {mode === 'signin' ? 'Welcome back 👋' : 'Create an account'}
          </h2>
          <p style={{ color: '#6b7280', fontSize: 16, margin: '0 0 32px', lineHeight: 1.5, fontWeight: 400 }}>
            {mode === 'signin' ? 'Sign in to access your conversations.' : 'Get started with your AI document assistant.'}
          </p>

          {/* Mode tabs */}
          <div style={{ display: 'flex', gap: 4, background: '#EEECF0', borderRadius: 14, padding: 4, marginBottom: 32 }}>
            {['signin', 'signup'].map(m => (
              <button key={m} onClick={() => { setMode(m); setError(''); setSuccess(''); }}
                style={{
                  flex: 1, padding: '13px 0', borderRadius: 11, border: 'none', cursor: 'pointer',
                  fontSize: 16, fontWeight: mode === m ? 600 : 500, fontFamily: F,
                  background: mode === m ? '#fff' : 'transparent',
                  color: mode === m ? '#111827' : '#6b7280',
                  boxShadow: mode === m ? '0 2px 10px rgba(0,0,0,.1)' : 'none',
                  transition: 'all .2s',
                }}>
                {m === 'signin' ? 'Sign In' : 'Sign Up'}
              </button>
            ))}
          </div>

          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

            {error && (
              <div style={{ padding: '14px 18px', borderRadius: 12, background: '#fef2f2', border: '1.5px solid #fecaca', color: '#dc2626', fontSize: 15, fontWeight: 500, animation: 'errorShake .4s ease', display: 'flex', alignItems: 'center', gap: 10 }}>
                ⚠️ {error}
              </div>
            )}
            {success && (
              <div style={{ padding: '14px 18px', borderRadius: 12, background: '#f0fdf4', border: '1.5px solid #bbf7d0', color: '#166534', fontSize: 15, fontWeight: 500, animation: 'successBounce .4s ease', display: 'flex', alignItems: 'center', gap: 10 }}>
                ✅ {success}
              </div>
            )}

            {mode === 'signup' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <label style={{ fontSize: 15, fontWeight: 600, color: '#374151', fontFamily: F }}>Full name</label>
                <input {...inp('name')} type="text" placeholder="Jane Smith" value={name} onChange={e => setName(e.target.value)} />
              </div>
            )}

            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <label style={{ fontSize: 15, fontWeight: 600, color: '#374151', fontFamily: F }}>Email address</label>
              <input {...inp('email')} type="email" placeholder="you@example.com" value={email} onChange={e => setEmail(e.target.value)} />
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <label style={{ fontSize: 15, fontWeight: 600, color: '#374151', fontFamily: F }}>Password</label>
              <input {...inp('pw')} type="password" placeholder="••••••••" value={password} onChange={e => setPassword(e.target.value)} />
              {mode === 'signup' && <span style={{ fontSize: 13, color: '#9ca3af' }}>At least 6 characters</span>}
            </div>

            <button
              type="submit"
              disabled={loading}
              className={!loading ? 'auth-btn-active' : ''}
              style={{
                marginTop: 4, padding: '17px', borderRadius: 14, border: 'none',
                background: loading ? '#e5e7eb' : 'linear-gradient(135deg, #1a1a2e 0%, #4f46e5 100%)',
                color: loading ? '#9ca3af' : '#fff', fontSize: 17, fontWeight: 700,
                cursor: loading ? 'not-allowed' : 'pointer', fontFamily: F,
                transition: 'all .2s', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10,
                animation: !loading ? 'pulseGlow 3s ease-in-out infinite' : 'none',
              }}>
              {loading && <span style={{ width: 18, height: 18, border: '2.5px solid rgba(255,255,255,.3)', borderTopColor: '#fff', borderRadius: '50%', display: 'inline-block', animation: 'spinArc .7s linear infinite' }} />}
              {loading ? 'Please wait…' : mode === 'signin' ? 'Sign In →' : 'Create Account →'}
            </button>

            <div style={{ display: 'flex', alignItems: 'center', gap: 14, margin: '2px 0' }}>
              <div style={{ flex: 1, height: 1, background: '#e5e7eb' }} />
              <span style={{ fontSize: 14, color: '#9ca3af' }}>or</span>
              <div style={{ flex: 1, height: 1, background: '#e5e7eb' }} />
            </div>

            <button
              type="button"
              onClick={handleGuest}
              disabled={loading}
              style={{
                padding: '15px', borderRadius: 14, border: '2px solid #e5e7eb',
                background: '#fff', color: '#374151', fontSize: 16, fontWeight: 500,
                cursor: loading ? 'not-allowed' : 'pointer', fontFamily: F, transition: 'all .15s',
              }}
              onMouseEnter={e => { e.currentTarget.style.background = '#f9fafb'; e.currentTarget.style.borderColor = '#d1d5db'; }}
              onMouseLeave={e => { e.currentTarget.style.background = '#fff'; e.currentTarget.style.borderColor = '#e5e7eb'; }}>
              Continue as Guest
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}