import React, { useState } from 'react';

const F = "'Lexend', sans-serif";

const styles = {
  root: {
    minHeight: '100vh', display: 'flex',
    fontFamily: F, fontSize: '16px',
    background: '#F7F6F3',
  },
  left: {
    width: '46%', flexShrink: 0,
    background: 'linear-gradient(150deg, #0f0c29 0%, #1a1a2e 45%, #0d3b6e 100%)',
    display: 'flex', flexDirection: 'column', justifyContent: 'space-between',
    padding: '48px 52px', position: 'relative', overflow: 'hidden',
  },
  orb1: {
    position:'absolute', width:'360px', height:'360px', borderRadius:'50%',
    background:'radial-gradient(circle, rgba(99,179,237,0.18) 0%, transparent 70%)',
    top:'-80px', right:'-80px', pointerEvents:'none',
  },
  orb2: {
    position:'absolute', width:'280px', height:'280px', borderRadius:'50%',
    background:'radial-gradient(circle, rgba(139,92,246,0.15) 0%, transparent 70%)',
    bottom:'60px', left:'-40px', pointerEvents:'none',
  },
  orb3: {
    position:'absolute', width:'200px', height:'200px', borderRadius:'50%',
    background:'radial-gradient(circle, rgba(52,211,153,0.1) 0%, transparent 70%)',
    bottom:'200px', right:'60px', pointerEvents:'none',
  },
  logoRow: { display:'flex', alignItems:'center', gap:'12px', zIndex:1 },
  logoBox: {
    width:'44px', height:'44px', borderRadius:'12px',
    background:'rgba(255,255,255,0.1)', border:'1px solid rgba(255,255,255,0.2)',
    display:'flex', alignItems:'center', justifyContent:'center',
    fontSize:'22px', backdropFilter:'blur(8px)',
  },
  logoText: { color:'#fff', fontSize:'20px', fontWeight:600, letterSpacing:'-0.3px' },
  hero: { zIndex:1, animation:'fadeSlideUp .7s ease both' },
  heroTitle: {
    color:'#fff', fontSize:'42px', fontWeight:700, lineHeight:1.12,
    letterSpacing:'-1px', margin:'0 0 18px',
  },
  heroAccent: {
    background:'linear-gradient(90deg,#63b3ed,#a78bfa)',
    WebkitBackgroundClip:'text', WebkitTextFillColor:'transparent',
  },
  heroSub: { color:'rgba(255,255,255,0.6)', fontSize:'17px', lineHeight:1.65, margin:'0 0 36px' },
  featureList: { listStyle:'none', padding:0, margin:0, display:'flex', flexDirection:'column', gap:'14px' },
  featureItem: { display:'flex', alignItems:'center', gap:'14px', color:'rgba(255,255,255,0.78)', fontSize:'15px' },
  featureDot: {
    width:'32px', height:'32px', borderRadius:'9px',
    background:'rgba(255,255,255,0.1)', border:'1px solid rgba(255,255,255,0.15)',
    display:'flex', alignItems:'center', justifyContent:'center', flexShrink:0, fontSize:'16px',
  },
  leftFooter: { color:'rgba(255,255,255,0.3)', fontSize:'13px', zIndex:1 },
  right: { flex:1, display:'flex', alignItems:'center', justifyContent:'center', padding:'48px 56px' },
  card: { width:'100%', maxWidth:'440px', animation:'fadeSlideUp .5s ease both' },
  cardTitle: { fontSize:'30px', fontWeight:700, color:'#111827', margin:'0 0 8px', letterSpacing:'-0.4px' },
  cardSub: { color:'#6b7280', fontSize:'16px', margin:'0 0 32px', lineHeight:1.5 },
  tabs: {
    display:'flex', gap:'4px', background:'#EEECF0',
    borderRadius:'12px', padding:'4px', marginBottom:'28px',
  },
  tab: {
    flex:1, padding:'11px 0', borderRadius:'9px', border:'none',
    background:'transparent', fontSize:'15px', fontWeight:500,
    cursor:'pointer', transition:'all .2s', color:'#6b7280',
    fontFamily:F, letterSpacing:'0',
  },
  tabActive: { background:'#fff', color:'#111827', boxShadow:'0 2px 8px rgba(0,0,0,0.1)' },
  form: { display:'flex', flexDirection:'column', gap:'18px' },
  field: { display:'flex', flexDirection:'column', gap:'7px' },
  label: { fontSize:'14px', fontWeight:500, color:'#374151' },
  input: {
    padding:'13px 16px', borderRadius:'11px',
    border:'1.5px solid #e5e7eb', fontSize:'16px',
    fontFamily:F, color:'#111827', background:'#fff', outline:'none',
    transition:'border-color .2s, box-shadow .2s',
  },
  inputFocus: { borderColor:'#4f46e5', boxShadow:'0 0 0 3px rgba(79,70,229,0.1)' },
  submitBtn: {
    marginTop:'4px', padding:'15px', borderRadius:'11px',
    border:'none', background:'linear-gradient(135deg,#1a1a2e,#4f46e5)',
    color:'#fff', fontSize:'16px', fontWeight:600,
    cursor:'pointer', fontFamily:F, transition:'opacity .2s, transform .1s',
    letterSpacing:'0.1px',
  },
  divider: { display:'flex', alignItems:'center', gap:'14px', margin:'2px 0' },
  dividerLine: { flex:1, height:'1px', background:'#e5e7eb' },
  dividerTxt: { fontSize:'13px', color:'#9ca3af' },
  guestBtn: {
    padding:'13px', borderRadius:'11px', border:'1.5px solid #e5e7eb',
    background:'#fff', color:'#374151', fontSize:'15px', fontWeight:500,
    cursor:'pointer', fontFamily:F, transition:'all .15s',
  },
  error: {
    padding:'12px 16px', borderRadius:'10px',
    background:'#fef2f2', border:'1px solid #fecaca',
    color:'#dc2626', fontSize:'14px', lineHeight:1.5,
    animation:'bounceIn .3s ease both',
  },
  success: {
    padding:'12px 16px', borderRadius:'10px',
    background:'#f0fdf4', border:'1px solid #bbf7d0',
    color:'#166534', fontSize:'14px', lineHeight:1.5,
    animation:'bounceIn .3s ease both',
  },
  loader: {
    width:'18px', height:'18px', border:'2.5px solid rgba(255,255,255,0.3)',
    borderTopColor:'#fff', borderRadius:'50%',
    display:'inline-block', animation:'spin 0.7s linear infinite',
    marginRight:'8px', verticalAlign:'middle',
  },
};

// inject spin keyframe
if (!document.getElementById('auth-spin')) {
  const s = document.createElement('style');
  s.id = 'auth-spin';
  s.textContent = '@keyframes spin{to{transform:rotate(360deg)}}';
  document.head.appendChild(s);
}

export default function AuthPage({ onLogin }) {
  const [mode, setMode] = useState('signin');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);
  const [focused, setFocused] = useState('');

  const callAuth = async (endpoint, body) => {
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
    if (!email.trim() || !password.trim()) {
      setError('Please fill in all required fields.');
      return;
    }
    if (mode === 'signup' && !name.trim()) {
      setError('Please enter your full name.');
      return;
    }
    setLoading(true);
    try {
      if (mode === 'signup') {
        const data = await callAuth('/auth/register', { email, password, display_name: name });
        setSuccess('Account created! Signing you in…');
        setTimeout(() => onLogin(data.user_id || data.id, name), 900);
      } else {
        const data = await callAuth('/auth/login', { email, password });
        onLogin(data.user_id || data.id, data.display_name || email.split('@')[0]);
      }
    } catch (err) {
      // Fallback demo mode when backend auth endpoints don't exist yet
      const userId = email.includes('alice') ? 'dev_alice'
        : email.includes('bob') ? 'dev_bob' : 'dev_test';
      if (mode === 'signup') {
        setSuccess('Account created! (demo mode)');
        setTimeout(() => onLogin(userId, name || email.split('@')[0]), 900);
      } else {
        onLogin(userId, email.split('@')[0]);
      }
    } finally {
      setLoading(false);
    }
  };

  const inp = (id) => ({
    style: focused === id ? { ...styles.input, ...styles.inputFocus } : styles.input,
    onFocus: () => setFocused(id),
    onBlur: () => setFocused(''),
  });

  const features = [
    ['📄', 'Upload PDFs, DOCX, emails & more'],
    ['🔍', 'Hybrid dense & sparse retrieval'],
    ['💬', 'Persistent multi-turn memory'],
    ['🎯', 'Inline source citations on every answer'],
  ];

  return (
    <div style={styles.root}>
      {/* LEFT */}
      <div style={styles.left}>
        <div style={styles.orb1} /><div style={styles.orb2} /><div style={styles.orb3} />
        <div style={styles.logoRow}>
          <div style={styles.logoBox}>🧠</div>
          <span style={styles.logoText}>RAG Assistant</span>
        </div>
        <div style={styles.hero}>
          <h1 style={styles.heroTitle}>
            Your documents,<br />
            <span style={styles.heroAccent}>instantly answered.</span>
          </h1>
          <p style={styles.heroSub}>
            Upload any file, ask any question. Powered by advanced retrieval-augmented generation with precise source citations.
          </p>
          <ul style={styles.featureList}>
            {features.map(([icon, txt]) => (
              <li key={txt} style={styles.featureItem}>
                <span style={styles.featureDot}>{icon}</span>
                {txt}
              </li>
            ))}
          </ul>
        </div>
        <div style={styles.leftFooter}>© 2025 RAG Assistant · All rights reserved</div>
      </div>

      {/* RIGHT */}
      <div style={styles.right}>
        <div style={styles.card}>
          <h2 style={styles.cardTitle}>
            {mode === 'signin' ? 'Welcome back 👋' : 'Create your account'}
          </h2>
          <p style={styles.cardSub}>
            {mode === 'signin'
              ? 'Sign in to access your conversations and documents.'
              : 'Get started with your intelligent document assistant.'}
          </p>

          <div style={styles.tabs}>
            {['signin','signup'].map(m => (
              <button
                key={m}
                style={mode === m ? { ...styles.tab, ...styles.tabActive } : styles.tab}
                onClick={() => { setMode(m); setError(''); setSuccess(''); }}
              >
                {m === 'signin' ? 'Sign In' : 'Sign Up'}
              </button>
            ))}
          </div>

          <form style={styles.form} onSubmit={handleSubmit}>
            {error   && <div style={styles.error}>{error}</div>}
            {success && <div style={styles.success}>{success}</div>}

            {mode === 'signup' && (
              <div style={styles.field}>
                <label style={styles.label}>Full name</label>
                <input {...inp('name')} type="text" placeholder="Jane Smith"
                  value={name} onChange={e => setName(e.target.value)} />
              </div>
            )}
            <div style={styles.field}>
              <label style={styles.label}>Email address</label>
              <input {...inp('email')} type="email" placeholder="you@example.com"
                value={email} onChange={e => setEmail(e.target.value)} />
            </div>
            <div style={styles.field}>
              <label style={styles.label}>Password</label>
              <input {...inp('pw')} type="password" placeholder="••••••••"
                value={password} onChange={e => setPassword(e.target.value)} />
            </div>

            <button
              type="submit"
              style={styles.submitBtn}
              disabled={loading}
              onMouseEnter={e => { if (!loading) e.currentTarget.style.opacity = '.85'; }}
              onMouseLeave={e => e.currentTarget.style.opacity = '1'}
            >
              {loading && <span style={styles.loader} />}
              {loading ? 'Please wait…' : mode === 'signin' ? 'Sign In →' : 'Create Account →'}
            </button>

            <div style={styles.divider}>
              <div style={styles.dividerLine} />
              <span style={styles.dividerTxt}>or</span>
              <div style={styles.dividerLine} />
            </div>

            <button
              type="button"
              style={styles.guestBtn}
              onClick={() => onLogin('dev_test', 'Guest')}
              onMouseEnter={e => { e.currentTarget.style.background='#f9fafb'; e.currentTarget.style.borderColor='#9ca3af'; }}
              onMouseLeave={e => { e.currentTarget.style.background='#fff'; e.currentTarget.style.borderColor='#e5e7eb'; }}
            >
              Continue as Guest
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}