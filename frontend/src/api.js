/**
 * api.js — HTTP client for the FastAPI backend.
 * Auth: token passed as ?token=... query parameter.
 */

const BASE = '/api';

// ── HELPER: get token from localStorage ──────────────────────────
function getToken() {
  return localStorage.getItem('rag_auth_token') || '';
}

// ── HELPER: standard request ──────────────────────────────────────
async function request(path, { method = 'GET', body, isFile = false } = {}) {
  const headers = {};

  if (body && !isFile) {
    headers['Content-Type'] = 'application/json';
  }

  const opts = { method, headers };

  if (body) {
    opts.body = isFile ? body : JSON.stringify(body);
  }

  // Append token as query param
  const token = getToken();
  const separator = path.includes('?') ? '&' : '?';
  const url = token ? `${BASE}${path}${separator}token=${token}` : `${BASE}${path}`;

  const response = await fetch(url, opts);

  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const err = await response.json();
      detail = err.detail || detail;
    } catch {}
    throw new Error(detail);
  }

  if (response.status === 204) return null;
  return response.json();
}


// ── STREAMING via Server-Sent Events ─────────────────────────────
async function streamQuery(sessionId, question, callbacks = {}) {
  const { onStatus, onToken, onDone, onError } = callbacks;

  try {
    const token = getToken();
    const url = `${BASE}/conversations/${sessionId}/query/stream${token ? `?token=${token}` : ''}`;

    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });

    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const err = await response.json();
        detail = err.detail || detail;
      } catch {}
      onError?.(detail);
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const messages = buffer.split('\n\n');
      buffer = messages.pop() || '';

      for (const message of messages) {
        if (!message.trim()) continue;
        const lines = message.split('\n');
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const dataStr = line.slice(6);
          try {
            const event = JSON.parse(dataStr);
            if (event.type === 'status')  onStatus?.(event.stage);
            else if (event.type === 'token') onToken?.(event.content);
            else if (event.type === 'done')  { onDone?.(event); return; }
            else if (event.type === 'error') { onError?.(event.message); return; }
          } catch (e) {
            console.error('Failed to parse SSE event:', dataStr, e);
          }
        }
      }
    }
  } catch (err) {
    onError?.(err.message || String(err));
  }
}


// ── API EXPORT ────────────────────────────────────────────────────
export const api = {
  // Auth
  register: (email, password, display_name) =>
    request('/auth/register', { method: 'POST', body: { email, password, display_name } }),
  login: (email, password) =>
    request('/auth/login', { method: 'POST', body: { email, password } }),

  // Users
  listUsers: () => request('/users'),

  // Conversations
  listConversations: () => request('/conversations'),
  createConversation: (title = null) =>
    request('/conversations', { method: 'POST', body: { title } }),
  getConversation: (sessionId) =>
    request(`/conversations/${sessionId}`),
  renameConversation: (sessionId, title) =>
    request(`/conversations/${sessionId}`, { method: 'PATCH', body: { title } }),
  deleteConversation: (sessionId) =>
    request(`/conversations/${sessionId}`, { method: 'DELETE' }),

  // Query
  query: (sessionId, question) =>
    request(`/conversations/${sessionId}/query`, {
      method: 'POST',
      body: { question },
    }),
  streamQuery,

  // Documents
  listDocuments: () => request('/documents'),
  uploadDocument: (file) => {
    const formData = new FormData();
    formData.append('file', file);
    return request('/documents', { method: 'POST', body: formData, isFile: true });
  },
  deleteDocument: (documentId) =>
    request(`/documents/${documentId}`, { method: 'DELETE' }),

  // Health
  health: () => request('/health'),
};

export default api;