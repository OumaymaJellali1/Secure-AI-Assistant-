/**
 * api.js — HTTP client for the FastAPI backend.
 *
 * NEW: streamQuery() — uses Server-Sent Events for streaming responses.
 */

const BASE = '/api';

// ── HELPER: standard request ──────────────────────────────────────
async function request(path, { method = 'GET', body, userId, isFile = false } = {}) {
  const headers = {};
  
  if (userId) {
    headers['X-Dev-User-Id'] = userId;
  }
  
  if (body && !isFile) {
    headers['Content-Type'] = 'application/json';
  }

  const opts = { method, headers };

  if (body) {
    opts.body = isFile ? body : JSON.stringify(body);
  }

  const response = await fetch(`${BASE}${path}`, opts);

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


// ── STREAMING via Server-Sent Events ──────────────────────────────
/**
 * Stream a query response token-by-token.
 *
 * @param {string} userId       - active user id
 * @param {string} sessionId    - conversation id
 * @param {string} question     - user's question
 * @param {object} callbacks    - { onStatus, onToken, onDone, onError }
 *   onStatus(stage)            - "retrieving" | "generating"
 *   onToken(text)              - called for each token chunk
 *   onDone({ sources, ... })   - called when stream ends with metadata
 *   onError(message)           - called on any error
 */
async function streamQuery(userId, sessionId, question, callbacks = {}) {
  const { onStatus, onToken, onDone, onError } = callbacks;

  try {
    const response = await fetch(
      `${BASE}/conversations/${sessionId}/query/stream`,
      {
        method: 'POST',
        headers: {
          'X-Dev-User-Id': userId,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question }),
      }
    );

    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const err = await response.json();
        detail = err.detail || detail;
      } catch {}
      onError?.(detail);
      return;
    }

    // Read the SSE stream
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE messages are separated by double newline
      const messages = buffer.split('\n\n');
      buffer = messages.pop() || ''; // keep incomplete message in buffer

      for (const message of messages) {
        if (!message.trim()) continue;

        // Each message has lines like "data: {...}"
        const lines = message.split('\n');
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const dataStr = line.slice(6);

          try {
            const event = JSON.parse(dataStr);

            if (event.type === 'status') {
              onStatus?.(event.stage);
            } else if (event.type === 'token') {
              onToken?.(event.content);
            } else if (event.type === 'done') {
              onDone?.(event);
              return; // stream complete
            } else if (event.type === 'error') {
              onError?.(event.message);
              return;
            }
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
  // Users
  listUsers: () => request('/users'),
  getMe: (userId) => request('/users/me', { userId }),

  // Conversations
  listConversations: (userId) => request('/conversations', { userId }),
  createConversation: (userId, title = null) =>
    request('/conversations', { method: 'POST', body: { title }, userId }),
  getConversation: (userId, sessionId) =>
    request(`/conversations/${sessionId}`, { userId }),
  renameConversation: (userId, sessionId, title) =>
    request(`/conversations/${sessionId}`, {
      method: 'PATCH',
      body: { title },
      userId,
    }),
  deleteConversation: (userId, sessionId) =>
    request(`/conversations/${sessionId}`, {
      method: 'DELETE',
      userId,
    }),

  // Non-streaming query (kept for fallback)
  query: (userId, sessionId, question, stream = false) =>
    request(`/conversations/${sessionId}/query`, {
      method: 'POST',
      body: { question, stream },
      userId,
    }),

  // STREAMING query
  streamQuery,

  // Documents
  listDocuments: (userId) => request('/documents', { userId }),
  uploadDocument: (userId, file) => {
    const formData = new FormData();
    formData.append('file', file);
    return request('/documents', {
      method: 'POST',
      body: formData,
      userId,
      isFile: true,
    });
  },
  deleteDocument: (userId, documentId) =>
    request(`/documents/${documentId}`, {
      method: 'DELETE',
      userId,
    }),

  // Health
  health: () => request('/health'),
};

export default api;