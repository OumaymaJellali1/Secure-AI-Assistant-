"""
memory — 4-layer memory system for the RAG pipeline.

Public API:
    from memory import MemoryManager, SessionManager

Layers:
    1. ConversationMemory (buffer)   — recent turns verbatim
    2. QueryRewriter                 — expands pronouns
    3. ConversationMemory (summary)  — compressed older turns
    4. UserMemory                    — persistent facts across sessions

Storage: PostgreSQL (via shared.db.engine)
"""

from memory.manager       import MemoryManager
from memory.sessions      import SessionManager
from memory.conversation  import ConversationMemory, Turn
from memory.user_facts    import UserMemory
from memory.rewriter      import QueryRewriter
from memory.permissions   import PermissionLoader

_all_ = [
    "MemoryManager",
    "SessionManager",
    "ConversationMemory",
    "UserMemory",
    "QueryRewriter",
    "PermissionLoader",
    "Turn",
]