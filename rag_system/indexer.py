"""
Indexer module for RAG System.
Handles file system crawling, session parsing, and embedding generation.
"""

import json
import sqlite3
import time
import random
from pathlib import Path
from typing import List, Sequence

from .config import Config
from .utils import sha, now_iso, to_blob, truncate_for_embedding
from .metadata import extractor
from .chunker import chunker


class Indexer:
    """
    Manages the RAG index database and content crawling.
    Handles files, Claude sessions, and ChatGPT conversations.
    """

    def __init__(
        self,
        paths: Sequence[Path] = None,
        db: Path = None,
        quiet: bool = False,
    ):
        """
        Initialize the indexer.

        Args:
            paths: List of paths to index (default: Config.DEFAULT_PATHS)
            db: Database path (default: Config.INDEX_DB)
            quiet: Suppress output messages
        """
        self.paths = list(paths) if paths else list(Config.DEFAULT_PATHS)
        self.db_path = db if db else Config.INDEX_DB
        self.quiet = quiet

        if not quiet:
            print("🔍 Initializing Aurora RAG v5.0...")
            print(f"   Database: {self.db_path}")
            print(f"   Paths: {len(self.paths)} directories")

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

        if not quiet:
            print("✅ Database initialized")

    def _create_tables(self) -> None:
        """Create SQLite tables for files, sessions, and chat."""
        c = self.conn.cursor()

        c.execute(
            """CREATE TABLE IF NOT EXISTS files(
                   id TEXT PRIMARY KEY,
                   path TEXT,
                   mtime REAL,
                   size INT,
                   preview TEXT,
                   sha TEXT,
                   embedding BLOB,
                   metadata TEXT)"""
        )

        c.execute(
            """CREATE TABLE IF NOT EXISTS sessions(
                   id TEXT PRIMARY KEY,
                   summary TEXT,
                   ts TEXT,
                   size INT,
                   embedding BLOB,
                   metadata TEXT)"""
        )

        c.execute(
            """CREATE TABLE IF NOT EXISTS chat(
                   id TEXT PRIMARY KEY,
                   title TEXT,
                   ts TEXT,
                   msg_count INT,
                   content TEXT,
                   embedding BLOB,
                   metadata TEXT)"""
        )

        # Chunks table for semantic chunking
        c.execute(
            """CREATE TABLE IF NOT EXISTS chunks(
                   id TEXT PRIMARY KEY,
                   parent_id TEXT,
                   parent_type TEXT,
                   chunk_id INT,
                   text TEXT,
                   start_pos INT,
                   end_pos INT,
                   embedding BLOB,
                   metadata TEXT)"""
        )

        self.conn.commit()

        # Add metadata column to existing tables if missing
        self._add_metadata_columns()

        # Create indexes for common metadata queries (after columns exist)
        c.execute("CREATE INDEX IF NOT EXISTS idx_files_metadata ON files(metadata)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_ts ON sessions(ts)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_chat_ts ON chat(ts)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(parent_type)")

        self.conn.commit()

    def _add_metadata_columns(self) -> None:
        """Add metadata column to existing tables if it doesn't exist."""
        c = self.conn.cursor()

        for table in ['files', 'sessions', 'chat']:
            try:
                # Check if metadata column exists
                c.execute(f"SELECT metadata FROM {table} LIMIT 1")
            except sqlite3.OperationalError:
                # Column doesn't exist, add it
                c.execute(f"ALTER TABLE {table} ADD COLUMN metadata TEXT")

        self.conn.commit()

    def ensure_up_to_date(self, force: bool = False) -> None:
        """
        Ensure index is current by scanning filesystem and generating embeddings.

        Args:
            force: Force full rescan even if recently scanned
        """
        cur = self.conn.cursor()
        cur.execute("PRAGMA user_version")
        last_scan = cur.fetchone()[0]

        if not force and (time.time() - last_scan) < Config.FILE_SCAN_INTERVAL:
            return

        if not self.quiet:
            print("🔄 Scanning file system...")

        self._scan_files()
        self._scan_sessions()
        self._scan_chat()

        new_tokens = self._embed_missing()
        if new_tokens:
            usd = new_tokens / 1000 * Config.COST_PER_1K[Config.OPENAI_MODEL]
            print(f"💸 Embedded {new_tokens:,} tokens ≈ ${usd:.4f}")

        cur.execute(f"PRAGMA user_version = {int(time.time())}")
        self.conn.commit()

    def file_rows(self) -> List[sqlite3.Row]:
        """Get all file rows from database."""
        return self.conn.execute("SELECT * FROM files").fetchall()

    def session_rows(self) -> List[sqlite3.Row]:
        """Get all session rows from database."""
        return self.conn.execute("SELECT * FROM sessions").fetchall()

    def chat_rows(self) -> List[sqlite3.Row]:
        """Get all chat rows from database."""
        return self.conn.execute("SELECT * FROM chat").fetchall()

    def _scan_files(self) -> None:
        """Scan file system and index new/modified files."""
        cur = self.conn.cursor()
        cur.execute("SELECT id, sha, mtime FROM files")
        known = {row["id"]: (row["sha"], row["mtime"]) for row in cur.fetchall()}

        added, updated = 0, 0

        for root in self.paths:
            if not root.exists():
                continue

            if root.is_file():
                # Handle individual file
                fp = root
                if self._should_skip_file(fp):
                    continue

                if self._process_file(fp, known, cur):
                    if str(fp) in known:
                        updated += 1
                    else:
                        added += 1
            else:
                # Handle directory recursively
                for fp in root.rglob("*"):
                    if not fp.is_file():
                        continue

                    # Skip files in system directories
                    if any(part in Config.SKIP_DIRS for part in fp.parts):
                        continue

                    if self._should_skip_file(fp):
                        continue

                    if self._process_file(fp, known, cur):
                        if str(fp) in known:
                            updated += 1
                        else:
                            added += 1

        if (added or updated) and not self.quiet:
            print(f"   Files: +{added} new, {updated} updated")

    def _should_skip_file(self, fp: Path) -> bool:
        """Check if file should be skipped during indexing."""
        if fp.suffix.lower() in Config.SENSITIVE_EXTS:
            return True
        if fp.suffix.lower() not in Config.ALLOWED_EXTS:
            return True
        if fp.stat().st_size > Config.MAX_FILE_SIZE:
            return True
        return False

    def _process_file(self, fp: Path, known: dict, cur: sqlite3.Cursor) -> bool:
        """
        Process a single file for indexing.

        Returns:
            True if file was added/updated, False otherwise
        """
        fid = str(fp)
        mtime = fp.stat().st_mtime
        dig = sha(fp)

        # Skip if unchanged
        if fid in known and known[fid] == (dig, mtime):
            return False

        preview = self._read_preview(fp)

        # Extract metadata
        metadata = extractor.extract_all(preview)
        metadata_json = extractor.to_json(metadata)

        cur.execute(
            """REPLACE INTO files(id,path,mtime,size,preview,sha,embedding,metadata)
               VALUES(?,?,?,?,?,?,NULL,?)""",
            (fid, fid, mtime, fp.stat().st_size, preview, dig, metadata_json),
        )

        return True

    def _read_preview(self, fp: Path, max_chars: int = None) -> str:
        """
        Read file preview for indexing.

        Args:
            fp: File path
            max_chars: Maximum characters to read (default: Config.PREVIEW_CHARS)

        Returns:
            File preview text
        """
        if max_chars is None:
            max_chars = Config.PREVIEW_CHARS

        try:
            txt = fp.read_text("utf-8", "ignore")[:max_chars]
        except Exception:
            return ""

        # Clean up code files
        if fp.suffix.lower() in {".py", ".js", ".sh"}:
            lines = [l.strip() for l in txt.splitlines()]
            txt = "\n".join(l for l in lines if l and not l.startswith("#"))

        return txt

    def _scan_sessions(self) -> None:
        """Scan Claude session directories for conversation files."""
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM sessions")
        known = {r["id"] for r in cur.fetchall()}

        new = 0
        for session_dir in Config.SESSION_DIRS:
            if not session_dir.exists():
                continue

            for jf in session_dir.glob("*.jsonl"):
                sid = jf.stem
                if sid in known:
                    continue

                try:
                    if not self.quiet and new % 10 == 0:
                        print(f"   Processing session {new}/{len(list(session_dir.glob('*.jsonl')))}: {sid[:20]}...")

                    meta = json.loads(jf.open().readline() or "{}")
                    summary = meta.get("summary", "No summary")
                    ts = meta.get("ts", now_iso())

                    # Read FULL content for chunking (no preview limit)
                    full_content = jf.read_text("utf-8", "ignore")

                    if not self.quiet and new == 0:
                        print(f"   Session size: {len(full_content)} chars")

                    # Extract metadata from full content
                    metadata = extractor.extract_all(full_content)
                    # Add session timestamp to metadata
                    if 'dates' in metadata and not metadata['dates']:
                        # If no dates found, use session timestamp
                        if ts:
                            try:
                                date_str = ts.split('T')[0]
                                metadata['dates'] = [date_str]
                            except:
                                pass
                    metadata_json = extractor.to_json(metadata)

                    # Create semantic chunks from full content
                    if not self.quiet and new == 0:
                        print(f"   Creating chunks...")
                    chunks = chunker.chunk_text(full_content, sid, chunk_type='session')

                    if not self.quiet and new == 0:
                        print(f"   Created {len(chunks)} chunks")

                    # Store chunks in database
                    for chunk in chunks:
                        chunk_metadata = extractor.extract_all(chunk.text)
                        chunk_metadata_json = extractor.to_json(chunk_metadata)

                        chunk_full_id = f"{sid}_chunk_{chunk.chunk_id}"
                        cur.execute(
                            """INSERT OR IGNORE INTO chunks(id,parent_id,parent_type,chunk_id,text,start_pos,end_pos,embedding,metadata)
                               VALUES(?,?,?,?,?,?,?,NULL,?)""",
                            (chunk_full_id, sid, 'session', chunk.chunk_id, chunk.text,
                             chunk.start_pos, chunk.end_pos, chunk_metadata_json)
                        )

                except Exception as e:
                    if not self.quiet:
                        print(f"   ❌ ERROR processing {sid}: {e}")
                    summary = "Parse error"
                    ts = now_iso()
                    metadata_json = "{}"

                # Still store parent session for reference
                cur.execute(
                    "INSERT OR IGNORE INTO sessions(id,summary,ts,size,embedding,metadata) VALUES(?,?,?,?,NULL,?)",
                    (sid, summary, ts, jf.stat().st_size, metadata_json),
                )
                new += 1

        if new and not self.quiet:
            print(f"   Sessions: +{new}")

    def _scan_chat(self) -> None:
        """Scan ChatGPT conversation index."""
        if not Config.CHATGPT_INDEX.exists():
            return

        try:
            data = json.loads(Config.CHATGPT_INDEX.read_text("utf-8"))
        except Exception:
            return

        cur = self.conn.cursor()
        cur.execute("SELECT id FROM chat")
        known = {r["id"] for r in cur.fetchall()}

        added = 0
        for conv in data:
            cid = f"chatgpt_{conv['id']}"
            if cid in known:
                continue

            content = conv.get("content", "")[:8000]
            metadata = extractor.extract_all(content)

            # Add create_date to metadata if available
            create_date = conv.get("create_date", "")
            if create_date and 'dates' in metadata:
                try:
                    date_str = create_date.split('T')[0]
                    if date_str not in metadata['dates']:
                        metadata['dates'].insert(0, date_str)
                except:
                    pass
            metadata_json = extractor.to_json(metadata)

            cur.execute(
                """INSERT INTO chat(id,title,ts,msg_count,content,embedding,metadata)
                   VALUES(?,?,?,?,?,NULL,?)""",
                (
                    cid,
                    conv.get("title", ""),
                    create_date,
                    conv.get("message_count", 0),
                    content,
                    metadata_json,
                ),
            )
            added += 1

        if added and not self.quiet:
            print(f"   ChatGPT: +{added}")

    def _embed_missing(self) -> int:
        """
        Generate embeddings for documents missing them.

        Returns:
            Total tokens processed
        """
        api_key = Config.get_openai_api_key()
        if not api_key:
            if not self.quiet:
                print("⚠️  OPENAI_API_KEY not set – heuristic mode only")
            return 0

        try:
            from openai import OpenAI
        except ImportError:
            print("⚠️  openai package missing – pip install openai")
            return 0

        client = OpenAI(api_key=api_key)
        cur = self.conn.cursor()

        # Collect all documents needing embeddings
        pending: List[tuple] = []
        for tbl, col in [
            ("files", "preview"),
            ("sessions", "summary"),
            ("chat", "content"),
            ("chunks", "text"),  # Add chunks for embedding
        ]:
            cur.execute(f"SELECT id,{col} FROM {tbl} WHERE embedding IS NULL")
            pending += [(tbl, r["id"], r[col]) for r in cur.fetchall()]

        if not pending:
            if not self.quiet:
                print("✅ No new documents to embed")
            return 0

        if not self.quiet:
            print(f"🧠 Embedding {len(pending)} documents...")
            print(f"   Using OpenAI model: {Config.OPENAI_MODEL}")
            print(f"   Rate limit: {Config.EMBEDDING_RATE_LIMIT_RPM} RPM")

        total_tokens = 0

        # Process with rate limiting and retry logic
        for batch_start in range(0, len(pending), Config.EMBEDDING_BATCH_SIZE):
            batch = pending[batch_start : batch_start + Config.EMBEDDING_BATCH_SIZE]

            for i, (tbl, rid, text) in enumerate(batch):
                # Truncate text to safe token limit
                safe_text = truncate_for_embedding(text or "empty", Config.MAX_EMBEDDING_TOKENS)

                # Retry with exponential backoff
                for retry in range(Config.MAX_RETRIES + 1):
                    try:
                        resp = client.embeddings.create(input=safe_text, model=Config.OPENAI_MODEL)
                        total_tokens += resp.usage.total_tokens
                        cur.execute(
                            f"UPDATE {tbl} SET embedding=? WHERE id=?",
                            (to_blob(resp.data[0].embedding), rid),
                        )

                        # Show progress
                        global_idx = batch_start + i
                        if global_idx % 50 == 0 and global_idx > 0:
                            print(f"   Embedded {global_idx}/{len(pending)} documents...")

                        # Rate limiting delay
                        time.sleep(Config.EMBEDDING_DELAY_MS / 1000)
                        break

                    except Exception as e:
                        if retry < Config.MAX_RETRIES and ("429" in str(e) or "rate limit" in str(e).lower()):
                            # Exponential backoff with jitter
                            wait_time = (Config.RETRY_DELAY_BASE ** retry) + random.uniform(0, 1)
                            print(
                                f"   Rate limit hit (retry {retry+1}/{Config.MAX_RETRIES}), waiting {wait_time:.1f}s..."
                            )
                            time.sleep(wait_time)
                        elif retry < Config.MAX_RETRIES and "400" in str(e):
                            # Token limit exceeded, try shorter text
                            safe_text = safe_text[: len(safe_text) // 2]
                            print(f"   Token limit exceeded, retrying with shorter text ({len(safe_text)} chars)...")
                            continue
                        else:
                            print(f"⚠️  embedding failed for {tbl}/{rid}: {e}")
                            break

            # Delay between batches
            if batch_start + Config.EMBEDDING_BATCH_SIZE < len(pending):
                time.sleep(1)

        self.conn.commit()
        return total_tokens