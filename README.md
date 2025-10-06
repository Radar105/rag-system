# RAG System v5.0

**Production-grade Retrieval-Augmented Generation system with hybrid search**

A modular, API-agnostic RAG implementation supporting multiple LLM providers (Claude, OpenAI, or custom). Built for real-world knowledge base applications with intelligent semantic search, cost optimization, and flexible configuration.

---

## Features

- **Hybrid Search**: Combines vector similarity (70%) + keyword matching (30%) for superior relevance
- **FAISS Acceleration**: Automatic GPU acceleration when available, graceful NumPy fallback
- **Multi-Source Indexing**: Files, conversation sessions, and structured data
- **Persistent Vector Database**: SQLite with float32 BLOB embeddings
- **Lazy Embedding Generation**: Only recomputes when content changes
- **Cost Tracking**: Real-time OpenAI embedding token/USD tracking
- **Modular Architecture**: Clean separation of concerns with full type hints
- **Tool Isolation**: Safe LLM integration with whitelisted directory access
- **API-Agnostic**: Supports Anthropic Claude, OpenAI, or custom LLM providers

---

## Quick Start

### Installation

```bash
# Install core dependencies
pip3 install openai numpy

# Optional: Install FAISS for GPU acceleration
pip3 install faiss-cpu  # or faiss-gpu for CUDA

# Optional: Install MCP SDK for Model Context Protocol support
pip3 install mcp
```

### Configuration

Set your API keys:
```bash
# For Claude (via Claude Code CLI)
export ANTHROPIC_API_KEY="your-key-here"

# For OpenAI (direct API)
export OPENAI_API_KEY="your-key-here"
export OPENAI_MODEL="gpt-5"  # Optional, defaults to gpt-5

# For custom LLM command
export RAG_LLM_COMMAND="your-custom-llm-cli"
```

### Basic Usage

```bash
# View index statistics
./rag_system.py stats

# Search your knowledge base (local, no LLM call)
./rag_system.py search "machine learning best practices"

# Query with LLM-powered answer (uses RAG context)
./rag_system.py query "What are the key ML deployment considerations?"

# Deep mode with extended context
./rag_system.py query --deep "Explain distributed training strategies"

# Disable file reading tools
./rag_system.py query --no-tools "Summarize the documentation"

# Force index rebuild
./rag_system.py rebuild
```

### MCP Server Usage

Run as a Model Context Protocol server for integration with Claude Code or other MCP clients:

```bash
# Start MCP server (stdio mode)
python3 mcp_server.py
```

**MCP Configuration** (add to your MCP settings):
```json
{
  "mcpServers": {
    "rag-system": {
      "command": "python3",
      "args": ["/path/to/rag-system/mcp_server.py"],
      "env": {
        "OPENAI_API_KEY": "your-openai-api-key"
      }
    }
  }
}
```

**Available MCP Tools**:
- `rag_search` - Search indexed documents with hybrid vector+keyword
- `rag_build_context` - Build RAG context for LLM consumption
- `rag_rebuild` - Force full index rebuild

**Available MCP Resources**:
- `rag://stats` - Index statistics (sessions, files, chats)
- `rag://config` - Current system configuration

**Example Usage in Claude Code**:
```
Use rag_search to find documents about "machine learning deployment"
Use rag_build_context with deep mode for "explain transformer architectures"
```

---

## Architecture

### Module Overview

```
rag_system/
├── config.py          # Centralized configuration (276 lines)
├── indexer.py         # Document indexing engine (367 lines)
├── retriever.py       # Hybrid search system (413 lines)
├── prompt_builder.py  # RAG context assembly (196 lines)
├── llm_runner.py      # LLM provider interface (182 lines)
├── chunker.py         # Document chunking logic
├── metadata.py        # Metadata extraction and management
└── utils.py           # Helper functions (70 lines)
```

### LLM Provider Support

The system auto-detects available providers:

1. **Claude** (via Claude Code CLI) - Default when `ANTHROPIC_API_KEY` is set
   - Supports tool use (Read file access with directory whitelisting)
   - Subprocess-based execution via `claude` CLI

2. **OpenAI** (direct API) - Used when `OPENAI_API_KEY` is set
   - Direct API calls via OpenAI Python SDK
   - Configurable model via `OPENAI_MODEL` environment variable

3. **Custom** - Fallback for any LLM CLI
   - Set `RAG_LLM_COMMAND` to your custom command
   - Prompt passed via stdin, response expected on stdout

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings | Required |
| `OPENAI_EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `ANTHROPIC_API_KEY` | Claude API key | Optional (enables Claude) |
| `OPENAI_MODEL` | OpenAI chat model | `gpt-5` |
| `RAG_LLM_COMMAND` | Custom LLM command | Optional |
| `RAG_DEPTH` | Current recursion depth | `0` |
| `RAG_CITATIONS` | Include explicit citations | `0` |
| `RAG_DEBUG` | Enable debug mode | `0` |

### Customization

Edit `rag_system/config.py` to customize:

- **Paths**: Session directories, file paths to index
- **Search weights**: Vector vs keyword balance (default 70/30)
- **Result limits**: Number of files/sessions/chats returned
- **Embedding settings**: Model, batch size, rate limits
- **Context size**: Maximum RAG context characters

---

## Advanced Usage

### Embedding Cost Optimization

```bash
# View token usage and costs
./rag_system.py stats

# Lazy generation means:
# - Embeddings only computed once per document
# - Only new/changed documents trigger re-embedding
# - Cost tracking shows cumulative OpenAI spend
```

### Custom Document Sources

Edit `rag_system/config.py`:

```python
DEFAULT_PATHS: List[Path] = [
    HOME / "my_knowledge_base",
    HOME / "documentation",
    HOME / "notes",
]
```

### FAISS Acceleration

Install FAISS for 10-100x faster vector search:

```bash
# CPU version
pip3 install faiss-cpu

# GPU version (requires CUDA)
pip3 install faiss-gpu
```

System automatically uses FAISS if available, falls back to NumPy otherwise.

---

## Performance

- **Indexing**: ~50 documents/second
- **Search**: <100ms for 4K documents (with FAISS)
- **Embedding**: 3,000 requests/min (OpenAI Tier 3)
- **Storage**: ~12KB per document (float32 embeddings)

### Benchmarks

- **3,939 documents indexed**: 34MB database
- **Hybrid search**: 70% vector + 30% keyword
- **Cost**: $0.00002 per 1K tokens (text-embedding-3-small)

---

## Development

### Running Tests

```bash
# Basic functionality
python3 test_rag_features.py

# Chunking strategy validation
python3 test_chunking.py
```

### Type Checking

```bash
mypy rag_system/
```

### Code Style

- Full type hints throughout
- Modular architecture with single responsibility
- Comprehensive docstrings

---

## Troubleshooting

### No LLM provider configured
```bash
# Set at least one API key
export ANTHROPIC_API_KEY="..."  # or
export OPENAI_API_KEY="..."
```

### Database locked errors
```bash
# Ensure no other instances running
pkill -f rag_system
rm ~/.rag_index.lock  # if exists
```

### High embedding costs
```bash
# Use smaller model
export OPENAI_EMBEDDING_MODEL="text-embedding-3-small"

# Reduce indexed paths in config.py
# Enable ALLOWED_EXTS filtering
```

---

## License

MIT License - See LICENSE file for details

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure type hints are complete
5. Submit a pull request

---

## Citation

```bibtex
@software{rag_system_2025,
  title = {RAG System v5.0: Production-Grade Retrieval-Augmented Generation},
  year = {2025},
  url = {https://github.com/Radar105/rag-system}
}
```
