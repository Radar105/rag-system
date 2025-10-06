# OpenAI Embeddings Best Practices for RAG v5.0

## Model: text-embedding-3-small

### Specifications
- **Dimensions**: 1536 (default)
- **Max input**: 8,191 tokens (~32,764 characters)
- **Cost**: $0.02 per million tokens ($0.00002 per 1K tokens)
- **Performance**: Optimized for latency and storage
- **Similarity**: Cosine similarity (normalized vectors, can use dot product)

## Current RAG v5.0 Implementation

### ✅ What We're Doing Right
1. **Lazy embedding generation** - Only embed when content changes
2. **Cost tracking** - Monitor token usage and costs
3. **FAISS acceleration** - Fast similarity search with graceful NumPy fallback
4. **Hybrid search** - 70% vector + 30% keyword (BM25)
5. **Metadata enrichment** - Dates, entities, keyphrases extracted
6. **Rate limiting** - 3000 RPM (within Tier 3 limits)

### ⚠️ Areas for Improvement (From OpenAI Docs)

## Chunking Strategy Recommendations

### Current Issues
- **Whole document embeddings** - Losing granular context
- **Preview limitation** - Session content truncated to 2K→10K chars (improved)
- **No overlap** - Context lost at chunk boundaries

### OpenAI Best Practices

#### 1. **Fixed-Size Chunking with Overlap**
```python
CHUNK_SIZE = 1000  # tokens (~4000 chars)
CHUNK_OVERLAP = 200  # 10-20% overlap recommended
```

**Why**: Preserves context across boundaries, prevents data loss at edges

#### 2. **Token-Based Chunking**
```python
# OpenAI models process tokens, not characters
# 1 token ≈ 4 characters for English text
MAX_TOKENS = 8191  # text-embedding-3-small limit
PRACTICAL_LIMIT = 7500  # Stay under limit with buffer
```

**Why**: Aligns with model's actual processing units

#### 3. **Structure-Aware Chunking**
For markdown/code:
- Split on headings, paragraphs, function definitions
- Preserve semantic boundaries
- Don't split mid-sentence or mid-block

**Why**: Semantically coherent chunks = better embeddings

#### 4. **Conversation-Specific Chunking**
For sessions:
- **One chunk per dialogue turn** (Q&A pair)
- Preserve user question + assistant response together
- Add conversation context (previous turn summary)

**Why**: Dialogue pairs have semantic unity

## Semantic Search Best Practices

### Similarity Measurement
```python
# OpenAI embeddings are normalized to length 1
# Use cosine similarity or dot product (faster)
similarity = np.dot(query_embedding, doc_embedding)
```

### Vector Storage
- Use vector database (we use SQLite + FAISS)
- Index for fast k-NN search
- Consider approximate nearest neighbor (ANN) for scale

### Hybrid Search Formula
```python
final_score = (vector_similarity * 0.7) + (keyword_match * 0.3)
```

**Research shows**: Hybrid consistently outperforms pure vector or pure keyword

## Dimension Reduction (Optional)

```python
# text-embedding-3-small supports dimension reduction
# Trade performance for cost/storage
dimensions = 512  # Instead of 1536
# Reduces storage by 67% with minimal accuracy loss
```

**When to use**:
- Storage constraints (millions of documents)
- Speed requirements (faster similarity computation)
- Acceptable slight accuracy drop (~5%)

## Rate Limits & Batching

### Current Implementation
- **Rate limit**: 3000 RPM (Tier 3)
- **Delay**: 20ms between requests = 50 req/sec = 3000/min ✅
- **Batch size**: 50 documents per embedding call
- **Retry logic**: Exponential backoff (3 retries)

### OpenAI Recommendations
- ✅ Batch multiple texts in single API call
- ✅ Implement exponential backoff on rate limits
- ✅ Monitor token usage to stay within limits

## Cost Optimization

### Current Costs
```python
# RAG v5.0 recent rebuild
751,855 tokens ≈ $0.0150 (for 1,126 documents)
Average: 668 tokens per document
Cost per document: ~$0.000013
```

### Optimization Strategies
1. **Chunk wisely** - Smaller chunks = more embeddings = higher cost
2. **Cache aggressively** - Don't re-embed unchanged content ✅
3. **Consider dimension reduction** - 67% storage savings
4. **Batch requests** - Reduce API overhead ✅

## Recommended Changes for RAG v5.0

### Phase 2: Semantic Chunking (Next)
- [ ] Implement 1000-token chunks with 200-token overlap
- [ ] Structure-aware splitting (preserve dialogue turns)
- [ ] Multiple embeddings per document for granular search
- [ ] Chunk metadata (position, parent document, context)

### Phase 3: Hybrid Search Enhancement
- [ ] Integrate BM25 for keyword matching
- [ ] Implement reranking layer after retrieval
- [ ] Dynamic weight adjustment based on query type

### Phase 4: Temporal Filtering
- [ ] Use metadata dates for precise time-range queries
- [ ] Boost recent documents in relevance scoring

### Phase 5: Conversation-Aware Embeddings
- [ ] Q&A pair chunking for sessions
- [ ] Contextual embeddings (include previous turn)
- [ ] Dialogue flow preservation

## References

- **OpenAI Embeddings Guide**: https://platform.openai.com/docs/guides/embeddings
- **text-embedding-3-small**: https://platform.openai.com/docs/models/text-embedding-3-small
- **Chunking Strategies**: Pinecone/LangChain/LlamaIndex best practices
- **Hybrid Search**: Weaviate, Qdrant research papers

---

**Last Updated**: September 29, 2025
**Current Implementation**: RAG v5.0 with Phase 1 (Metadata Enrichment) complete