# 🧠 RAG Learning & Improvements Roadmap

## 🎯 Core RAG Learning Enhancements

### 1. Advanced Chunking Strategies

#### Different Chunking Methods
```python
def chunk_by_sentences(text: str, max_sentences: int = 3) -> List[str]:
    sentences = text.split('. ')
    return ['. '.join(sentences[i:i+max_sentences]) for i in range(0, len(sentences), max_sentences)]

def chunk_by_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in text.split('\n\n') if p.strip()]

def chunk_with_overlap_sliding_window(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
```

### 2. Multiple Embedding Models Comparison

```python
EMBEDDING_MODELS = {
    "minilm": "all-MiniLM-L6-v2",
    "mpnet": "all-mpnet-base-v2", 
    "distilbert": "all-distilbert-base-v1"
}

async def embed_with_multiple_models(text: str) -> Dict[str, List[float]]:
    embeddings = {}
    for model_name, model_path in EMBEDDING_MODELS.items():
        model = SentenceTransformer(model_path)
        embeddings[model_name] = model.encode(text).tolist()
    return embeddings
```

### 3. Retrieval Strategy Experiments

```python
async def hybrid_search(query: str, n_results: int = 5):
    # Combine semantic + keyword search
    semantic_results = await semantic_search(query, n_results//2)
    keyword_results = await keyword_search(query, n_results//2)
    return merge_and_rank_results(semantic_results, keyword_results)

async def multi_query_retrieval(query: str):
    # Generate multiple query variations
    query_variations = generate_query_variations(query)
    all_results = []
    for q in query_variations:
        results = await retrieve_documents(q, 3)
        all_results.extend(results)
    return deduplicate_and_rank(all_results)
```

### 4. Reranking and Relevance Scoring

```python
def calculate_relevance_score(query: str, document: str) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([query, document])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

def rerank_documents(query: str, documents: List[str]) -> List[str]:
    scored_docs = [(doc, calculate_relevance_score(query, doc)) for doc in documents]
    return [doc for doc, score in sorted(scored_docs, key=lambda x: x[1], reverse=True)]
```

### 5. Query Enhancement Techniques

```python
def expand_query_with_synonyms(query: str) -> str:
    import nltk
    from nltk.corpus import wordnet
    
    words = query.split()
    expanded_words = []
    for word in words:
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        expanded_words.extend(list(synonyms)[:2])  # Add top 2 synonyms
    return ' '.join(words + expanded_words)
    
def query_classification(query: str) -> str:
    if any(word in query.lower() for word in ["how", "steps", "process"]):
        return "procedural"
    elif any(word in query.lower() for word in ["what", "who", "when", "where"]):
        return "factual"
    return "conceptual"
    
def generate_sub_queries(complex_query: str) -> List[str]:
    return [q.strip() for q in complex_query.split("and") if q.strip()]
```

## 🚀 High-Impact Optimization Strategies

### Core RAG Enhancements

#### A. Relevance Re-ranking (Cross-encoder / MiniLM reranker)
After Chroma returns top_k chunks, pass them through a **local reranker** to boost accuracy.

**Works offline using:**
- `cross-encoder/ms-marco-MiniLM-L6-v2`
- `bge-reranker-base` (if CPU ok)

**Impact:** Much higher answer accuracy.

#### B. Auto-summarization of Large Chunks
Before storing into Chroma:
- Summaries can be generated locally using your LLM
- Also store summary embeddings for fallback retrieval

#### C. Hybrid Retrieval (keyword + vector)
Implement:
- TF-IDF + vector
- BM25 + vector

Better for legal/technical text.

### Security + RBAC Enhancements

#### E. Fine-grained Access Rules Based on Attributes
Add support for:
- `classification_level`
- `region_restricted` (ex: EU-only)
- `expire_after_date`
- `requires_manager_access=true`

#### F. Redaction Before LLM
If chunk is partially confidential:
- Replace sensitive fields with [[REDACTED]]
- Keep rest of chunk visible

Great for HR & Legal data.

#### G. Audit Logging (enterprise-style)
Log:
- Who accessed what document
- Timestamp
- Purpose
- Whether content was filtered or redacted
- Fallbacks executed

### Support Chat System Enhancements

#### H. Persistent Long-term Memory (per session / per user)
Keep:
- Preferences
- Past questions
- Previous task threads
- Known issues from support history

#### I. Assistant Personality Modes
Based on `profile.role`:
- HR mode
- IT support mode
- Finance mode
- Legal mode
- Manager mode

Each mode loads different prefixed behavior.

#### J. Conversation Continuation After Restart
Right now your support chat keeps short-term history.

Add:
- `/session/resume`
- Restore last conversation state
- Allow multi-day debugging & chats

#### K. User Sentiment / Tone Detection (local classifier)
Simple offline BERT classifier that detects:
- Angry user
- Confused user
- Happy user

LLM adjusts tone accordingly.

### LLM Layer Upgrades

#### L. Model Auto-selection / Routing
Use smaller model for simple tasks:
- Summarization
- Classification
- Tagging
- Intent detection

Use Mistral only for heavy reasoning.

#### M. Offline Fine-tuning / LoRA
Train your model locally on:
- Your policies
- Your workflows
- HR/IT knowledge base

Even small LoRA improves performance a lot.

#### N. Local Embedding Model Switch
Support:
- bge-small-en
- nomic-embed-text
- gte-small

Some are faster/better on CPU.

#### O. Prompt Caching
Store:
- Previous LLM outputs
- Recent embeddings
- Common prefix structures

Reduces latency significantly.

## 📊 RAG Evaluation & Testing Framework

### RAG Evaluation Metrics

```python
class RAGEvaluator:
    def calculate_retrieval_precision(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        relevant_retrieved = len(set(retrieved_docs) & set(relevant_docs))
        return relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0
        
    def calculate_retrieval_recall(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        relevant_retrieved = len(set(retrieved_docs) & set(relevant_docs))
        return relevant_retrieved / len(relevant_docs) if relevant_docs else 0
        
    def calculate_answer_relevance(self, question: str, answer: str) -> float:
        return calculate_relevance_score(question, answer)
        
    def calculate_faithfulness(self, answer: str, context: str) -> float:
        return calculate_relevance_score(context, answer)
```

### Top 8 Prioritized Evaluation Steps

1. **Create an evaluation/benchmark suite (high priority)**
   - Collect ~100–300 representative Q&A pairs across HR/Finance/IT/Legal
   - Include easy, ambiguous, and restricted queries (RBAC)
   - Add labeled "gold answer" or expected behavior
   - **Outcome:** Baseline metrics (precision@k, MRR, exact-match, RBAC compliance)

2. **Tune chunking & retrieval parameters**
   - Experiment with chunk sizes (256, 512, 1024 chars) and overlap (32,64,128)
   - Measure retrieval recall and prompt token budget efficiency
   - **Outcome:** Optimal chunk settings that maximize useful context

3. **Optimize token-budgeted chunk selection**
   - Formalize `select_chunks_by_token_budget` policy
   - Try greedy vs. scoring approaches
   - **Outcome:** Lower hallucination and fewer truncated contexts

4. **Evaluate & improve embeddings quality**
   - Run retrieval-only experiments comparing local MiniLM embeddings to alternatives
   - Test different distance metrics and normalization
   - **Outcome:** Higher retrieval precision at k, better prompts to LLM

5. **Model-router & task mapping validation**
   - Enable `ENABLE_DYNAMIC_MODEL_SELECTION=True` in dev run
   - Test routing for tasks (summarize → small, QA → tiny, reasoning → mistral)
   - **Outcome:** Rules/thresholds to keep Mistral reserved for heavy tasks

6. **RBAC stress tests + audit logs**
   - Create automated tests that query restricted docs with different API keys/roles
   - Measure false exposures and false denies
   - **Outcome:** Confident RBAC with count of policy violations

7. **Prompt engineering & persona prefixes**
   - A/B test different LLM prefixes (short vs. detailed persona, instruction templates)
   - Measure answer helpfulness and hallucination
   - **Outcome:** Stable prefix templates per support category and per role

8. **Factuality & hallucination mitigation strategy**
   - Add "source citing" prompts (ask LLM to include doc ids or quotes)
   - Enforce "I don't know" fallback
   - **Outcome:** Lower hallucination and traceable answers

## 🔬 Different RAG Patterns

### RAG Architecture Variations

```python
async def naive_rag(query: str):
    # Basic: Retrieve -> Generate
    docs = await retrieve_documents(query, 3)
    return await generate_response(query, docs)

async def iterative_rag(query: str, max_iterations: int = 3):
    # Iterative: Generate -> Check -> Retrieve more if needed
    for i in range(max_iterations):
        docs = await retrieve_documents(query, 3)
        answer = await generate_response(query, docs)
        if is_answer_complete(answer):
            return answer
        query = refine_query_based_on_answer(query, answer)
    return answer

async def self_rag(query: str):
    # Self-RAG: Model decides when to retrieve
    if should_retrieve(query):
        docs = await retrieve_documents(query, 3)
        return await generate_response(query, docs)
    else:
        return await generate_response(query, [])
```

## 🔧 Context Window Management

```python
def optimize_context_window(chunks: List[str], max_tokens: int = 2000) -> List[str]:
    token_count = 0
    selected_chunks = []
    
    for chunk in chunks:
        chunk_tokens = estimate_tokens_from_text(chunk)
        if token_count + chunk_tokens <= max_tokens:
            selected_chunks.append(chunk)
            token_count += chunk_tokens
        else:
            break
    return selected_chunks

def dynamic_context_selection(query: str, chunks: List[str]) -> List[str]:
    if is_simple_query(query):
        return chunks[:2]  # Less context for simple queries
    else:
        return chunks[:5]  # More context for complex queries
```

## 📈 Concrete Evaluation Metrics to Track

### Retrieval Quality
- **Precision@k, Recall@k, MRR**
- **End-to-end:** Exact match / BLEU / ROUGE for short answers
- **Factuality:** Hallucination rate (human or automatic checks)

### Security & Compliance
- **Safety/RBAC:** False exposure rate (% of filtered docs returned)
- **False denial rate:** % of allowed docs blocked

### Performance
- **Latency/Cost:** Average response time (s) and CPU usage per request
- **User quality:** Helpfulness/clarity rating (1–5) from manual checks

### Usage Analytics
- Most used documents
- Top errors and failure modes
- Most asked queries by department
- User engagement metrics

## 🧪 Small Experiments to Run Locally

### A. Ablation Studies
- **Embeddings only vs embeddings + re-ranking** — does re-ranking improve final answer quality?
- **Onboarding personalization impact** — compare personalized prefix vs stateless on same queries
- **Public-summary efficacy** — test whether `public_summary` is sufficient for policy questions

### B. Data & Tooling Housekeeping
- Build a small labeled **eval dataset** (CSV or JSONL) with fields: `query, expected_answer_or_policy, role, department, expected_access`
- Keep versioned copy of **ingestion metadata** for documents
- Enable verbose **decision logging** (which chunks were visible, which filtered with reasons)

### C. Resource Configuration Tips for 16GB RAM CPU
- Use smaller `n_ctx` (1024–2048) for Mistral in most tests
- Use `n_batch=1` and moderate `n_threads` (half cores) to avoid OOM
- Keep embedding ops batched but small when computing many vectors locally
- Use swap (8GB) if necessary for heavy seeding operations

## 🎯 Implementation Priority (Learning-Focused)

### Week 1-2: Foundation & Evaluation
1. Create evaluation/benchmark suite
2. Implement different chunking strategies
3. Add multiple retrieval methods
4. Build RAG evaluation metrics

### Week 3-4: Advanced Techniques
5. Add query enhancement techniques
6. Implement reranking algorithms
7. Experiment with different RAG patterns
8. RBAC stress testing

### Week 5-6: Optimization & Analysis
9. Context window management optimization
10. Vector database experiments
11. Pipeline visualization and logging
12. Prompt engineering experiments

### Week 7-8: Production Ready
13. Document preprocessing improvements
14. Multiple embedding models support
15. Performance optimization
16. Comprehensive testing suite

## 🔥 Low-Effort, High-Impact Wins (Do First)

1. **Create the 100–300 query eval suite** (highest impact)
2. **Run RBAC automated test cases** (exposure/deny checks)
3. **Tune chunk size** to one chosen setting and re-index sample docs
4. **Optimize prompt prefix and temperature** to reduce hallucinations

This roadmap provides a systematic approach to improving RAG performance through experimentation, evaluation, and iterative enhancement while maintaining focus on practical, measurable improvements.