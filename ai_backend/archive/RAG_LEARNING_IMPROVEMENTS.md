# 🧠 Core RAG Learning Enhancements

## 1. Advanced Chunking Strategies

```python
# Add different chunking methods to learn their impact
def chunk_by_sentences(text: str, max_sentences: int = 3) -> List[str]:
    sentences = text.split('. ')
    return ['. '.join(sentences[i:i+max_sentences]) for i in range(0, len(sentences), max_sentences)]

def chunk_by_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in text.split('\n\n') if p.strip()]

def chunk_with_overlap_sliding_window(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    # Learn how overlap affects retrieval quality
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
```

## 2. Multiple Embedding Models Comparison

```python
# Compare different embedding models
EMBEDDING_MODELS = {
    "minilm": "all-MiniLM-L6-v2",
    "mpnet": "all-mpnet-base-v2", 
    "distilbert": "all-distilbert-base-v1"
}

async def embed_with_multiple_models(text: str) -> Dict[str, List[float]]:
    # Learn how different models affect retrieval
    embeddings = {}
    for model_name, model_path in EMBEDDING_MODELS.items():
        model = SentenceTransformer(model_path)
        embeddings[model_name] = model.encode(text).tolist()
    return embeddings
```

## 3. Retrieval Strategy Experiments

```python
# Add different retrieval methods
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

## 4. Reranking and Relevance Scoring

```python
# Learn about reranking retrieved documents
def calculate_relevance_score(query: str, document: str) -> float:
    # Simple TF-IDF or cosine similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([query, document])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

def rerank_documents(query: str, documents: List[str]) -> List[str]:
    # Rerank based on relevance scores
    scored_docs = [(doc, calculate_relevance_score(query, doc)) for doc in documents]
    return [doc for doc, score in sorted(scored_docs, key=lambda x: x[1], reverse=True)]
```

## 5. Query Enhancement Techniques

```python
# Learn query expansion and refinement
def expand_query_with_synonyms(query: str) -> str:
    # Add synonyms to improve retrieval
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
    # Classify query type (factual, procedural, conceptual)
    query_types = ["factual", "procedural", "conceptual", "comparative"]
    # Use simple keyword matching or ML model
    if any(word in query.lower() for word in ["how", "steps", "process"]):
        return "procedural"
    elif any(word in query.lower() for word in ["what", "who", "when", "where"]):
        return "factual"
    return "conceptual"
    
def generate_sub_queries(complex_query: str) -> List[str]:
    # Break complex queries into simpler ones
    return [q.strip() for q in complex_query.split("and") if q.strip()]
```

## 6. Context Window Management

```python
# Learn about context optimization
def optimize_context_window(chunks: List[str], max_tokens: int = 2000) -> List[str]:
    # Select most relevant chunks within token limit
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
    # Select context based on query complexity
    if is_simple_query(query):
        return chunks[:2]  # Less context for simple queries
    else:
        return chunks[:5]  # More context for complex queries
```

## 7. RAG Evaluation Metrics

```python
# Learn to measure RAG performance
class RAGEvaluator:
    def calculate_retrieval_precision(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        # Precision = relevant retrieved / total retrieved
        relevant_retrieved = len(set(retrieved_docs) & set(relevant_docs))
        return relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0
        
    def calculate_retrieval_recall(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        # Recall = relevant retrieved / total relevant
        relevant_retrieved = len(set(retrieved_docs) & set(relevant_docs))
        return relevant_retrieved / len(relevant_docs) if relevant_docs else 0
        
    def calculate_answer_relevance(self, question: str, answer: str) -> float:
        # How relevant is the generated answer to the question
        return calculate_relevance_score(question, answer)
        
    def calculate_faithfulness(self, answer: str, context: str) -> float:
        # How faithful is the answer to the provided context
        return calculate_relevance_score(context, answer)
```

## 8. Different RAG Patterns

```python
# Learn various RAG architectures
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

## 9. Document Preprocessing Experiments

```python
# Learn how preprocessing affects RAG
def preprocess_document_basic(text: str) -> str:
    # Basic cleaning
    return text.strip().lower()

def preprocess_document_advanced(text: str) -> str:
    # Advanced: Remove headers, footers, normalize whitespace
    import re
    text = re.sub(r'\n+', '\n', text)  # Normalize newlines
    text = re.sub(r' +', ' ', text)    # Normalize spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove special chars
    return text

def extract_key_information(text: str) -> Dict[str, str]:
    # Extract entities, dates, numbers for better retrieval
    return {
        "entities": extract_entities(text),
        "dates": extract_dates(text),
        "numbers": extract_numbers(text)
    }
```

## 10. Vector Database Experiments

```python
# Learn about vector operations
def similarity_search_with_filters(query_vector: List[float], filters: Dict):
    # Learn how metadata filtering affects retrieval
    results = chroma_collection.query(
        query_embeddings=[query_vector],
        where=filters,
        n_results=10
    )
    return results
    
def vector_clustering_analysis():
    # Analyze how documents cluster in vector space
    from sklearn.cluster import KMeans
    embeddings = get_all_embeddings()
    kmeans = KMeans(n_clusters=5)
    clusters = kmeans.fit_predict(embeddings)
    return clusters
    
def dimension_reduction_experiment():
    # Learn about dimensionality reduction effects
    from sklearn.decomposition import PCA
    embeddings = get_all_embeddings()
    pca = PCA(n_components=50)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings
```

## 11. Prompt Engineering for RAG

```python
# Learn different prompting strategies
PROMPT_TEMPLATES = {
    "basic": "Context: {context}\nQuestion: {question}\nAnswer:",
    
    "chain_of_thought": """Context: {context}
Question: {question}
Let me think step by step:
1. First, I'll identify the key information in the context
2. Then, I'll relate it to the question
3. Finally, I'll provide a comprehensive answer
Answer:""",
    
    "few_shot": """Here are some examples:
Context: [example context 1]
Question: [example question 1]  
Answer: [example answer 1]

Context: {context}
Question: {question}
Answer:""",
    
    "role_based": """You are a {role} expert. Based on the following context, answer the question from a {role} perspective.
Context: {context}
Question: {question}
Answer:"""
}
```

## 12. RAG Pipeline Visualization

```python
# Learn to visualize RAG process
def log_rag_pipeline_steps(query: str):
    steps = []
    
    # Step 1: Query processing
    processed_query = preprocess_query(query)
    steps.append(f"Query processed: {processed_query}")
    
    # Step 2: Retrieval
    docs = retrieve_documents(processed_query, 5)
    steps.append(f"Retrieved {len(docs)} documents")
    
    # Step 3: Reranking
    reranked_docs = rerank_documents(processed_query, docs)
    steps.append(f"Reranked documents")
    
    # Step 4: Context building
    context = build_context(reranked_docs)
    steps.append(f"Built context: {len(context)} chars")
    
    # Step 5: Generation
    answer = generate_response(processed_query, context)
    steps.append(f"Generated answer: {len(answer)} chars")
    
    return steps
```

---

# 🎯 Learning-Focused Implementation Priority

## Week 1-2: Chunking & Retrieval
- Implement different chunking strategies (#1)
- Add multiple retrieval methods (#3)
- Build RAG evaluation metrics (#7)

## Week 3-4: Advanced Techniques
- Add query enhancement (#5)
- Implement reranking (#4)
- Experiment with different RAG patterns (#8)

## Week 5-6: Optimization & Analysis
- Context window management (#6)
- Vector database experiments (#10)
- Pipeline visualization (#12)

## Week 7-8: Production Ready
- Document preprocessing (#9)
- Prompt engineering (#11)
- Multiple embedding models (#2)ry enhancement (#5)
5. Implement reranking (#4)
6. Experiment with RAG patterns (#8)

Week 5-6: Optimization & Analysis
7. Context window optimization (#6)
8. Vector analysis experiments (#10)
9. Prompt engineering (#11)