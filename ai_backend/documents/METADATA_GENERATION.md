## 6. LLM-Assisted Metadata Generation System

### Overview

The system implements an **LLM-assisted ingestion pipeline** that enriches documents with semantic metadata before storage in the vector database. This improves retrieval quality, enables better filtering, and enhances explainability while maintaining CPU efficiency.

### Problem Statement

In a RAG system, relying only on embeddings causes:
- Mixed retrieval results
- Poor filtering capabilities
- Low explainability

Using a large LLM for everything is inefficient on CPU-only systems. The solution is to use LLM **only at ingestion time** to generate high-quality metadata.

### Architecture

#### Metadata Separation

**Strict Metadata (System-Controlled)**:
- `document_type`, `department`, `sensitivity`
- `source`, `domain`, `published_year`
- `allowed_roles`, `tags`, `effective_date`
- **Purpose**: Hard filtering and RBAC enforcement
- **Never modified by LLM**

**Soft Metadata (LLM-Generated)**:
- `summary` (2-3 sentences)
- `keywords` (5-10 relevant terms)
- `themes` (3-5 main topics)
- `entities` (people, organizations, locations)
- **Purpose**: Ranking, context, and semantic search
- **Generated once per document**

#### Ingestion Flow

```
1. Scan data/company/v* directories
   ↓
2. For each document:
   - Extract text using doc_parser
   - Load existing .meta.json (strict metadata)
   - Generate LLM metadata (summary, keywords, themes, entities)
   - Merge strict + soft metadata
   ↓
3. Save to cleaned/company/v*:
   - Original document text
   - Enriched metadata (.enriched.json)
   ↓
4. Generate cleanup report with statistics
```

### Components

#### Metadata Models (`metadata_models.py`)

```python
class StrictMetadata(BaseModel):
    """System-controlled metadata fields"""
    document_type: str
    department: str
    sensitivity: str
    source: str
    domain: Optional[str]
    published_year: Optional[int]
    allowed_roles: Optional[List[str]]
    tags: Optional[List[str]]

class SoftMetadata(BaseModel):
    """LLM-generated metadata fields"""
    summary: str
    keywords: List[str]
    themes: List[str]
    entities: Optional[Dict[str, List[str]]]
    generated_at: str
    llm_model: Optional[str]
    confidence: Optional[float]

class EnrichedMetadata(BaseModel):
    """Combined metadata"""
    strict: StrictMetadata
    soft: SoftMetadata
    enriched_at: str
    processing_time_ms: Optional[float]
```

#### Metadata Generator (`metadata_generator.py`)

**Key Features**:
- **Token-optimized prompts**: Structured format for consistent extraction
- **Text truncation**: Handles long documents (70% start, 30% end)
- **Robust parsing**: Extracts structured data from LLM response
- **Fallback handling**: Graceful degradation on LLM failures
- **Low temperature (0.1)**: Ensures consistent metadata extraction

**LLM Prompt Structure**:
```
Extract metadata from the following document.

Document:
[truncated text]

Extract the following information:
1. SUMMARY: 2-3 sentence summary
2. KEYWORDS: 5-10 relevant keywords (comma-separated)
3. THEMES: 3-5 main themes (comma-separated)
4. ENTITIES:
   - PEOPLE: Names mentioned
   - ORGANIZATIONS: Companies, departments
   - LOCATIONS: Places mentioned

Format:
SUMMARY: [summary]
KEYWORDS: [keywords]
THEMES: [themes]
PEOPLE: [people]
ORGANIZATIONS: [orgs]
LOCATIONS: [locations]
```

#### Cleanup Service (`cleanup_service.py`)

**Orchestrates the enrichment pipeline**:
- Scans `data/company/v*` directories
- Processes each document with LLM metadata generation
- Saves enriched versions to `cleaned/company/v*`
- Generates detailed cleanup reports
- **Idempotent**: Safe to re-run without side effects
- **Error handling**: Continues processing on individual failures

### File Structure

**Input (Original)**:
```
data/company/
├── v1/
│   ├── CEO_memo_strategic_vision.md
│   └── CEO_memo_strategic_vision.meta.json
└── v2/
    ├── AI_system_architecture.md
    └── AI_system_architecture.meta.json
```

**Output (Enriched)**:
```
cleaned/company/
├── v1/
│   ├── CEO_memo_strategic_vision.md
│   └── CEO_memo_strategic_vision.enriched.json
└── v2/
    ├── AI_system_architecture.md
    └── AI_system_architecture.enriched.json
```

### Design Principles

1. **Separation of Concerns**: LLM metadata never overrides system metadata
2. **Efficiency**: Single LLM call per document, token-optimized prompts
3. **Robustness**: Fallback metadata on LLM failures, graceful degradation
4. **Observability**: Detailed reporting, per-document status tracking
5. **Safety**: Original files never modified (read-only operation)

### Performance

**Expected Performance** (CPU-only, local LLM):
- **Per document**: 2-5 seconds
- **18 documents**: ~36-90 seconds total
- **Token usage**: ~300-500 tokens per document
- **Memory**: Minimal (streaming processing)

### Integration with Document Manager

The `document_manager.py` has been updated to:
- Skip `.enriched.json` files during seeding (they're metadata companions)
- Support loading enriched metadata alongside original `.meta.json`
- Preserve backward compatibility with existing documents

### Testing

**Unit Tests** (`test_metadata_generator.py`):
- ✅ 8/8 tests passing
- Covers success cases, failures, truncation, parsing

**Integration Tests** (`test_cleanup_service.py`):
- ✅ 9/9 tests passing
- End-to-end pipeline validation
- Error handling and edge cases

---
