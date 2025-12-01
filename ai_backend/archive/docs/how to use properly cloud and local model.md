# Secure RAG — end-to-end guide (small org → startup → large enterprise)

Great — I’ll walk you through a complete, practical path from small companies to large enterprises, and then give concrete, copy-pasteable code snippets and blueprints for each security control you asked about.

I’ll cover:

* Short strategic guidance for small companies, startups, and large companies
* How big companies handle secure RAG (architecture & controls)
* Practical solutions: sanitization, classification (safe/unsafe), two-tier LLMs, “no-leak” wrappers, redaction pipelines, and sensitive-info detection
* Code examples (Python) you can drop into your stack
* A checklist & recommended tech stack

---

## Quick strategic summary (one-line)

* **Small orgs / early startups:** keep everything **local** or use enterprise cloud contracts; start simple (local embeddings + local LLM + minimal redaction).
* **Scaling startups:** add metadata policies, retrieval filters, summarization & redaction, and two-tier LLM architecture (local + cloud fallback with strict filtering).
* **Large enterprises:** strict data classification, DLP, hardened enclave/private cloud LLMs, audit logs, contracts (no retention), encryption, legal review, and fine-grained access control.

---

# 1) For small companies (practical, fast, low-cost)

**Goal:** minimize leakage risk while moving fast.

**Recommended baseline:**

* Keep **embeddings + vector DB local** (Chroma, SQLite, FAISS).
* Use a **local LLM** (TinyLlama, Mistral, LlamaCpp, GPT4All) for confidential documents.
* If you need cloud LLMs, **never** send raw confidential text — first run **sanitization + classification** and only then send allowed non-sensitive context.
* Use simple redaction rules (PII regex), and store minimal logs (or sanitize logs).

**Why:** low infra overhead, full control, fast iteration.

---

# 2) For startups scaling to product / customers

**Goal:** keep data safe while enabling higher quality (maybe cloud models).

**Recommended additions:**

* Implement **per-tenant encryption** at rest and in transit.
* Add **metadata tagging** for docs (sensitivity, owner, retention).
* Create **retrieval policies**: e.g., “never return chunks tagged `sensitive: true` to cloud LLMs”.
* Add **summarization + compression** of past conversations (convert raw PII into high-level facts) and store summaries locally.
* Implement **two-tier LLM**: first try local LLM; for harder queries use cloud LLM but only after policy checks and explicit consent.

---

# 3) For large enterprise / regulated data

**Goal:** certifiable (legal + compliance) protected data handling.

**Controls to add:**

* Data classification & DLP (Data Loss Prevention) integrated into ingestion pipeline.
* Use **private cloud LLMs** (e.g., Azure OpenAI with no retention / enterprise contract) or fully self-hosted models in secure VPCs.
* Strict IAM, network segmentation, encryption keys (KMS), HSM if needed.
* Auditing, monitoring, alerting, and regular independent security reviews.
* Legal/Privacy team approves model usage & SLAs.

---

# How big companies handle secure RAG (architecture & controls)

1. **Ingestion pipeline**

   * Document ingestion → classifier → tag (sensitivity, PII flags, owner) → store (encrypted) → chunk & embed locally.
2. **Vector DB & access control**

   * Vector DB with per-tenant/ per-document ACLs; filter retrievals by ACL.
3. **Retrieval policy**

   * Retrieval returns candidate chunks with metadata (sensitivity/score). Policy engine decides what can be sent to LLM.
4. **Sanitization/redaction**

   * Apply redaction rules + named-entity filters (remove SSNs, emails, secrets). Store both raw (encrypted) and redacted versions.
5. **LLM execution**

   * Prefer local/private LLMs for confidential workloads. If cloud LLM used, pass only redacted context or ask cloud model to run a transformation (but be cautious).
6. **Audit & monitoring**

   * Log requests, redactions, decisions, and who authorized cloud usage.
7. **Legal & contracts**

   * Enterprise agreements (no retention, data residency clauses).

---

# How to sanitize data before sending to an LLM

Sanitization removes or masks sensitive tokens from the prompt.

### Simple pipeline

1. **Detect** PII/sensitive elements (regex or NER)
2. **Mask/replace** with placeholders (`[EMAIL_REDACTED]`, `[SSN_REDACTED]`)
3. **Optionally summarize** sensitive fragments to a high-level statement (`"User identity information removed"`)
4. **Log the transformation** (audit trail)

### Python example (regex + phonenumbers)

```python
import re
import phonenumbers

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")  # US SSN example

def redact_simple(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL_REDACTED]", text)
    text = SSN_RE.sub("[SSN_REDACTED]", text)
    # phone numbers
    def repl_phone(match):
        try:
            pn = phonenumbers.parse(match.group(0), None)
            return "[PHONE_REDACTED]"
        except:
            return "[PHONE_REDACTED]"
    text = re.sub(r"\+?\d[\d\-\s]{7,}\d", repl_phone, text)
    return text
```

**Note:** regex is necessary but not sufficient. Use NER (spaCy) or Presidio for more coverage.

---

# How to classify documents as safe/unsafe

You want an automated gate: *is this doc allowed to go to a cloud LLM?*

### Two approaches:

* **Rule-based (fast):** look for keywords, file metadata (e.g., `confidential`, `PII`, `SSN`, `invoice`) and patterns.
* **ML-based (better):** embed the doc and a classifier model that outputs `safe/unsafe/needs_review`.

### Example rule-based check

```python
SENSITIVE_KEYWORDS = ["confidential", "ssn", "social security", "salary", "patient", "medical"]

def is_doc_sensitive(text:str) -> bool:
    lower = text.lower()
    for kw in SENSITIVE_KEYWORDS:
        if kw in lower:
            return True
    # regex checks:
    if EMAIL_RE.search(text) or SSN_RE.search(text):
        return True
    return False
```

### Example ML approach (quick)

* Label a small dataset of docs `safe/unsafe` and train a classifier (LogisticRegression on TF-IDF or small transformer). Use it to auto-classify and set `needs_review` threshold.

---

# How to use two-tier LLMs (local + cloud fallback)

**Pattern:** attempt local inference first; if unsatisfactory or user requests higher quality, route to cloud with strict filtering.

### Flow:

1. Retrieve relevant context.
2. Apply sanitization + classify.
3. If **safe_for_cloud** and user consents → send redacted context + question to cloud LLM.
4. Otherwise use local LLM.
5. Log the decision & evidence.

### Pseudocode

```python
def answer_query(query, user_id):
    retrieved = retrieve_for_user(query, user_id)
    context = "\n".join(retrieved)
    if is_doc_sensitive(context):
        # use local LLM
        return local_llm_answer(query, context)
    else:
        # optionally sanitize
        redacted = redact_sensitive(context)
        if should_use_cloud(query):
            return cloud_llm_answer(query, redacted)
        else:
            return local_llm_answer(query, redacted)
```

**Notes:** always audit cloud calls and keep redaction logs.

---

# How to implement “no-leak” wrappers (middleware)

A no-leak wrapper wraps every call to cloud LLM and ensures nothing sensitive is sent.

### Responsibilities of the wrapper:

* Ensure **policy checks** pass (classification/sensitivity).
* Apply **redaction** and **consent checks**.
* Strip out confidential metadata and prevent accidental logging of raw prompts.
* Add **watermarking or tagging** (e.g., include a unique request id).
* Route to local LLM if policy denies cloud usage.
* Enforce rate limits & encryption.

### Example wrapper (Python)

```python
class NoLeakWrapper:
    def __init__(self, cloud_client, policy_engine):
        self.cloud_client = cloud_client
        self.policy_engine = policy_engine

    def safe_call(self, prompt, user_id, metadata=None):
        # 1. classify
        if self.policy_engine.is_sensitive(prompt, metadata):
            return {"error": "sensitive_content_blocked", "fallback":"local"}
        # 2. sanitize
        prompt_sanitized = redact_simple(prompt)
        # 3. call cloud
        res = self.cloud_client.create(prompt=prompt_sanitized)
        # 4. audit
        audit_log(user_id, prompt_hash=hash(prompt_sanitized), model=self.cloud_client.name)
        return res
```

---

# How to add redaction pipelines

Redaction pipeline = detectors → transformers → storage

### Steps:

1. **Detect** PII (NER, regex, external tools).
2. **Transform**: redact or summarize detected spans.
3. **Store**: keep raw securely (encrypted) and store redacted copy for LLM usage.
4. **Audit**: record what was redacted & why.

### Tools:

* Microsoft Presidio (PII detection & masking)
* spaCy + custom NER
* AWS Comprehend / Google DLP (if using cloud with enterprise agreements)
* Regex & `phonenumbers` for phone detection

### Example using Presidio (conceptual)

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def presidio_redact(text):
    results = analyzer.analyze(text=text, entities=["PHONE_NUMBER","EMAIL_ADDRESS","US_SSN"], language="en")
    redacted = anonymizer.anonymize(text=text, analyzer_results=results)
    return redacted.text
```

---

# How to detect and block sensitive information in prompts

Combine pattern matching + NER + ML classifier.

### 1. Fast checks (deny list patterns)

Use regex for SSNs, credit card numbers, API keys, and credentials:

```python
API_KEY_RE = re.compile(r"(sk_live|AKIA)[A-Za-z0-9\-_]{16,}")
CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
```

### 2. NER for names/locations/emails using spaCy

```python
import spacy
nlp = spacy.load("en_core_web_sm")

def has_personal_info(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ("PERSON","GPE","LOC","ORG","MONEY","DATE"):
            # make judgement or return ent.text
            pass
```

### 3. ML scoring

Train a binary classifier on prompts labeled `sensitive` / `not_sensitive` and threshold.

### 4. Enforce in wrapper

If detection positive → block or redact before cloud call.

---

# Practical code bundle — small, usable components

Below are simple functions you can integrate:

### a) Redact & detect (combined)

```python
import re
import phonenumbers
import spacy

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
API_KEY_RE = re.compile(r"(sk_live|AKIA)[A-Za-z0-9\-_]{16,}")

nlp = spacy.load("en_core_web_sm")

def detect_sensitive(text: str) -> dict:
    hits = []
    if EMAIL_RE.search(text):
        hits.append("email")
    if SSN_RE.search(text):
        hits.append("ssn")
    if API_KEY_RE.search(text):
        hits.append("api_key")
    # phone
    for match in re.finditer(r"\+?\d[\d\-\s]{7,}\d", text):
        try:
            phonenumbers.parse(match.group(0))
            hits.append("phone")
        except:
            pass
    # NER
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ("PERSON","GPE","ORG", "MONEY"):
            hits.append(ent.label_)
    return {"sensitive_types": list(set(hits)), "count": len(hits)}

def redact(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL_REDACTED]", text)
    text = SSN_RE.sub("[SSN_REDACTED]", text)
    text = API_KEY_RE.sub("[API_KEY_REDACTED]", text)
    # phone redaction naive
    text = re.sub(r"\+?\d[\d\-\s]{7,}\d","[PHONE_REDACTED]", text)
    # optionally apply Presidio or custom replacements for NER
    return text
```

### b) Policy engine (pluggable)

```python
class PolicyEngine:
    def __init__(self, strict=True):
        self.strict = strict

    def is_sensitive(self, text, metadata=None):
        res = detect_sensitive(text)
        if res["count"]>0:
            return True
        if metadata and metadata.get("sensitivity")== "high":
            return True
        return False
```

---

# Logging, auditing & retention

* **Log every decision** (what was redacted, why, who authorized cloud use) with hashed prompts (avoid logging raw prompt).
* **Retention**: keep audit logs for required duration per compliance.
* **Encryption**: store raw documents encrypted (KMS/HSM).
* **Access controls**: role-based access for any operation that can de-redact.

---

# Operational best practices & monitoring

* Use canaries and redaction unit tests to ensure no new patterns leak.
* Monitor for repeated policy violations → auto-block or alert security.
* Use differential logging: success summaries vs full redaction traces stored in secure vault.

---

# Recommended open-source tools

* **Presidio** (PII detection & anonymization)
* **spaCy** (NER)
* **phonenumbers** (phone parsing)
* **sentence-transformers** (embeddings)
* **Chroma/FAISS** (vector DB)
* **LangChain** (memory/chain helpers)
* **Sentry/ELK/Prometheus** (monitoring & auditing)
* **Vault/KMS** (secret management & encryption keys)

---

# Minimal checklist to ship a secure small/medium RAG product

1. Local embeddings + local vector DB.
2. Tag & classify docs on ingestion.
3. Implement `is_doc_sensitive()` and `redact()` functions.
4. Add `NoLeakWrapper` for any cloud LLM use (block until policy passes).
5. Store raw docs encrypted, redacted docs accessible for LLM.
6. Add audit logs for every LLM invocation and redaction decision.
7. Periodic summaries of old chats, store summaries separately.
8. Add monitoring & alerts for policy violations.

---

## Final practical tip

If you handle regulated data (health, finance, personal data), **consult legal/compliance** before sending anything to third-party cloud LLMs. Prefer self-hosted or enterprise agreements.

---