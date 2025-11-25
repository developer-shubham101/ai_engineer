# Top 8 prioritized next steps (what → why → expected outcome)

1. **Create an evaluation/benchmark suite (high priority)**

   * **What:** Collect ~100–300 representative Q&A pairs across HR/Finance/IT/Legal etc. Include easy, ambiguous, and restricted queries (RBAC). Add a labeled “gold answer” or expected behavior (allowed / blocked / partial).
   * **Why:** Quantify retrieval quality, RBAC enforcement, hallucination and personalization.
   * **Outcome:** Baseline metrics (precision@k, MRR, exact-match / answer correctness, RBAC false-positive/negative).

2. **Tune chunking & retrieval parameters**

   * **What:** Experiment with chunk sizes (256, 512, 1024 chars) and overlap (32,64,128). Measure retrieval recall and prompt token budget efficiency.
   * **Why:** Chunking affects recall and prompt length; smaller chunks increase recall but raise token usage.
   * **Outcome:** Optimal chunk settings that maximize useful context while fitting Mistral’s n_ctx.

3. **Optimize token-budgeted chunk selection**

   * **What:** Formalize and test your `select_chunks_by_token_budget` policy (by token cost, semantic value, recency, owner). Try greedy vs. scoring approaches.
   * **Why:** Keeps prompts within context window and prioritizes highest-value chunks.
   * **Outcome:** Lower hallucination and fewer truncated contexts.

4. **Evaluate & improve embeddings quality**

   * **What:** Run retrieval-only experiments comparing the local MiniLM embeddings to alternatives (if available locally). Test different distance metrics and normalization.
   * **Why:** Embeddings drive retrieval; modest embedding improvements give big RAG gains.
   * **Outcome:** Higher retrieval precision at k, better prompts to LLM.

5. **Model-router & task mapping validation**

   * **What:** Enable `ENABLE_DYNAMIC_MODEL_SELECTION=True` in a dev run and test routing for tasks (summarize → small, QA → tiny, reasoning → mistral). Log choices and latency.
   * **Why:** Confirms router saves CPU without harming quality.
   * **Outcome:** Rules/thresholds to keep Mistral reserved for heavy tasks.

6. **RBAC stress tests + audit logs**

   * **What:** Create automated tests that query restricted docs with different API keys/roles to measure false exposures and false denies. Record detailed audit logs of decisions.
   * **Why:** RBAC correctness is mission-critical for enterprise simulation.
   * **Outcome:** Confident RBAC with count of policy violations and remediation plan.

7. **Prompt engineering & persona prefixes**

   * **What:** A/B test different LLM prefixes (short vs. detailed persona, instruction templates, tone guidance). Measure answer helpfulness and hallucination.
   * **Why:** Prefixing often reduces hallucination and increases alignment with RBAC.
   * **Outcome:** Stable prefix templates per support category and per role.

8. **Factuality & hallucination mitigation strategy**

   * **What:** Add “source citing” prompts (e.g., ask LLM to include doc ids or quotes), enforce “I don’t know” fallback, and measure hallucination rate. Consider conservative temperature / deterministic generation settings.
   * **Why:** Offline LLMs can confidently hallucinate; explicit guardrails reduce that.
   * **Outcome:** Lower hallucination and traceable answers (doc ids + snippets).

# Concrete evaluation metrics to track

* **Retrieval:** Precision@k, Recall@k, MRR
* **End-to-end:** Exact match / BLEU / ROUGE for short answers where applicable
* **Safety/RBAC:** False exposure rate (% of filtered docs returned) and false denial rate
* **Factuality:** Hallucination rate (human or automatic checks)
* **Latency/Cost:** Average response time (s) and CPU usage per request
* **User quality:** Helpfulness/clarity rating (1–5) from manual checks

# Small experiments you can run locally (no code here — conceptual)

* **A. Ablation: embeddings only vs embeddings + re-ranking** — does re-ranking (e.g., by LLM) improve final answer quality?
* **B. Onboarding personalization impact** — compare personalized prefix vs stateless on the same queries to measure helpfulness lift.
* **C. Public-summary efficacy** — for filtered docs test whether `public_summary` is sufficient to answer policy questions without exposing details.

# Data & tooling housekeeping (important)

* Build a small labeled **eval dataset** (CSV or JSONL) with fields: `query, expected_answer_or_policy, role, department, expected_access`
* Keep a versioned copy of **ingestion metadata** for documents (so you can reproduce RBAC issues)
* Enable verbose **decision logging** (which chunks were visible, which filtered with reasons) for each query — critical for debugging.

# Resource / configuration tips for 16GB RAM CPU

* Use smaller `n_ctx` (1024–2048) for Mistral in most tests.
* Use `n_batch=1` and moderate `n_threads` (half cores) to avoid OOM.
* Keep embedding ops batched but small when computing many vectors locally.
* Use swap (8GB) if necessary for heavy seeding operations, but prefer smaller chunking during ingest.

# Low-effort wins (do first)

1. Create the 100–300 query eval suite (highest impact).
2. Run a set of RBAC automated test cases (exposure/deny checks).
3. Tune chunk size to one chosen setting and re-index a sample doc set.
4. Optimize prompt prefix and temperature to reduce hallucinations.

---

If you want, next I can (pick one):

* produce a **detailed eval plan** (exact metrics, CSV schema, how to run manual checks),
* or draft the **RBAC test matrix** (role × sensitivity × expected outcome) you can use to automate tests,
* or outline the **prompt templates** to A/B test per support category.

Which of those do you want me to prepare next?
