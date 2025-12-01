Below is the **quick comparison** of the CPU-friendly models based on the key things that matter for our micro-pilot:

---

# **CPU Model Comparison (Speed • Quantization • RAG Quality • Context Window)**

## **1. Phi-2 (Microsoft, 2.7B)**

* **Speed:** Very fast on CPU, even faster than many 3B models.
* **Quantization:** Supports 4-bit, 5-bit, and 8-bit GGUF — excellent CPU performance.
* **RAG Suitability:** Strong reasoning + good for short retrieval chunks; not great for long context synthesis.
* **Context Window:** ~2k–4k tokens depending on build.

---

## **2. Llama 3.2 — 1B & 3B (Meta)**

* **Speed:** 1B = extremely fast; 3B = still very usable on CPU.
* **Quantization:** 3B model works well at Q4_K_M or Q5_K_M; 1B runs well even at Q2_K.
* **RAG Suitability:** Very good small-model choice; better alignment and instruction following than most small models.
* **Context Window:** Typically 4k tokens; some builds support extended contexts.

---

## **3. Gemma 2B (Google)**

* **Speed:** Fast on CPU, optimized with low memory footprint.
* **Quantization:** Good results at Q4_K_M and Q6_K — maintains quality even at lower bitrates.
* **RAG Suitability:** High for its size—very stable, low hallucination, good for structured answers.
* **Context Window:** ~4k tokens.

---

## **4. Qwen2 1.5B (Alibaba)**

* **Speed:** Among the fastest on CPU due to architecture efficiency.
* **Quantization:** Works very well at ultra-low bit quantization (Q2–Q4) with minimal quality loss.
* **RAG Suitability:** Surprisingly strong retrieval grounding for its size, multilingual capability useful for global docs.
* **Context Window:** Usually 4k–8k tokens depending on build.

---

## **5. Mistral Tiny / NeMo Mini**

* **Speed:** Fastest in the list — designed for edge/low-power CPUs.
* **Quantization:** Excellent scaling down to Q2; super small memory footprint (<1GB).
* **RAG Suitability:** Good for high-speed queries, but weaker in long-form synthesis or multi-hop reasoning.
* **Context Window:** 2k–4k tokens depending on variant.

---

# **Short Recommendation (CTO opinion)**

### **Best overall for the demo:**

**Llama 3.2 (3B)** → best balance of speed, alignment, and RAG grounding.

### **Fastest & simplest option:**

**Phi-2** or **Qwen2 1.5B** → almost instant on CPU.

### **Safest/most stable answers:**

**Gemma 2B** → very clean, low-hallucination behavior.

### **Lightest for ultra-low hardware:**

**Mistral Tiny / NeMo Mini**

---