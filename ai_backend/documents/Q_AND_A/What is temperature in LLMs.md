Short, sweet, and clear 👇

---

# ✅ **What is `temperature` in LLMs?**

**Temperature controls how random or creative the model’s output is.**

It is a number between **0.0 and 2.0** (most useful range: 0–1.2).

---

# 🎚️ **How it works (simple)**

Think of temperature as a **creativity dial**:

| Temperature | Behavior                                                    |
| ----------- | ----------------------------------------------------------- |
| **0.0**     | Very strict, deterministic, factual, same answer every time |
| **0.3–0.7** | Balanced, natural, useful for most tasks                    |
| **1.0**     | Creative, varied, more risk of mistakes                     |
| **>1.2**    | Very random, poetic, sometimes nonsense                     |

---

# 🔥 Tiny diagram (required)

```
Low Temp (0.0–0.3) → Precise, predictable
Medium (0.5–0.8) → Normal, conversational
High (1.0–1.5) → Creative, random

Temperature ↑ → randomness ↑
Temperature ↓ → accuracy ↑
```

---

# 🧠 Why this happens

Temperature affects how the model selects the **next token**.

* Low temperature → chooses the **most likely token**
* High temperature → chooses from a **wider set of possible tokens**

Example:

Prompt: `"Write a greeting"`

### Temp = 0.1

```
"Hello, how can I help you today?"
```

### Temp = 1.2

```
"Hey there! What’s up in your world today?"
```

### Temp = 1.8

```
"Yo cosmic traveler! What waves are you surfing today?"
```

---

# ⭐ Recommended for your project (CPU + enterprise-style assistant)

| Use Case               | Temperature            |
| ---------------------- | ---------------------- |
| RAG answers            | **0.1–0.3**            |
| Summaries              | **0.2–0.4**            |
| HR/IT/Company policies | **0.1** (safe answers) |
| Support chat           | **0.3–0.6**            |
| Creative writing       | **0.8–1.2**            |

---

# 🧩 Shortest explanation

> **Temperature controls randomness.
> Low = accurate and predictable.
> High = creative and unpredictable.**

---

If you're ready, ask **Q6**.
