Here are **ready-to-run cURL examples** so you can quickly test:

* sentiment & tone detection
* tone-aware support chat
* tone-guided RAG
* sentiment analytics
* model routing behaviors

All using your existing APIs.

---

# ✅ 1. Test Sentiment / Tone Detection Directly

(works if you added `/api/local/sentiment` endpoint)

### Angry

```bash
curl -X POST http://localhost:8000/api/local/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text":"I am extremely angry! Why is this broken again?"}'
```

### Confused

```bash
curl -X POST http://localhost:8000/api/local/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text":"I don’t understand step 3, can you explain?"}'
```

### Happy

```bash
curl -X POST http://localhost:8000/api/local/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text":"Wow, thank you! This helped a lot!"}'
```

---

# ✅ 2. Test Support Chat + Tone Storage

(Saves tone into `support_messages` table)

### Start a session

```bash
curl -X POST http://localhost:8000/api/local/session/start
```

This returns:

```json
{
  "session_id": "sess_xxxxx",
  "message": "Session started"
}
```

Copy `session_id`.

---

### User sends an angry message (tone should be “angry”)

```bash
curl -X POST http://localhost:8000/api/local/query \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: sess_xxxxx" \
  -d '{
        "query_text": "This is ridiculous, it keeps failing again!",
        "requester": {"role":"Employee", "department":"IT"}
      }'
```

Tone stored in DB:

* `"tone": "angry"`

---

### Check stored messages

(assumes you added `/api/local/messages/:sessionId` or you query SQLite)

```bash
curl http://localhost:8000/api/local/messages/sess_xxxxx
```

---

# ✅ 3. Test Tone-Aware RAG Response

(Triggers “empathetic/softened” prefix injection)

```bash
curl -X POST http://localhost:8000/api/local/query \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: sess_xxxxx" \
  -d '{
        "query_text": "My laptop still won’t start after I tried everything.",
        "requester": {"role":"Employee", "department":"IT"}
      }'
```

Expected assistant behavior:

* Calmer tone
* Apology
* Step-by-step guidance

---

# ✅ 4. Test Neutral Tone RAG Request

(should not add empathy guidance)

```bash
curl -X POST http://localhost:8000/api/local/query \
  -H "Content-Type: application/json" \
  -d '{
        "query_text": "What are the company leave policies?",
        "requester": {"role":"Employee", "department":"HR"}
      }'
```

---

# ✅ 5. Test Sentiment / Tone Analytics

(after several user messages)

```bash
curl http://localhost:8000/api/local/sentiment/stats
```

Expected JSON:

```json
{
  "tone_counts": {
     "angry": 3,
     "confused": 2,
     "happy": 1
  },
  "sentiment_counts": {
     "negative": 5,
     "neutral": 2,
     "positive": 1
  },
  "tone_by_department": [
     {"department":"IT","tone":"angry","count":2},
     {"department":"HR","tone":"confused","count":1}
  ]
}
```

---
#Need to implement it later on if required
# ✅ 6. Test Model Auto-Routing (if router endpoint added)

If you added something like `/api/local/debug/model-route`:

```bash
curl -X POST http://localhost:8000/api/local/debug/model-route \
  -H "Content-Type: application/json" \
  -d '{"task":"summarize"}'
```

Expected:

```json
{"model":"small"}
```

```bash
curl -X POST http://localhost:8000/api/local/debug/model-route \
  -H "Content-Type: application/json" \
  -d '{"task":"reason"}'
```

Expected:

```json
{"model":"mistral"}
```

---