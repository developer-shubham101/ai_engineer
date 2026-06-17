# Llama.cpp Server Commands

## Download llama.cpp

Download the prebuilt Windows CPU binary from the official releases page:

👉 **[https://github.com/ggml-org/llama.cpp/releases](https://github.com/ggml-org/llama.cpp/releases)**

Look for a release asset named like `llama-b7445-bin-win-cpu-x64.zip`, extract it, and you will have `llama-server.exe` inside.

## Running Local LLM Server

Navigate to the extracted llama.cpp binary directory:
```bash
cd C:\Users\<your-username>\Downloads\llama-b7445-bin-win-cpu-x64
```

## Command Structure
```bash
.\llama-server.exe -m "<MODEL_PATH>" -c <CONTEXT_SIZE> -t <THREADS> --host <HOST> --port <PORT>
```

### Parameters:
- `-m`: Model file path (GGUF format)
- `-c`: Context window size (4096 tokens)
- `-t`: Number of CPU threads (8)
- `--host`: Server host address (127.0.0.1 for localhost)
- `--port`: Server port (8080)

## Example Commands

### Mistral 7B Model
```bash
.\llama-server.exe -m "I:\Workspace\GitHub\ai_engineer\ai_backend\models\mistral-7b-instruct-v0.2.Q3_K_M.gguf" -c 4096 -t 8 --host 127.0.0.1 --port 8080
```

### Llama 3.2 Model
```bash
.\llama-server.exe -m "I:\Workspace\GitHub\ai_engineer\ai_backend\models\llama-3.2" -c 4096 -t 8 --host 127.0.0.1 --port 8080
```

## Server Access
Once running, the server will be available at: `http://127.0.0.1:8080`

## API Examples

### Health Check
```bash
curl http://127.0.0.1:8080/health
```

### Chat Completion (OpenAI Compatible)
```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral",
    "messages": [
      {"role": "user", "content": "What is artificial intelligence?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Text Completion
```bash
curl -X POST http://127.0.0.1:8080/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of AI is",
    "n_predict": 50,
    "temperature": 0.8
  }'
```

### Model Information
```bash
curl http://127.0.0.1:8080/v1/models
```

### Server Metrics
```bash
curl http://127.0.0.1:8080/metrics
```