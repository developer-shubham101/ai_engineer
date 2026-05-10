# 🎭 Simplified Template System Usage Examples

## Overview

The new template system uses JSON message arrays instead of string parsing, providing strict LLM behavior control. Templates are stored as message arrays in the database and follow a specific structure for optimal message ordering.

## Template Structure

Each template **must** contain exactly 2 messages:
1. **System message** (index 0) - Defines AI behavior and context
2. **User message** (index 1) - Contains the user's question with variables

### Message Flow
When a template is used, messages are assembled in this order:
1. **System message** (from template[0]) - with variable substitution
2. **Conversation history** (optional) - previous user/assistant exchanges
3. **User message** (from template[1]) - with variable substitution

This ensures all system instructions come first, followed by context (history), and finally the current user query.

## Example Templates

### 1. Pirate Template
```json
{
  "name": "pirate_template",
  "messages": [
    {
      "role": "system", 
      "content": "You are a pirate. Always respond like a pirate with 'Ahoy!' and pirate language. Use words like 'matey', 'arrr', 'ye', 'me hearty' in every response. Never break character - you are always a pirate."
    },
    {
      "role": "user", 
      "content": "{user_question}"
    }
  ],
  "prompt_variables": "user_question"
}
```

### 2. JSON Bot Template
```json
{
  "name": "json_bot_template",
  "messages": [
    {
      "role": "system", 
      "content": "You only respond with valid JSON. No extra text. Always format your response as proper JSON."
    },
    {
      "role": "user", 
      "content": "{user_question}"
    }
  ],
  "prompt_variables": "user_question"
}
```

### 3. Enterprise Assistant Template
```json
{
  "name": "enterprise_assistant",
  "messages": [
    {
      "role": "system", 
      "content": "You are a professional enterprise assistant. Use the provided context to answer questions accurately. User role: {user_role}, Department: {department}\n\nRelevant documents:\n{source_docs}"
    },
    {
      "role": "user", 
      "content": "{user_question}"
    }
  ],
  "prompt_variables": "user_role|department|source_docs|user_question"
}
```

## API Usage Examples

### Create a Template
```bash
curl -X POST "http://localhost:8000/api/templates" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "pirate_template",
    "messages": [
      {
        "role": "system",
        "content": "You are a pirate. Always respond like a pirate with \"Ahoy!\" and pirate language. Use words like \"matey\", \"arrr\", \"ye\", \"me hearty\" in every response. Never break character - you are always a pirate."
      },
      {
        "role": "user",
        "content": "{user_question}"
      }
    ],
    "prompt_variables": "user_question"
  }'
```

### List All Templates
```bash
curl "http://localhost:8000/api/templates"
```

### Get Specific Template
```bash
curl "http://localhost:8000/api/templates/pirate_template"
```

### Test Template
```bash
curl -X POST "http://localhost:8000/api/templates/test/pirate_template" \
  -H "Content-Type: application/json" \
  -d '{
    "user_question": "What is the capital of France?"
  }'
```

### Use Template in RAG Query
```bash
curl -X POST "http://localhost:8000/api/rag/local/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the capital of France?",
    "conversation_id": "test123",
    "prompt_template": "pirate_template",
    "use_llm": true
  }'
```

**Expected Response**: *"Ahoy matey! The capital of France be Paris, arrr! That be a fine city for any seafarin' soul to visit, me hearty!"*

## Available Variables

The system supports these variables in templates:
- `{user_question}` - The user's question (required)
- `{source_docs}` - Retrieved document context
- `{user_role}` - User's role (Employee, Manager, etc.)
- `{department}` - User's department
- `{user_profile_summary}` - User profile information
- `{max_tokens}` - Token limit

**Note**: History is automatically inserted between system and user messages, so you don't need a `{history}` variable.

## Message Assembly Example

Given this template:
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "{user_question}"}
  ]
}
```

With conversation history, the final message array becomes:
```json
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Previous question"},
  {"role": "assistant", "content": "Previous answer"},
  {"role": "user", "content": "Current question"}
]
```

## Benefits of New System

1. **No String Parsing**: Direct JSON message arrays
2. **Strict Role Enforcement**: System messages properly separated from user messages
3. **Proper Message Ordering**: System → History → User
4. **Easy Testing**: Built-in template testing endpoint
5. **Variable Substitution**: Clean variable replacement in both messages
6. **Database Storage**: Templates stored in SQLite with versioning
7. **API Management**: Full CRUD operations via REST API
8. **Multi-Provider Support**: Works with all LLM providers (local, OpenAI, Google, etc.)

## Provider Compatibility

All providers now accept message arrays:
- **LocalLLMProvider**: Converts messages to prompt string
- **OpenAILLMProvider**: Uses messages directly (native support)
- **GoogleLLMProvider**: Converts messages to prompt string
- **HuggingFaceLLMProvider**: Converts messages to prompt string
- **CustomLLMProvider**: Converts messages to prompt string
- **LlamaServerProvider**: Converts to AutoGen core messages

## Template Best Practices

1. **Always Use 2 Messages**: System (index 0) and User (index 1)
2. **Clear System Messages**: Define the AI's role and behavior clearly
3. **Variable Placement**: Put context variables in system message, question in user message
4. **Role Consistency**: Keep system messages focused on behavior and context
5. **Testing**: Always test templates before production use
6. **Documentation**: Document template purpose and required variables

## Migration from Old System

Old templates using string parsing are automatically converted. The new system:
- Stores templates as JSON message arrays
- Provides backward compatibility
- Offers better error handling
- Enables easier template management
- Ensures proper message ordering for all LLM providers