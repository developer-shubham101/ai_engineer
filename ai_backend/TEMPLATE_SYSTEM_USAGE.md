# 🎭 Simplified Template System Usage Examples

## Overview

The new template system uses JSON message arrays instead of string parsing, making it much more reliable and easier to use. Templates are stored as message arrays in the database.

## Template Structure

Each template consists of:
- **messages**: Array of message objects with `role` and `content`
- **prompt_variables**: Pipe-separated list of variables used in the template

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
      "content": "You are a professional enterprise assistant. Use the provided context to answer questions accurately. User role: {user_role}, Department: {department}"
    },
    {
      "role": "user", 
      "content": "Context: {source_docs}\n\nQuestion: {user_question}"
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
- `{user_question}` - The user's question
- `{source_docs}` - Retrieved document context
- `{history}` - Conversation history
- `{user_role}` - User's role (Employee, Manager, etc.)
- `{department}` - User's department
- `{user_profile_summary}` - User profile information
- `{max_tokens}` - Token limit

## Benefits of New System

1. **No String Parsing**: Direct JSON message arrays
2. **Strict Role Enforcement**: System messages are properly separated
3. **Easy Testing**: Built-in template testing endpoint
4. **Variable Substitution**: Clean variable replacement
5. **Database Storage**: Templates stored in SQLite with versioning
6. **API Management**: Full CRUD operations via REST API

## Migration from Old System

Old templates using string parsing are automatically converted. The new system:
- Stores templates as JSON message arrays
- Provides backward compatibility
- Offers better error handling
- Enables easier template management

## Template Best Practices

1. **Clear System Messages**: Define the AI's role and behavior clearly
2. **Variable Naming**: Use descriptive variable names
3. **Role Consistency**: Keep system messages focused on behavior
4. **Testing**: Always test templates before production use
5. **Documentation**: Document template purpose and variables