# 🤖 Agents Module Documentation

## Overview

The Agents module provides a **sandbox environment** for LangChain agent experimentation with **safety constraints** and **dynamic tool management**. Designed for research and learning purposes.

## Architecture

```
/api/agents/
├── query          # Execute agent workflows
├── status         # Get system status
├── tools          # List available tools
└── tools/{name}/test  # Test individual tools
```

## Safety Features

Following MOTIVATION.md guidelines:

- ✅ **Hard Step Limit**: Maximum 5 steps per workflow
- ✅ **Tool Whitelisting**: Only pre-approved tools allowed
- ✅ **No Direct DB Access**: Tools use wrapped functions only
- ✅ **Sandboxed Execution**: Isolated from production systems
- ✅ **Error Handling**: Graceful failure recovery

## Available Tools

### Core Tools

| Tool | Description | Input Example |
|------|-------------|---------------|
| `search_documents` | Search knowledge base | "vacation policy" |
| `get_user_tickets` | Get support tickets | "current" or user_id |
| `get_ticket_comments` | Get ticket history | "TKT-001" |
| `analyze_data` | Analyze patterns | "ticket trends" |
| `research_data` | Generate metrics | "users" or "performance" |
| `summarize_status` | Compile information | "data to summarize" |

### Research Tools

Tools generate **mock data** for experimentation:
- User analytics and engagement metrics
- Performance and system statistics  
- Trend analysis and forecasting
- Feedback and satisfaction data

## API Endpoints

### GET /api/agents/status

Get agent system status and available tools.

**Response:**
```json
{
  "available_tools": [
    {
      "name": "search_documents",
      "description": "Search company documents and knowledge base"
    }
  ],
  "max_steps": 5,
  "status": "active"
}
```

### POST /api/agents/query

Execute agent workflow with tools.

**Request:**
```json
{
  "question": "What is the status of my tickets?",
  "tools": ["get_user_tickets", "get_ticket_comments"],
  "max_steps": 5,
  "temperature": 0.1,
  "debug": true
}
```

**Response:**
```json
{
  "answer": "Based on my analysis:\n\nStep 1 (get_user_tickets): Found 2 tickets...",
  "steps": [
    {
      "step": 1,
      "tool": "get_user_tickets", 
      "input": "current",
      "result": "Found 2 tickets for user...",
      "timestamp": 1640995200.0
    }
  ],
  "tools_used": ["get_user_tickets"],
  "available_tools": ["search_documents", "get_user_tickets", ...],
  "debug_info": {
    "processing_time_ms": 1250,
    "enabled_tools": ["get_user_tickets"],
    "actual_steps": 1
  }
}
```

## Usage Examples

### Ticket Status Query
```bash
curl -X POST "/api/agents/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the status of my tickets?",
    "tools": ["get_user_tickets", "get_ticket_comments"],
    "debug": true
  }'
```

### Document Search
```bash
curl -X POST "/api/agents/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Search for vacation policy documents",
    "tools": ["search_documents"],
    "max_steps": 3
  }'
```

### Research Data Analysis
```bash
curl -X POST "/api/agents/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Analyze user engagement trends",
    "tools": ["research_data", "analyze_data"],
    "debug": true
  }'
```

## Development Guide

### Adding New Tools

1. **Create Tool Class:**
```python
class CustomTool(ITool):
    @property
    def name(self) -> str:
        return "custom_tool"
    
    @property  
    def description(self) -> str:
        return "Tool description for LLM"
    
    async def execute(self, input_data: str, context: Dict[str, Any]) -> str:
        # Tool implementation
        return "Tool result"
```

2. **Register Tool:**
```python
orchestrator = AgentOrchestrator()
orchestrator.register_tool(CustomTool())
```

### Tool Testing

Test individual tools:
```bash
curl -X POST "/api/agents/tools/search_documents/test" \
  -H "Content-Type: application/json" \
  -d "vacation policy"
```

## Research Applications

### Learning Scenarios
- **Multi-step reasoning**: Tool orchestration patterns
- **Error handling**: Agent failure modes
- **Tool selection**: LLM decision making
- **Context management**: Information flow between steps

### Experimentation
- **Custom workflows**: Domain-specific agent patterns
- **Tool combinations**: Optimal tool sequences
- **Performance analysis**: Step timing and efficiency
- **Failure analysis**: Common failure patterns

## Safety Constraints

### Hard Limits
- **Max Steps**: 5 (cannot be exceeded)
- **Tool Whitelist**: Only registered tools allowed
- **No System Access**: Tools cannot access system resources
- **Timeout Protection**: Steps have execution timeouts

### Error Handling
- **Graceful Degradation**: Partial results on tool failure
- **Step Recording**: All steps logged for debugging
- **Safe Fallbacks**: Default responses for critical failures

## Future Enhancements

### LangChain Integration
- **ReAct Agent**: Full LangChain agent implementation
- **Custom Prompts**: Agent reasoning templates
- **Memory Management**: Cross-session context
- **Tool Descriptions**: Enhanced LLM tool understanding

### Advanced Tools
- **API Integration**: External service tools
- **Data Processing**: Advanced analytics tools
- **Workflow Tools**: Multi-step process automation
- **Validation Tools**: Result verification and quality checks

## Troubleshooting

### Common Issues
- **Tool Not Found**: Check available tools with `/api/agents/tools`
- **Step Limit Exceeded**: Reduce max_steps or optimize workflow
- **Tool Execution Failed**: Use `/api/agents/tools/{name}/test` to debug
- **Empty Results**: Verify tool inputs and context

### Debug Mode
Enable debug mode for detailed execution information:
```json
{
  "question": "your question",
  "debug": true
}
```

Provides:
- Processing time metrics
- Tool execution details
- Step-by-step breakdown
- Error information