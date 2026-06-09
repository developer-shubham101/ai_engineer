export const BASE_API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const CONFIG_TOOLTIPS = {
  USE_LLM: 'If you disable this, RAG will not use LLM to generate answers. Only use it to embed and retrieve documents without AI-generated responses',
  USE_DOCS: 'Include document retrieval in search results. When enabled, system searches through uploaded documents to find relevant context',
  USE_HISTORY: 'Include conversation history in context for better continuity. Helps AI understand previous messages in the conversation',
  TOP_K: 'Number of most relevant documents to retrieve (1-20). Higher values provide more context but may include less relevant information',
  MAX_TOKENS: 'Maximum number of tokens in the response (128-2048). Controls the length of AI-generated answers',
  TEMPERATURE: 'Controls randomness in responses (0.0 = deterministic, 1.0 = creative). Balanced is 0.7. Lower values give consistent answers, higher values give more creative responses',
  MODEL_PROVIDER: 'Select the AI model provider for processing queries. Different providers offer various capabilities and performance characteristics',
  LOCAL_MODEL: 'Choose specific local model when using local provider. Each model has different strengths in reasoning, speed, and resource usage',
  THEME: 'Select application color theme preference. Choose between light, dark, or system-based theme that follows your device settings',
  TEMPLATE: 'Select prompt template to structure the AI response format. Configure in admin settings to manage templates. Based on template, LLM will give different response styles like simple, technical, descriptive, or structured responses',
  USE_TOOLS: 'Enable agentic tools so the AI can perform actions like file saving, web search, or calculations alongside answering questions'
}