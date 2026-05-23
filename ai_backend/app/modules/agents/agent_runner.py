"""Agent runner implementing the UML flow for tool-based interactions.

This module handles function-based tools with static imports for better reliability.
"""
import json
import time
import logging
from typing import Dict, Any, List, Tuple

# Static imports for tools
from .function_tools.tool_stock import get_stock_price
from .function_tools.tool_weather import get_weather
from .function_tools.tool_file import save_text_file
from .function_tools.tool_web_search import web_search
from .function_tools.tool_web_scraper import scrape_url

logger = logging.getLogger(__name__)

# Tool registry with static definitions
REGISTRY: Dict[str, Dict[str, Any]] = {
    "get_stock_price": {
        "fn": get_stock_price,
        "args": ["symbol"],
        "description": "Get current stock price for a symbol"
    },
    "get_weather": {
        "fn": get_weather,
        "args": ["city"],
        "description": "Get current weather for a city"
    },
    "save_text_file": {
        "fn": save_text_file,
        "args": ["filename", "content"],
        "description": "Save text content to a file"
    },
    "web_search": {
        "fn": web_search,
        "args": ["query"],
        "description": "Search the internet for real-time information. Use for current events, facts, or anything not in internal documents"
    },
    "scrape_url": {
        "fn": scrape_url,
        "args": ["url"],
        "description": "Fetch and extract full text content from a URL. Use after web_search to get detailed information from a specific result"
    }
}


def build_system_prompt() -> str:
    """Build system prompt with available tools."""
    lines = [
        "You are an AI assistant with access to tools. You can only use ONE tool at a time.",
        "",
        "Available tools:",
        ""
    ]
    
    for i, (name, meta) in enumerate(REGISTRY.items(), 1):
        args_str = ", ".join(meta["args"])
        lines.append(f"{i}. {name}({args_str}) — {meta['description']}")
    
    lines += [
        "",
        "IMPORTANT RULES:",
        "- If you need to use a tool, respond with ONLY valid JSON, nothing else",
        "- Use this exact format: { \"tool\": \"tool_name\", \"args\": { \"arg1\": \"value\" } }",
        "- Do NOT include any explanation or additional text with the JSON",
        "- Use only ONE tool per response",
        "- If no tool is needed, respond normally with text (no JSON)",
        "",
        "Examples:",
        "- Tool needed: { \"tool\": \"get_stock_price\", \"args\": { \"symbol\": \"AAPL\" } }",
        "- No tool: The weather is sunny today.",
    ]
    
    return "\n".join(lines)


def call_tool(name: str, args: Dict[str, Any]) -> Any:
    """Call a tool by name with arguments."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown tool: {name}")
    
    return REGISTRY[name]["fn"](**args)


async def ask_llm(messages: List[Dict[str, str]], llm_provider) -> Tuple[str, Dict[str, Any]]:
    """Ask LLM with messages and return response and usage."""
    try:
        # Use the LLM provider to generate response
        full_prompt = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        response = await llm_provider.generate(
            prompt=full_prompt,
            max_tokens=256,  # Reduced for more focused responses
            temperature=0.1  # Lower temperature for more consistent JSON
        )
        
        return response.text, response.usage or {}
    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        raise


async def run_agent(user_prompt: str, llm_provider) -> str:
    """Run agent with tool support following the UML flow."""
    logger.info(f"Agent session start: {user_prompt}")
    
    system_prompt = build_system_prompt()
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    iteration = 0
    max_iterations = 5
    tool_results = []  # Track tool results for final summary
    
    while iteration < max_iterations:
        iteration += 1
        logger.info(f"Loop iteration {iteration}")

        logger.info(f"Query: {messages}")
        
        reply, usage = await ask_llm(messages, llm_provider)

        logger.info(f"Agent Replay: {reply}")
        
        # Clean the reply to handle mixed responses
        cleaned_reply = reply.strip()

        logger.info(f"Agent cleaned_reply: {cleaned_reply}")
        
        # Try to extract JSON from the response
        json_match = None
        try:
            # Look for JSON pattern in the response
            import re
            json_pattern = r'\{\s*"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{[^}]*\}\s*\}'
            match = re.search(json_pattern, cleaned_reply)
            
            if match:
                json_str = match.group(0)
                tool_call = json.loads(json_str)
                
                tool_name = tool_call["tool"]
                args = tool_call.get("args", {})
                
                logger.info(f"Tool decision: {tool_name} with args {args}")
                
                # Call the tool
                result = call_tool(tool_name, args)
                logger.info(f"Tool result: {result}")
                
                # Store result for final summary
                tool_results.append({
                    "tool": tool_name,
                    "args": args,
                    "result": result
                })
                
                # Add to conversation
                messages.append({"role": "assistant", "content": json_str})
                messages.append({"role": "user", "content": f"Tool result: {json.dumps(result)}. Continue or provide final answer."})
                
            else:
                # No valid JSON found, treat as final answer
                logger.info("No valid tool JSON found, treating as final answer")
                
                # If we have tool results, create a comprehensive summary
                if tool_results:
                    summary_parts = []
                    for tr in tool_results:
                        if tr["tool"] == "get_stock_price":
                            if tr["result"].get("status") == "success":
                                summary_parts.append(f"Stock price for {tr['args']['symbol']}: ${tr['result']['price']}")
                            else:
                                summary_parts.append(f"Failed to get stock price for {tr['args']['symbol']}: {tr['result'].get('error', 'Unknown error')}")
                        elif tr["tool"] == "get_weather":
                            if tr["result"].get("status") in ["success", "demo_data"]:
                                summary_parts.append(f"Weather in {tr['result']['city']}: {tr['result']['temperature']}, {tr['result']['description']}")
                            else:
                                summary_parts.append(f"Failed to get weather for {tr['args']['city']}: {tr['result'].get('error', 'Unknown error')}")
                        elif tr["tool"] == "web_search":
                            if tr["result"].get("status") == "success":
                                summary_parts.append(f"Web search for '{tr['args']['query']}': Found {tr['result']['count']} results via {tr['result']['source']}\n{tr['result']['formatted']}")
                            else:
                                summary_parts.append(f"Web search failed: {tr['result'].get('error', 'Unknown error')}")
                        elif tr["tool"] == "scrape_url":
                            if tr["result"].get("status") == "success":
                                summary_parts.append(f"Content from {tr['args']['url']}:\n{tr['result']['content']}")
                            else:
                                summary_parts.append(f"Failed to scrape {tr['args']['url']}: {tr['result'].get('error', 'Unknown error')}")
                        elif tr["tool"] == "save_text_file":
                            if tr["result"].get("status") == "success":
                                summary_parts.append(f"Saved file: {tr['result']['filename']} ({tr['result']['size']} characters)")
                            else:
                                summary_parts.append(f"Failed to save file {tr['args']['filename']}: {tr['result'].get('error', 'Unknown error')}")
                    
                    final_answer = "\n".join(summary_parts)
                    if cleaned_reply and not any(word in cleaned_reply.lower() for word in ["tool", "json", "{", "}"]):
                        final_answer += f"\n\n{cleaned_reply}"
                    
                    return final_answer
                else:
                    return cleaned_reply
                
        except json.JSONDecodeError:
            logger.info("No valid JSON found, treating as final answer")
            return cleaned_reply
        except KeyError as e:
            logger.warning(f"Malformed tool JSON - missing key: {e}")
            return f"Error: Malformed tool request - missing {e}"
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return f"Error: Tool execution failed - {e}"
    
    # If we reach max iterations, provide summary
    if tool_results:
        summary = "Completed the following actions:\n"
        for tr in tool_results:
            summary += f"- Used {tr['tool']} with result: {tr['result']}\n"
        return summary
    else:
        return "Maximum iterations reached without completing the task"