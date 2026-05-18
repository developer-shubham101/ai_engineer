"""Test script for agent tools functionality."""
import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.modules.agents.agent_runner import run_agent, _load_tools, REGISTRY


class MockLLMProvider:
    """Mock LLM provider for testing."""
    
    def __init__(self):
        self.call_count = 0
    
    async def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.1):
        """Mock generate method."""
        self.call_count += 1
        
        # Simulate tool call response
        if "get_stock_price" in prompt and self.call_count == 1:
            return MockResponse('{"tool": "get_stock_price", "args": {"symbol": "AAPL"}}')
        elif "get_weather" in prompt and self.call_count == 1:
            return MockResponse('{"tool": "get_weather", "args": {"city": "New York"}}')
        elif "save_text_file" in prompt and self.call_count == 1:
            return MockResponse('{"tool": "save_text_file", "args": {"filename": "test.txt", "content": "Hello World"}}')
        else:
            # Final response after tool execution
            return MockResponse("Based on the tool results, I can provide you with the requested information.")


class MockResponse:
    """Mock response object."""
    
    def __init__(self, text: str):
        self.text = text
        self.usage = {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}


async def test_agent_tools():
    """Test the agent tools functionality."""
    print("Testing agent tools...")
    
    # Ensure tools are loaded
    _load_tools()
    print(f"Loaded tools: {list(REGISTRY.keys())}")
    
    # Test with mock LLM
    mock_llm = MockLLMProvider()
    
    # Test stock tool
    result = await run_agent("Get the stock price for AAPL", mock_llm)
    print(f"Stock test result: {result}")
    
    # Test weather tool
    mock_llm.call_count = 0
    result = await run_agent("What's the weather in New York?", mock_llm)
    print(f"Weather test result: {result}")
    
    # Test file save tool
    mock_llm.call_count = 0
    result = await run_agent("Save 'Hello World' to a file called test.txt", mock_llm)
    print(f"File save test result: {result}")
    
    print("Agent tools test completed!")


if __name__ == "__main__":
    asyncio.run(test_agent_tools())