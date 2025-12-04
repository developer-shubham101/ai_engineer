import pytest
import asyncio
from app.modules.llm.prompt_chain import PromptChain, PromptContext

@pytest.mark.asyncio
async def test_prompt_chain_includes_history():
    chain = PromptChain()
    
    question = "What is my name?"
    context = "User profile says name is Alice."
    history = "[2024-01-01T12:00:00Z] USER: My name is Alice.\n[2024-01-01T12:00:05Z] ASSISTANT: Hello Alice!"
    
    final_prompt = await chain.build_prompt(
        question=question,
        context=context,
        history=history,
        user={"role": "Guest"}
    )
    
    print(f"\nFinal Prompt:\n{final_prompt}")
    
    assert "Conversation History:" in final_prompt
    assert "My name is Alice" in final_prompt
    assert "Hello Alice!" in final_prompt
    assert "Question: What is my name?" in final_prompt

if __name__ == "__main__":
    asyncio.run(test_prompt_chain_includes_history())
