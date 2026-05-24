from app.modules.agents.orchestrators.autogen.autogen_orchestrator import AutoGenOrchestrator


class DummyModel:
    async def generate(self, prompt, max_tokens=256, temperature=0.1):
        return type("Resp", (), {"text": "Summary: done.", "usage": {}})()


def test_select_tools_for_stock_chart_query():
    orchestrator = AutoGenOrchestrator(model_client=DummyModel())
    tools = orchestrator._select_tools_for_query("show google 5 year stock chart")
    assert "get_stock_history" in tools
    assert "generate_stock_chart" in tools


def test_select_tools_for_crypto_query():
    orchestrator = AutoGenOrchestrator(model_client=DummyModel())
    tools = orchestrator._select_tools_for_query("bitcoin price today")
    assert tools == ["get_crypto_price"]


def test_select_tools_for_news_query():
    orchestrator = AutoGenOrchestrator(model_client=DummyModel())
    tools = orchestrator._select_tools_for_query("latest ai news")
    assert tools == ["web_search", "scrape_url"]


def test_select_tools_for_weather_trend_query():
    orchestrator = AutoGenOrchestrator(model_client=DummyModel())
    tools = orchestrator._select_tools_for_query("weather trend this week")
    assert "get_weather" in tools
    assert "generate_chart" in tools
