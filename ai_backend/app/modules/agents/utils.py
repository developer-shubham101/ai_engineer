"""Utility functions for agent tools."""

from typing import Dict, List, Any


class MockDataProvider:
    """Provides mock data for research and testing."""
    
    @staticmethod
    def get_mock_tickets(user_id: str) -> List[Dict[str, Any]]:
        """Get mock ticket data."""
        return [
            {
                "id": "TKT-001",
                "status": "IN_PROGRESS",
                "title": "Login Issues",
                "created": "2024-01-15",
                "priority": "High"
            },
            {
                "id": "TKT-002", 
                "status": "RESOLVED",
                "title": "Password Reset",
                "created": "2024-01-10",
                "priority": "Medium"
            }
        ]
    
    @staticmethod
    def get_mock_comments(ticket_id: str) -> List[Dict[str, Any]]:
        """Get mock comment data."""
        mock_comments = {
            "TKT-001": [
                {"author": "user", "date": "2024-01-15", "text": "Cannot login to the system"},
                {"author": "support", "date": "2024-01-15", "text": "Please try clearing browser cache"},
                {"author": "user", "date": "2024-01-16", "text": "Still not working"}
            ],
            "TKT-002": [
                {"author": "user", "date": "2024-01-10", "text": "Need password reset"},
                {"author": "support", "date": "2024-01-10", "text": "Reset link sent to email"}
            ]
        }
        return mock_comments.get(ticket_id, [])
    
    @staticmethod
    def get_research_data(data_type: str) -> str:
        """Get mock research data."""
        research_data = {
            "users": "User Research: 1,250 total users, 85% retention, 15% growth rate",
            "performance": "Performance Research: 99.2% uptime, 1.8s avg response, 15% improvement",
            "trends": "Trend Research: Mobile usage +25%, API calls +40%, support tickets -10%",
            "engagement": "Engagement Research: 65% daily active, 4.2 sessions/user, 12min avg session",
            "feedback": "Feedback Research: 4.3/5 rating, 78% recommend, top request: mobile app"
        }
        return research_data.get(data_type, f"Research data for '{data_type}': Custom metrics would be generated here")


class AnalysisProvider:
    """Provides analysis utilities."""
    
    @staticmethod
    def analyze_query(query: str) -> str:
        """Analyze query and return insights."""
        query_lower = query.strip().lower()
        
        if "ticket" in query_lower:
            return "Ticket Analysis: 60% resolved within 24h, 25% high priority, peak hours 9-11 AM"
        elif "user" in query_lower:
            return "User Analysis: 150 active users, 80% satisfaction rate, top issues: login (40%), password (30%)"
        elif "performance" in query_lower:
            return "Performance Analysis: Avg response time 2.3s, 99.5% uptime, peak load 500 req/min"
        else:
            return f"Analysis for '{query}': Sample metrics and patterns would be calculated here"


class DocumentFormatter:
    """Formats document search results."""
    
    @staticmethod
    def format_search_results(results: List[Dict[str, Any]], max_results: int = 3) -> str:
        """Format search results for display."""
        if not results:
            return "No relevant documents found."
        
        formatted_results = []
        for i, doc in enumerate(results[:max_results], 1):
            text = doc.get("text", "")[:200]
            metadata = doc.get("metadata", {})
            dept = metadata.get("department", "Unknown")
            formatted_results.append(f"Document {i} ({dept}): {text}...")
        
        return "\n".join(formatted_results)


class StepFormatter:
    """Formats agent execution steps."""
    
    @staticmethod
    def format_final_answer(steps: List[Dict[str, Any]]) -> str:
        """Format final answer from execution steps."""
        if not steps:
            return "I couldn't find relevant information to answer your question."
        
        final_answer = "Based on my analysis:\n\n"
        for i, step in enumerate(steps, 1):
            tool_name = step.get("tool", "unknown")
            result = step.get("result", "")
            final_answer += f"Step {i} ({tool_name}): {result}\n\n"
        
        return final_answer.strip()
    
    @staticmethod
    def create_step_record(
        step_num: int,
        tool_name: str,
        input_data: str,
        result: str,
        timestamp: float,
        error: bool = False
    ) -> Dict[str, Any]:
        """Create standardized step record."""
        return {
            "step": step_num,
            "tool": tool_name,
            "input": input_data,
            "result": result,
            "timestamp": timestamp,
            "error": error
        }