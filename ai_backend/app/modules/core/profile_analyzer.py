"""User profile analysis service."""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ProfileAnalyzer:
    """User profile analysis service."""

    def __init__(self, user_manager=None, session_manager=None):
        self.user_manager = user_manager
        self.session_manager = session_manager

    async def analyze_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Analyze user profile and return insights."""
        if not self.user_manager:
            return {}

        user = await self.user_manager.get_user(user_id)
        if not user:
            return {}

        profile = user.get("profile", {})

        analysis = {
            "user_id": user_id,
            "role": user.get("role"),
            "department": user.get("department"),
            "profile_completeness": self._calculate_profile_completeness(profile),
            "preferences": self._extract_preferences(profile),
            "communication_style": await self._analyze_communication_style(user_id)
        }

        return analysis

    async def get_personalization_context(self, user_id: str, session_id: Optional[str] = None) -> str:
        """Get personalization context for prompts."""
        analysis = await self.analyze_user_profile(user_id)

        context_parts = []

        # Add role and department
        if analysis.get("role"):
            context_parts.append(f"Role: {analysis['role']}")

        if analysis.get("department"):
            context_parts.append(f"Department: {analysis['department']}")

        # Add communication style
        comm_style = analysis.get("communication_style", {})
        if comm_style.get("tone"):
            context_parts.append(f"Preferred tone: {comm_style['tone']}")

        # Add recent session context if available
        if session_id and self.session_manager:
            recent_messages = await self.session_manager.fetch_recent_messages(session_id, limit=2)
            if recent_messages:
                last_sentiment = recent_messages[-1].get("sentiment", "neutral")
                context_parts.append(f"Recent sentiment: {last_sentiment}")

        return " | ".join(context_parts)

    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """Update user preferences."""
        if not self.user_manager:
            return False

        # Get current profile
        current_profile = await self.user_manager.get_user_metadata(user_id, "profile") or {}

        # Update preferences
        current_profile.update(preferences)

        # Save updated profile
        return await self.user_manager.set_user_metadata(user_id, "profile", current_profile)

    def _calculate_profile_completeness(self, profile: Dict[str, Any]) -> float:
        """Calculate profile completeness score."""
        required_fields = ["name", "position", "department", "preferences"]
        optional_fields = ["bio", "skills", "interests", "communication_style"]

        required_score = sum(1 for field in required_fields if profile.get(field))
        optional_score = sum(1 for field in optional_fields if profile.get(field))

        # Weight required fields more heavily
        total_score = (required_score * 2) + optional_score
        max_score = (len(required_fields) * 2) + len(optional_fields)

        return total_score / max_score if max_score > 0 else 0.0

    def _extract_preferences(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Extract user preferences from profile."""
        preferences = profile.get("preferences", {})

        # Default preferences
        default_prefs = {
            "response_length": "medium",
            "technical_level": "moderate",
            "communication_style": "professional",
            "preferred_format": "structured"
        }

        # Merge with user preferences
        return {**default_prefs, **preferences}

    async def _analyze_communication_style(self, user_id: str) -> Dict[str, Any]:
        """Analyze user's communication style from message history."""
        if not self.session_manager:
            return {"tone": "neutral", "style": "professional"}

        # This is a placeholder - in production, you'd analyze actual message history
        # across all sessions for this user

        return {
            "tone": "professional",
            "style": "direct",
            "formality": "moderate",
            "preferred_length": "concise"
        }

    async def get_role_based_guidance(self, role: str) -> Dict[str, Any]:
        """Get role-based guidance for responses."""
        role_guidance = {
            "SuperAdmin": {
                "tone": "direct",
                "detail_level": "comprehensive",
                "focus": "strategic",
                "format": "executive_summary"
            },
            "Manager": {
                "tone": "professional",
                "detail_level": "detailed",
                "focus": "operational",
                "format": "structured"
            },
            "HR": {
                "tone": "empathetic",
                "detail_level": "thorough",
                "focus": "policy_compliance",
                "format": "policy_focused"
            },
            "Employee": {
                "tone": "helpful",
                "detail_level": "practical",
                "focus": "task_oriented",
                "format": "step_by_step"
            },
            "Guest": {
                "tone": "welcoming",
                "detail_level": "basic",
                "focus": "informational",
                "format": "simple"
            }
        }

        return role_guidance.get(role, role_guidance["Employee"])
