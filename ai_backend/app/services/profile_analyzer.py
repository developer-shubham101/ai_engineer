# app/services/profile_analyzer.py
"""
Profile analysis service for personalized AI responses.
Analyzes user profiles and chat history to provide contextual suggestions.
"""

import logging
from typing import Dict, List, Optional, Any
import json

logger = logging.getLogger(__name__)

class ProfileAnalyzer:
    """Analyzes user profiles and suggests personalized actions."""
    
    def __init__(self):
        self.job_keywords = {
            "software": ["python", "java", "javascript", "react", "node", "developer", "programming", "coding"],
            "data": ["data", "analytics", "sql", "python", "machine learning", "ai", "statistics"],
            "design": ["design", "ui", "ux", "figma", "photoshop", "creative", "visual"],
            "marketing": ["marketing", "social media", "content", "seo", "campaigns", "digital"],
            "hr": ["hr", "human resources", "recruitment", "hiring", "people", "talent"],
            "finance": ["finance", "accounting", "excel", "financial", "budget", "analysis"]
        }
    
    def analyze_profile_for_jobs(self, profile: Dict[str, str], chat_history: List[Dict]) -> Dict[str, Any]:
        """Analyze user profile and suggest relevant job categories."""
        
        # Extract skills/experience from profile and chat
        text_to_analyze = " ".join([
            profile.get("name", ""),
            profile.get("experience", ""),
            profile.get("skills", ""),
            profile.get("background", ""),
            " ".join([msg.get("content", "") for msg in chat_history if msg.get("speaker") == "user"])
        ]).lower()
        
        # Match against job categories
        job_matches = {}
        for category, keywords in self.job_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_to_analyze)
            if score > 0:
                job_matches[category] = score
        
        # Sort by relevance
        sorted_matches = sorted(job_matches.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "relevant_categories": [cat for cat, score in sorted_matches[:3]],
            "profile_summary": self._create_profile_summary(profile),
            "suggested_actions": self._get_suggested_actions(sorted_matches, profile)
        }
    
    def _create_profile_summary(self, profile: Dict[str, str]) -> str:
        """Create a concise profile summary."""
        parts = []
        if profile.get("name"):
            parts.append(f"Name: {profile['name']}")
        if profile.get("location"):
            parts.append(f"Location: {profile['location']}")
        if profile.get("experience"):
            parts.append(f"Experience: {profile['experience']}")
        if profile.get("skills"):
            parts.append(f"Skills: {profile['skills']}")
        
        return " | ".join(parts) if parts else "Limited profile information"
    
    def _get_suggested_actions(self, job_matches: List[tuple], profile: Dict[str, str]) -> List[str]:
        """Generate personalized action suggestions."""
        actions = []
        
        if job_matches:
            top_category = job_matches[0][0]
            actions.append(f"I can help you find {top_category} positions that match your background")
            actions.append(f"Would you like me to draft a cover letter for {top_category} roles?")
            actions.append("I can provide interview tips specific to your field")
        
        if not profile.get("skills"):
            actions.append("Let me know your key skills to better match you with opportunities")
        
        if not profile.get("experience"):
            actions.append("Share your work experience for more targeted job suggestions")
        
        return actions[:3]  # Limit to top 3 suggestions

def build_personalized_prompt(
    base_prompt: str,
    profile: Dict[str, str],
    chat_history: List[Dict],
    query: str,
    requester: Dict[str, str]
) -> str:
    """Build enhanced prompt with profile analysis."""
    
    analyzer = ProfileAnalyzer()
    
    # Analyze profile for job-related queries
    is_job_query = any(keyword in query.lower() for keyword in 
                      ["job", "opening", "position", "career", "hire", "work", "employment"])
    
    enhanced_prompt = base_prompt
    
    if profile and is_job_query:
        analysis = analyzer.analyze_profile_for_jobs(profile, chat_history)
        
        enhanced_prompt += f"""

PERSONALIZATION CONTEXT:
User Profile: {analysis['profile_summary']}
Relevant Job Categories: {', '.join(analysis['relevant_categories'])}
Suggested Actions: {' | '.join(analysis['suggested_actions'])}

INSTRUCTIONS:
- Use the user's profile to provide personalized job recommendations
- If relevant openings exist, mention specific matches to their background
- Offer to help with cover letters, interview prep, or application guidance
- Be proactive in suggesting next steps based on their profile
"""
    
    elif profile:
        # General personalization for non-job queries
        profile_context = analyzer._create_profile_summary(profile)
        enhanced_prompt += f"""

USER CONTEXT: {profile_context}
- Personalize responses based on their background and role
- Reference their previous conversations when relevant
"""
    
    return enhanced_prompt