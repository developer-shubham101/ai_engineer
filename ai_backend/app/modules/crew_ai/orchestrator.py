"""CrewAI orchestrator implementation using official CrewAI library."""

import logging
import time
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional

from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM

from .interfaces import ICrewOrchestrator, CrewRequest, CrewResponse

logger = logging.getLogger(__name__)


class CrewOrchestrator(ICrewOrchestrator):
    """CrewAI orchestrator using official CrewAI library with YAML configuration."""
    
    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider
        self.config_path = Path("crew_config")
        self.agents_config = self._load_yaml("agents.yaml")
        self.tasks_config = self._load_yaml("tasks.yaml")
        
        # Create LLM instance for CrewAI
        self.llm = self._create_llm()
    
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(self.config_path / filename, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            return {}
    
    def _create_llm(self) -> Optional[LLM]:
        """Create LLM instance for CrewAI."""
        try:
            if self.llm_provider:
                # Use local LLM if available
                return LLM(
                    model="local",
                    base_url="http://localhost:8000",  # Adjust as needed
                    temperature=0.7
                )
            else:
                # Fallback to OpenAI if configured
                import os
                if os.getenv("OPENAI_API_KEY"):
                    return LLM(model="gpt-3.5-turbo", temperature=0.7)
        except Exception as e:
            logger.warning(f"Failed to create LLM: {e}")
        return None
    
    def _create_agent(self, agent_key: str) -> Agent:
        """Create CrewAI agent from configuration."""
        config = self.agents_config.get(agent_key, {})
        return Agent(
            role=config.get("role", agent_key),
            goal=config.get("goal", f"Complete tasks as {agent_key}"),
            backstory=config.get("backstory", f"You are a {agent_key} agent."),
            verbose=config.get("verbose", True),
            llm=self.llm
        )
    
    def _create_task(self, task_key: str, agent: Agent, topic: str) -> Task:
        """Create CrewAI task from configuration."""
        config = self.tasks_config.get(task_key, {})
        return Task(
            description=config.get("description", f"Complete task for {topic}").format(topic=topic),
            expected_output=config.get("expected_output", "Task completion summary"),
            agent=agent
        )
    
    async def execute_workflow(self, request: CrewRequest, user: Optional[Dict[str, Any]] = None) -> CrewResponse:
        """Execute CrewAI workflow."""
        start_time = time.time()
        
        try:
            if request.workflow_type == "debate":
                result, agents_used = await self._execute_debate_crew(request.topic)
            elif request.workflow_type == "research":
                result, agents_used = await self._execute_research_crew(request.topic)
            else:
                raise ValueError(f"Unknown workflow type: {request.workflow_type}")
            
            execution_time = int((time.time() - start_time) * 1000)
            
            return CrewResponse(
                result=result,
                workflow_type=request.workflow_type,
                agents_used=agents_used,
                iterations=len(agents_used),
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"CrewAI workflow failed: {e}")
            execution_time = int((time.time() - start_time) * 1000)
            
            return CrewResponse(
                result=f"Workflow failed: {str(e)}",
                workflow_type=request.workflow_type,
                agents_used=[],
                iterations=0,
                execution_time_ms=execution_time,
                debug_info={"error": str(e)}
            )
    
    async def _execute_debate_crew(self, topic: str) -> tuple:
        """Execute debate workflow using CrewAI."""
        # Create agents
        advocate = self._create_agent("debate_advocate")
        critic = self._create_agent("debate_critic")
        moderator = self._create_agent("debate_moderator")
        
        # Create tasks
        advocate_task = self._create_task("debate_advocate_task", advocate, topic)
        critic_task = self._create_task("debate_critic_task", critic, topic)
        moderator_task = self._create_task("debate_moderator_task", moderator, topic)
        
        # Create crew
        crew = Crew(
            agents=[advocate, critic, moderator],
            tasks=[advocate_task, critic_task, moderator_task],
            process=Process.sequential,
            verbose=True
        )
        
        # Execute crew
        try:
            result = crew.kickoff()
            return str(result), ["Advocate", "Critic", "Moderator"]
        except Exception as e:
            logger.error(f"Debate crew execution failed: {e}")
            return f"Debate execution failed: {str(e)}", []
    
    async def _execute_research_crew(self, topic: str) -> tuple:
        """Execute research workflow using CrewAI."""
        # Create agents
        researcher = self._create_agent("researcher")
        analyst = self._create_agent("analyst")
        synthesizer = self._create_agent("synthesizer")
        
        # Create tasks
        research_task = self._create_task("research_task", researcher, topic)
        analysis_task = self._create_task("analysis_task", analyst, topic)
        synthesis_task = self._create_task("synthesis_task", synthesizer, topic)
        
        # Create crew
        crew = Crew(
            agents=[researcher, analyst, synthesizer],
            tasks=[research_task, analysis_task, synthesis_task],
            process=Process.sequential,
            verbose=True
        )
        
        # Execute crew
        try:
            result = crew.kickoff()
            return str(result), ["Researcher", "Analyst", "Synthesizer"]
        except Exception as e:
            logger.error(f"Research crew execution failed: {e}")
            return f"Research execution failed: {str(e)}", []
    
    def get_available_workflows(self) -> List[str]:
        """Get available workflow types."""
        return ["debate", "research"]