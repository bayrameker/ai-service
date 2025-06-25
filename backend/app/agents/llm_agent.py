"""
LLM-powered Agent implementation using LangGraph
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from app.agents.base import BaseAgent, AgentConfig, Task, AgentState
from app.llm_providers.base import LLMRequest
from app.llm_providers.manager import LLMProviderManager

logger = logging.getLogger(__name__)


class LLMAgent(BaseAgent):
    """LLM-powered agent using LangGraph for workflow management"""
    
    def __init__(self, config: AgentConfig, llm_manager: LLMProviderManager):
        super().__init__(config)
        self.llm_manager = llm_manager
        self.conversation_history: List[Dict[str, str]] = []
        self.context: Dict[str, Any] = {}
        
        # Default system prompt if not provided
        if not self.config.system_prompt:
            self.config.system_prompt = self._generate_default_system_prompt()
    
    def _generate_default_system_prompt(self) -> str:
        """Generate default system prompt based on agent role"""
        base_prompt = f"""You are an AI agent named {self.config.name} with the role of {self.config.role}.

Description: {self.config.description}

Your capabilities include:
{chr(10).join(f"- {cap.name}: {cap.description}" for cap in self.config.capabilities)}

Guidelines:
1. Always be helpful, accurate, and professional
2. Break down complex tasks into manageable steps
3. Provide clear and actionable responses
4. Ask for clarification when needed
5. Learn from previous interactions to improve performance

Current context and previous interactions will be provided to help you maintain continuity."""
        
        return base_prompt
    
    async def initialize(self) -> None:
        """Initialize the LLM agent"""
        try:
            logger.info(f"Initializing LLM agent {self.config.name}")
            
            # Test LLM connection
            test_request = LLMRequest(
                prompt="Hello, I am initializing. Please respond with 'Agent initialized successfully.'",
                model=self.config.llm_model,
                max_tokens=50,
                temperature=0.1
            )
            
            response = await self.llm_manager.generate(
                test_request, 
                provider_name=self.config.llm_provider
            )
            
            if response and response.content:
                self.state = AgentState.IDLE
                logger.info(f"LLM agent {self.config.name} initialized successfully")
            else:
                raise Exception("Failed to get response from LLM")
                
        except Exception as e:
            self.state = AgentState.ERROR
            logger.error(f"Failed to initialize LLM agent {self.config.name}: {e}")
            raise
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a task using LLM"""
        try:
            logger.info(f"LLM agent {self.config.name} executing task: {task.title}")
            
            # Prepare context
            context_info = self._prepare_context(task)
            
            # Prepare messages for LLM
            messages = []
            
            # Add system message
            messages.append({
                "role": "system",
                "content": self.config.system_prompt
            })
            
            # Add context if available
            if context_info:
                messages.append({
                    "role": "system",
                    "content": f"Context: {context_info}"
                })
            
            # Add conversation history (last 10 messages)
            recent_history = self.conversation_history[-10:] if self.conversation_history else []
            messages.extend(recent_history)
            
            # Add current task
            task_prompt = f"""Task: {task.title}

Description: {task.description}

Priority: {task.priority}

Please execute this task and provide a detailed response with your findings, actions taken, and results."""
            
            messages.append({
                "role": "user",
                "content": task_prompt
            })
            
            # Create LLM request
            llm_request = LLMRequest(
                messages=messages,
                model=self.config.llm_model,
                max_tokens=2000,
                temperature=0.7
            )
            
            # Generate response
            response = await self.llm_manager.generate(
                llm_request,
                provider_name=self.config.llm_provider
            )
            
            if not response or not response.content:
                raise Exception("No response from LLM")
            
            # Update conversation history
            self.conversation_history.append({
                "role": "user",
                "content": task_prompt
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": response.content
            })
            
            # Prepare result
            result = {
                "response": response.content,
                "model_used": response.model,
                "provider_used": response.provider,
                "tokens_used": response.usage.get("total_tokens", 0),
                "latency_ms": response.latency_ms,
                "execution_time": datetime.utcnow().isoformat(),
                "agent_id": self.id,
                "agent_name": self.config.name
            }
            
            # Extract structured data if possible
            structured_result = self._extract_structured_result(response.content)
            if structured_result:
                result["structured_data"] = structured_result
            
            logger.info(f"LLM agent {self.config.name} completed task: {task.title}")
            return result
            
        except Exception as e:
            logger.error(f"LLM agent {self.config.name} failed to execute task: {e}")
            raise
    
    def _prepare_context(self, task: Task) -> str:
        """Prepare context information for the task"""
        context_parts = []
        
        # Add agent context
        context_parts.append(f"Agent: {self.config.name} ({self.config.role})")
        context_parts.append(f"Current time: {datetime.utcnow().isoformat()}")
        
        # Add task metadata
        if task.metadata:
            context_parts.append(f"Task metadata: {task.metadata}")
        
        # Add agent context
        if self.context:
            context_parts.append(f"Agent context: {self.context}")
        
        # Add recent task history
        if self.completed_tasks:
            recent_tasks = self.completed_tasks[-3:]  # Last 3 tasks
            task_summaries = []
            for t in recent_tasks:
                summary = f"- {t.title} ({t.status})"
                if t.result and "response" in t.result:
                    # Truncate response for context
                    response_preview = t.result["response"][:100] + "..." if len(t.result["response"]) > 100 else t.result["response"]
                    summary += f": {response_preview}"
                task_summaries.append(summary)
            
            if task_summaries:
                context_parts.append(f"Recent completed tasks:\n{chr(10).join(task_summaries)}")
        
        return "\n\n".join(context_parts)
    
    def _extract_structured_result(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract structured data from LLM response"""
        # This is a simple implementation
        # In a real system, you might use more sophisticated parsing
        
        structured = {}
        
        # Look for common patterns
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for section headers
            if line.endswith(':') and len(line.split()) <= 3:
                current_section = line[:-1].lower().replace(' ', '_')
                structured[current_section] = []
            elif current_section and line.startswith('-'):
                # List item
                structured[current_section].append(line[1:].strip())
            elif current_section and not line.startswith('-'):
                # Continue previous section
                if isinstance(structured[current_section], list):
                    if structured[current_section]:
                        structured[current_section][-1] += f" {line}"
                    else:
                        structured[current_section].append(line)
        
        return structured if structured else None
    
    async def shutdown(self) -> None:
        """Shutdown the agent"""
        try:
            logger.info(f"Shutting down LLM agent {self.config.name}")
            
            # Cancel any running tasks
            for task in self.current_tasks.values():
                task.status = "cancelled"
                task.completed_at = datetime.utcnow()
                task.error = "Agent shutdown"
            
            self.current_tasks.clear()
            self.state = AgentState.STOPPED
            
            logger.info(f"LLM agent {self.config.name} shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during agent shutdown: {e}")
    
    def update_context(self, key: str, value: Any) -> None:
        """Update agent context"""
        self.context[key] = value
    
    def get_context(self, key: str) -> Any:
        """Get context value"""
        return self.context.get(key)
    
    def clear_context(self) -> None:
        """Clear agent context"""
        self.context.clear()
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Get conversation history"""
        if limit:
            return self.conversation_history[-limit:]
        return self.conversation_history.copy()
    
    def clear_conversation_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history.clear()
