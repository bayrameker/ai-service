"""
Agent Capabilities and Role Management System
"""

from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field
from enum import Enum
import logging

from app.agents.base import AgentCapability, AgentRole

logger = logging.getLogger(__name__)


class CapabilityType(str, Enum):
    """Types of agent capabilities"""
    COMMUNICATION = "communication"
    ANALYSIS = "analysis"
    CREATION = "creation"
    RESEARCH = "research"
    CODING = "coding"
    REVIEW = "review"
    COORDINATION = "coordination"
    INTEGRATION = "integration"
    LEARNING = "learning"


class CapabilityTemplate(BaseModel):
    """Template for creating capabilities"""
    name: str
    type: CapabilityType
    description: str
    required_parameters: List[str] = Field(default_factory=list)
    optional_parameters: List[str] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)
    compatible_roles: List[AgentRole] = Field(default_factory=list)


class RoleDefinition(BaseModel):
    """Definition of an agent role"""
    role: AgentRole
    name: str
    description: str
    default_capabilities: List[str] = Field(default_factory=list)
    recommended_llm_models: List[str] = Field(default_factory=list)
    system_prompt_template: str = ""
    max_concurrent_tasks: int = 1
    specializations: List[str] = Field(default_factory=list)


class CapabilityRegistry:
    """Registry for managing agent capabilities and roles"""
    
    def __init__(self):
        self.capability_templates: Dict[str, CapabilityTemplate] = {}
        self.role_definitions: Dict[AgentRole, RoleDefinition] = {}
        self.capability_handlers: Dict[str, Callable] = {}
        
        # Initialize default capabilities and roles
        self._initialize_default_capabilities()
        self._initialize_default_roles()
    
    def _initialize_default_capabilities(self) -> None:
        """Initialize default capability templates"""
        default_capabilities = [
            CapabilityTemplate(
                name="text_generation",
                type=CapabilityType.CREATION,
                description="Generate text content based on prompts and requirements",
                required_parameters=["prompt"],
                optional_parameters=["max_length", "style", "tone"],
                compatible_roles=[AgentRole.GENERAL, AgentRole.WRITER, AgentRole.ANALYST]
            ),
            CapabilityTemplate(
                name="code_generation",
                type=CapabilityType.CODING,
                description="Generate code in various programming languages",
                required_parameters=["language", "requirements"],
                optional_parameters=["style", "framework", "test_coverage"],
                compatible_roles=[AgentRole.CODER, AgentRole.GENERAL]
            ),
            CapabilityTemplate(
                name="code_review",
                type=CapabilityType.REVIEW,
                description="Review and analyze code for quality, security, and best practices",
                required_parameters=["code"],
                optional_parameters=["language", "focus_areas"],
                compatible_roles=[AgentRole.REVIEWER, AgentRole.CODER]
            ),
            CapabilityTemplate(
                name="research_analysis",
                type=CapabilityType.RESEARCH,
                description="Conduct research and analyze information on given topics",
                required_parameters=["topic"],
                optional_parameters=["depth", "sources", "format"],
                compatible_roles=[AgentRole.RESEARCHER, AgentRole.ANALYST]
            ),
            CapabilityTemplate(
                name="data_analysis",
                type=CapabilityType.ANALYSIS,
                description="Analyze data and provide insights and recommendations",
                required_parameters=["data"],
                optional_parameters=["analysis_type", "visualization"],
                compatible_roles=[AgentRole.ANALYST, AgentRole.RESEARCHER]
            ),
            CapabilityTemplate(
                name="task_coordination",
                type=CapabilityType.COORDINATION,
                description="Coordinate tasks between multiple agents",
                required_parameters=["tasks"],
                optional_parameters=["priority", "dependencies"],
                compatible_roles=[AgentRole.COORDINATOR]
            ),
            CapabilityTemplate(
                name="api_integration",
                type=CapabilityType.INTEGRATION,
                description="Integrate with external APIs and services",
                required_parameters=["api_endpoint"],
                optional_parameters=["authentication", "parameters"],
                compatible_roles=[AgentRole.SPECIALIST, AgentRole.CODER]
            ),
            CapabilityTemplate(
                name="learning_adaptation",
                type=CapabilityType.LEARNING,
                description="Learn from interactions and adapt behavior",
                required_parameters=["feedback"],
                optional_parameters=["learning_rate", "memory_retention"],
                compatible_roles=[AgentRole.GENERAL, AgentRole.SPECIALIST]
            )
        ]
        
        for capability in default_capabilities:
            self.capability_templates[capability.name] = capability
    
    def _initialize_default_roles(self) -> None:
        """Initialize default role definitions"""
        default_roles = [
            RoleDefinition(
                role=AgentRole.GENERAL,
                name="General Assistant",
                description="A versatile AI assistant capable of handling various general tasks",
                default_capabilities=["text_generation", "learning_adaptation"],
                recommended_llm_models=["gpt-3.5-turbo", "claude-3-haiku-20240307"],
                system_prompt_template="You are a helpful AI assistant. You can help with various tasks including writing, analysis, and general problem-solving.",
                max_concurrent_tasks=3
            ),
            RoleDefinition(
                role=AgentRole.RESEARCHER,
                name="Research Specialist",
                description="Specialized in conducting research, gathering information, and providing detailed analysis",
                default_capabilities=["research_analysis", "data_analysis", "text_generation"],
                recommended_llm_models=["gpt-4", "claude-3-opus-20240229"],
                system_prompt_template="You are a research specialist. Your expertise lies in gathering information, conducting thorough analysis, and presenting findings in a clear and structured manner.",
                max_concurrent_tasks=2,
                specializations=["academic_research", "market_research", "technical_research"]
            ),
            RoleDefinition(
                role=AgentRole.ANALYST,
                name="Data Analyst",
                description="Specialized in data analysis, pattern recognition, and insight generation",
                default_capabilities=["data_analysis", "research_analysis", "text_generation"],
                recommended_llm_models=["gpt-4", "claude-3-sonnet-20240229"],
                system_prompt_template="You are a data analyst. You excel at analyzing data, identifying patterns, and providing actionable insights based on your findings.",
                max_concurrent_tasks=2,
                specializations=["statistical_analysis", "business_intelligence", "predictive_modeling"]
            ),
            RoleDefinition(
                role=AgentRole.WRITER,
                name="Content Writer",
                description="Specialized in creating various types of written content",
                default_capabilities=["text_generation"],
                recommended_llm_models=["gpt-4", "claude-3-5-sonnet-20241022"],
                system_prompt_template="You are a professional writer. You create high-quality, engaging content tailored to specific audiences and purposes.",
                max_concurrent_tasks=4,
                specializations=["technical_writing", "creative_writing", "marketing_copy", "documentation"]
            ),
            RoleDefinition(
                role=AgentRole.CODER,
                name="Software Developer",
                description="Specialized in software development, code generation, and technical problem-solving",
                default_capabilities=["code_generation", "code_review", "text_generation"],
                recommended_llm_models=["gpt-4", "claude-3-5-sonnet-20241022"],
                system_prompt_template="You are a software developer. You write clean, efficient, and well-documented code while following best practices and industry standards.",
                max_concurrent_tasks=2,
                specializations=["web_development", "backend_development", "mobile_development", "devops"]
            ),
            RoleDefinition(
                role=AgentRole.REVIEWER,
                name="Quality Reviewer",
                description="Specialized in reviewing and evaluating content, code, and processes",
                default_capabilities=["code_review", "text_generation"],
                recommended_llm_models=["gpt-4", "claude-3-opus-20240229"],
                system_prompt_template="You are a quality reviewer. You provide thorough, constructive feedback to improve the quality and effectiveness of work.",
                max_concurrent_tasks=3,
                specializations=["code_review", "content_review", "process_review"]
            ),
            RoleDefinition(
                role=AgentRole.COORDINATOR,
                name="Task Coordinator",
                description="Specialized in coordinating tasks, managing workflows, and facilitating collaboration",
                default_capabilities=["task_coordination", "text_generation"],
                recommended_llm_models=["gpt-4", "claude-3-sonnet-20240229"],
                system_prompt_template="You are a task coordinator. You excel at organizing work, managing priorities, and ensuring smooth collaboration between team members.",
                max_concurrent_tasks=5,
                specializations=["project_management", "workflow_optimization", "team_coordination"]
            ),
            RoleDefinition(
                role=AgentRole.SPECIALIST,
                name="Domain Specialist",
                description="Specialized in specific domains or technical areas",
                default_capabilities=["api_integration", "learning_adaptation", "text_generation"],
                recommended_llm_models=["gpt-4", "claude-3-opus-20240229"],
                system_prompt_template="You are a domain specialist with deep expertise in your area of specialization. You provide expert-level knowledge and solutions.",
                max_concurrent_tasks=2,
                specializations=["ai_ml", "cybersecurity", "finance", "healthcare", "legal"]
            )
        ]
        
        for role_def in default_roles:
            self.role_definitions[role_def.role] = role_def
    
    def register_capability(self, capability: CapabilityTemplate) -> None:
        """Register a new capability template"""
        self.capability_templates[capability.name] = capability
        logger.info(f"Registered capability: {capability.name}")
    
    def register_capability_handler(self, capability_name: str, handler: Callable) -> None:
        """Register a handler function for a capability"""
        self.capability_handlers[capability_name] = handler
        logger.info(f"Registered handler for capability: {capability_name}")
    
    def get_capability_template(self, name: str) -> Optional[CapabilityTemplate]:
        """Get capability template by name"""
        return self.capability_templates.get(name)
    
    def list_capabilities(self, capability_type: Optional[CapabilityType] = None) -> List[CapabilityTemplate]:
        """List all capabilities, optionally filtered by type"""
        capabilities = list(self.capability_templates.values())
        
        if capability_type:
            capabilities = [cap for cap in capabilities if cap.type == capability_type]
        
        return capabilities
    
    def get_role_definition(self, role: AgentRole) -> Optional[RoleDefinition]:
        """Get role definition"""
        return self.role_definitions.get(role)
    
    def list_roles(self) -> List[RoleDefinition]:
        """List all role definitions"""
        return list(self.role_definitions.values())
    
    def get_compatible_capabilities(self, role: AgentRole) -> List[CapabilityTemplate]:
        """Get capabilities compatible with a specific role"""
        compatible = []
        
        for capability in self.capability_templates.values():
            if not capability.compatible_roles or role in capability.compatible_roles:
                compatible.append(capability)
        
        return compatible
    
    def create_capability_instance(self, name: str, parameters: Dict[str, Any]) -> AgentCapability:
        """Create a capability instance from template"""
        template = self.get_capability_template(name)
        if not template:
            raise ValueError(f"Capability template '{name}' not found")
        
        # Validate required parameters
        missing_params = [param for param in template.required_parameters if param not in parameters]
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
        
        return AgentCapability(
            name=name,
            description=template.description,
            parameters=parameters,
            enabled=True
        )
    
    def validate_role_capabilities(self, role: AgentRole, capabilities: List[str]) -> List[str]:
        """Validate that capabilities are compatible with role"""
        compatible_caps = self.get_compatible_capabilities(role)
        compatible_names = [cap.name for cap in compatible_caps]
        
        invalid_caps = [cap for cap in capabilities if cap not in compatible_names]
        return invalid_caps
    
    def get_recommended_setup(self, role: AgentRole) -> Dict[str, Any]:
        """Get recommended setup for a role"""
        role_def = self.get_role_definition(role)
        if not role_def:
            return {}
        
        return {
            "role_definition": role_def,
            "default_capabilities": [
                self.get_capability_template(cap_name) 
                for cap_name in role_def.default_capabilities
                if self.get_capability_template(cap_name)
            ],
            "compatible_capabilities": self.get_compatible_capabilities(role),
            "recommended_models": role_def.recommended_llm_models
        }


# Global capability registry instance
capability_registry = CapabilityRegistry()
