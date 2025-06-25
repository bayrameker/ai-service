"""
Self-Learning Algorithms for AI Agents
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum
import uuid
import json
import numpy as np
from collections import defaultdict

from app.memory.episodic import EpisodicMemory, Experience, ExperienceType
from app.memory.semantic import SemanticMemory, Knowledge, KnowledgeType, ConfidenceLevel
from app.memory.working import WorkingMemory
from app.llm_providers.manager import LLMProviderManager
from app.llm_providers.base import LLMRequest

logger = logging.getLogger(__name__)


class LearningType(str, Enum):
    """Types of learning"""
    PATTERN_RECOGNITION = "pattern_recognition"
    OUTCOME_PREDICTION = "outcome_prediction"
    STRATEGY_OPTIMIZATION = "strategy_optimization"
    ERROR_CORRECTION = "error_correction"
    KNOWLEDGE_SYNTHESIS = "knowledge_synthesis"
    BEHAVIORAL_ADAPTATION = "behavioral_adaptation"


class LearningInsight(BaseModel):
    """Learning insight derived from experiences"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    type: LearningType
    title: str
    description: str
    confidence: float = 0.5  # 0.0 to 1.0
    evidence: List[str] = Field(default_factory=list)
    supporting_experiences: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_validated: Optional[datetime] = None
    validation_count: int = 0
    success_rate: float = 0.0
    applicability_score: float = 0.5
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LearningPattern(BaseModel):
    """Identified pattern from experiences"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pattern_type: str
    conditions: Dict[str, Any]
    outcomes: Dict[str, Any]
    frequency: int = 1
    success_rate: float = 0.0
    confidence: float = 0.5
    examples: List[str] = Field(default_factory=list)


class SelfLearningSystem:
    """Self-learning system for AI agents"""
    
    def __init__(
        self,
        agent_id: str,
        episodic_memory: EpisodicMemory,
        semantic_memory: SemanticMemory,
        working_memory: WorkingMemory,
        llm_manager: LLMProviderManager
    ):
        self.agent_id = agent_id
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        self.working_memory = working_memory
        self.llm_manager = llm_manager
        
        # Learning state
        self.learning_insights: Dict[str, LearningInsight] = {}
        self.identified_patterns: Dict[str, LearningPattern] = {}
        self.learning_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.min_experiences_for_pattern = 3
        self.confidence_threshold = 0.6
        self.learning_interval_hours = 24
        self.max_insights = 1000
        
        # Learning task
        self.learning_task: Optional[asyncio.Task] = None
        
    async def initialize(self) -> None:
        """Initialize the self-learning system"""
        try:
            # Start periodic learning
            self.learning_task = asyncio.create_task(self._periodic_learning())
            
            logger.info(f"Self-learning system initialized for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize self-learning system: {e}")
            raise
    
    async def learn_from_experience(self, experience: Experience) -> List[LearningInsight]:
        """Learn from a single experience"""
        try:
            insights = []
            
            # Analyze the experience for learning opportunities
            if experience.success is not None:
                # Learn from success/failure patterns
                pattern_insights = await self._analyze_success_patterns(experience)
                insights.extend(pattern_insights)
            
            # Learn from errors
            if experience.type == ExperienceType.ERROR:
                error_insights = await self._analyze_error_patterns(experience)
                insights.extend(error_insights)
            
            # Learn from task execution
            if experience.type == ExperienceType.TASK_EXECUTION:
                task_insights = await self._analyze_task_patterns(experience)
                insights.extend(task_insights)
            
            # Store insights
            for insight in insights:
                await self._store_learning_insight(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to learn from experience: {e}")
            return []
    
    async def learn_from_experiences_batch(self, experiences: List[Experience]) -> List[LearningInsight]:
        """Learn from a batch of experiences"""
        try:
            all_insights = []
            
            # Group experiences by type for pattern analysis
            experiences_by_type = defaultdict(list)
            for exp in experiences:
                experiences_by_type[exp.type].append(exp)
            
            # Analyze patterns within each type
            for exp_type, type_experiences in experiences_by_type.items():
                if len(type_experiences) >= self.min_experiences_for_pattern:
                    patterns = await self._identify_patterns(type_experiences)
                    
                    for pattern in patterns:
                        insight = await self._pattern_to_insight(pattern, type_experiences)
                        if insight:
                            all_insights.append(insight)
                            await self._store_learning_insight(insight)
            
            # Cross-type pattern analysis
            cross_patterns = await self._analyze_cross_type_patterns(experiences)
            all_insights.extend(cross_patterns)
            
            return all_insights
            
        except Exception as e:
            logger.error(f"Failed to learn from experiences batch: {e}")
            return []
    
    async def _analyze_success_patterns(self, experience: Experience) -> List[LearningInsight]:
        """Analyze patterns related to success/failure"""
        try:
            insights = []
            
            # Get similar experiences
            similar_experiences = await self.episodic_memory.get_similar_experiences(experience, limit=10)
            
            # Analyze success factors
            if experience.success:
                success_factors = await self._identify_success_factors(experience, similar_experiences)
                
                if success_factors:
                    insight = LearningInsight(
                        agent_id=self.agent_id,
                        type=LearningType.OUTCOME_PREDICTION,
                        title=f"Success factors for {experience.type.value}",
                        description=f"Identified factors that contribute to success: {', '.join(success_factors)}",
                        confidence=0.7,
                        evidence=success_factors,
                        supporting_experiences=[exp.id for exp in similar_experiences if exp.success],
                        tags=["success", "pattern", experience.type.value]
                    )
                    insights.append(insight)
            
            # Analyze failure patterns
            elif experience.success is False:
                failure_patterns = await self._identify_failure_patterns(experience, similar_experiences)
                
                if failure_patterns:
                    insight = LearningInsight(
                        agent_id=self.agent_id,
                        type=LearningType.ERROR_CORRECTION,
                        title=f"Failure patterns for {experience.type.value}",
                        description=f"Identified patterns that lead to failure: {', '.join(failure_patterns)}",
                        confidence=0.6,
                        evidence=failure_patterns,
                        supporting_experiences=[exp.id for exp in similar_experiences if exp.success is False],
                        tags=["failure", "pattern", experience.type.value]
                    )
                    insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to analyze success patterns: {e}")
            return []
    
    async def _analyze_error_patterns(self, experience: Experience) -> List[LearningInsight]:
        """Analyze error patterns for learning"""
        try:
            insights = []
            
            # Get recent error experiences
            error_experiences = await self.episodic_memory.get_experiences_by_type(
                ExperienceType.ERROR, limit=20
            )
            
            # Look for common error patterns
            error_contexts = [exp.context for exp in error_experiences if exp.context]
            
            if len(error_contexts) >= 3:
                common_patterns = await self._find_common_patterns(error_contexts)
                
                for pattern in common_patterns:
                    insight = LearningInsight(
                        agent_id=self.agent_id,
                        type=LearningType.ERROR_CORRECTION,
                        title=f"Common error pattern: {pattern}",
                        description=f"This pattern appears frequently in error situations and should be avoided",
                        confidence=0.8,
                        evidence=[pattern],
                        supporting_experiences=[exp.id for exp in error_experiences],
                        tags=["error", "pattern", "avoidance"]
                    )
                    insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to analyze error patterns: {e}")
            return []
    
    async def _analyze_task_patterns(self, experience: Experience) -> List[LearningInsight]:
        """Analyze task execution patterns"""
        try:
            insights = []
            
            # Get task execution experiences
            task_experiences = await self.episodic_memory.get_experiences_by_type(
                ExperienceType.TASK_EXECUTION, limit=50
            )
            
            # Analyze duration patterns
            if experience.duration_seconds:
                duration_insight = await self._analyze_duration_patterns(experience, task_experiences)
                if duration_insight:
                    insights.append(duration_insight)
            
            # Analyze strategy effectiveness
            strategy_insight = await self._analyze_strategy_effectiveness(experience, task_experiences)
            if strategy_insight:
                insights.append(strategy_insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to analyze task patterns: {e}")
            return []
    
    async def _identify_patterns(self, experiences: List[Experience]) -> List[LearningPattern]:
        """Identify patterns from a group of experiences"""
        try:
            patterns = []
            
            # Group by context similarity
            context_groups = await self._group_by_context_similarity(experiences)
            
            for group in context_groups:
                if len(group) >= self.min_experiences_for_pattern:
                    pattern = await self._extract_pattern_from_group(group)
                    if pattern:
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to identify patterns: {e}")
            return []
    
    async def _group_by_context_similarity(self, experiences: List[Experience]) -> List[List[Experience]]:
        """Group experiences by context similarity"""
        try:
            # Simple grouping by common context keys
            groups = defaultdict(list)
            
            for exp in experiences:
                if exp.context:
                    # Create a signature from context keys
                    context_keys = sorted(exp.context.keys())
                    signature = "|".join(context_keys)
                    groups[signature].append(exp)
            
            return [group for group in groups.values() if len(group) >= 2]
            
        except Exception as e:
            logger.error(f"Failed to group by context similarity: {e}")
            return []
    
    async def _extract_pattern_from_group(self, experiences: List[Experience]) -> Optional[LearningPattern]:
        """Extract a pattern from a group of similar experiences"""
        try:
            if not experiences:
                return None
            
            # Analyze common conditions
            common_conditions = {}
            for exp in experiences:
                if exp.context:
                    for key, value in exp.context.items():
                        if key not in common_conditions:
                            common_conditions[key] = []
                        common_conditions[key].append(value)
            
            # Find truly common conditions (appear in most experiences)
            threshold = len(experiences) * 0.7  # 70% threshold
            filtered_conditions = {}
            
            for key, values in common_conditions.items():
                unique_values = list(set(str(v) for v in values))
                if len(values) >= threshold:
                    filtered_conditions[key] = unique_values[0] if len(unique_values) == 1 else unique_values
            
            # Analyze outcomes
            outcomes = {}
            success_count = sum(1 for exp in experiences if exp.success)
            outcomes["success_rate"] = success_count / len(experiences)
            
            # Create pattern
            pattern = LearningPattern(
                pattern_type=experiences[0].type.value,
                conditions=filtered_conditions,
                outcomes=outcomes,
                frequency=len(experiences),
                success_rate=outcomes["success_rate"],
                confidence=min(0.9, len(experiences) / 10),  # Higher confidence with more examples
                examples=[exp.id for exp in experiences[:5]]  # Store first 5 as examples
            )
            
            return pattern
            
        except Exception as e:
            logger.error(f"Failed to extract pattern from group: {e}")
            return None
    
    async def _pattern_to_insight(self, pattern: LearningPattern, experiences: List[Experience]) -> Optional[LearningInsight]:
        """Convert a pattern to a learning insight"""
        try:
            # Determine insight type based on pattern
            if pattern.success_rate > 0.8:
                insight_type = LearningType.STRATEGY_OPTIMIZATION
                title = f"Effective strategy for {pattern.pattern_type}"
                description = f"This approach has a {pattern.success_rate:.1%} success rate"
            elif pattern.success_rate < 0.3:
                insight_type = LearningType.ERROR_CORRECTION
                title = f"Problematic pattern in {pattern.pattern_type}"
                description = f"This pattern leads to failure {1-pattern.success_rate:.1%} of the time"
            else:
                insight_type = LearningType.PATTERN_RECOGNITION
                title = f"Pattern identified in {pattern.pattern_type}"
                description = f"Recurring pattern with {pattern.success_rate:.1%} success rate"
            
            insight = LearningInsight(
                agent_id=self.agent_id,
                type=insight_type,
                title=title,
                description=description,
                confidence=pattern.confidence,
                evidence=[f"Pattern occurs {pattern.frequency} times"],
                supporting_experiences=pattern.examples,
                tags=["pattern", pattern.pattern_type],
                metadata={"pattern_id": pattern.id, "conditions": pattern.conditions}
            )
            
            return insight
            
        except Exception as e:
            logger.error(f"Failed to convert pattern to insight: {e}")
            return None
    
    async def _analyze_cross_type_patterns(self, experiences: List[Experience]) -> List[LearningInsight]:
        """Analyze patterns across different experience types"""
        try:
            insights = []
            
            # Look for sequences (e.g., error followed by success)
            sequence_patterns = await self._identify_sequence_patterns(experiences)
            
            for pattern in sequence_patterns:
                insight = LearningInsight(
                    agent_id=self.agent_id,
                    type=LearningType.BEHAVIORAL_ADAPTATION,
                    title=f"Sequence pattern: {pattern['description']}",
                    description=f"Identified behavioral sequence with {pattern['frequency']} occurrences",
                    confidence=pattern['confidence'],
                    evidence=[pattern['description']],
                    tags=["sequence", "behavior"]
                )
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to analyze cross-type patterns: {e}")
            return []
    
    async def _identify_sequence_patterns(self, experiences: List[Experience]) -> List[Dict[str, Any]]:
        """Identify temporal sequence patterns"""
        try:
            patterns = []
            
            # Sort experiences by timestamp
            sorted_experiences = sorted(experiences, key=lambda x: x.timestamp)
            
            # Look for common sequences
            sequences = defaultdict(int)
            
            for i in range(len(sorted_experiences) - 1):
                current = sorted_experiences[i]
                next_exp = sorted_experiences[i + 1]
                
                # Check if they're close in time (within 1 hour)
                time_diff = (next_exp.timestamp - current.timestamp).total_seconds()
                if time_diff <= 3600:  # 1 hour
                    sequence_key = f"{current.type.value} -> {next_exp.type.value}"
                    sequences[sequence_key] += 1
            
            # Convert to patterns
            for sequence, frequency in sequences.items():
                if frequency >= 3:  # At least 3 occurrences
                    patterns.append({
                        "description": sequence,
                        "frequency": frequency,
                        "confidence": min(0.9, frequency / 10)
                    })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to identify sequence patterns: {e}")
            return []
    
    async def _store_learning_insight(self, insight: LearningInsight) -> None:
        """Store a learning insight"""
        try:
            self.learning_insights[insight.id] = insight
            
            # Convert insight to knowledge for semantic memory
            knowledge = Knowledge(
                agent_id=self.agent_id,
                type=KnowledgeType.PATTERN,
                subject=insight.title,
                content=insight.description,
                confidence=ConfidenceLevel.MEDIUM if insight.confidence > 0.5 else ConfidenceLevel.LOW,
                source="self_learning",
                evidence=insight.evidence,
                tags=insight.tags + ["learned", "insight"],
                domain="self_improvement",
                importance=insight.confidence,
                verified=False
            )
            
            await self.semantic_memory.store_knowledge(knowledge)
            
            logger.info(f"Stored learning insight: {insight.title}")
            
        except Exception as e:
            logger.error(f"Failed to store learning insight: {e}")
    
    async def _periodic_learning(self) -> None:
        """Periodic learning from accumulated experiences"""
        while True:
            try:
                await asyncio.sleep(self.learning_interval_hours * 3600)
                
                # Get recent experiences
                recent_experiences = await self.episodic_memory.get_recent_experiences(
                    hours=self.learning_interval_hours * 2,
                    limit=100
                )
                
                if len(recent_experiences) >= self.min_experiences_for_pattern:
                    insights = await self.learn_from_experiences_batch(recent_experiences)
                    
                    if insights:
                        logger.info(f"Periodic learning generated {len(insights)} new insights")
                
            except Exception as e:
                logger.error(f"Error in periodic learning: {e}")
    
    async def get_learning_insights(self, limit: Optional[int] = None) -> List[LearningInsight]:
        """Get learning insights"""
        insights = list(self.learning_insights.values())
        insights.sort(key=lambda x: x.confidence, reverse=True)
        
        if limit:
            return insights[:limit]
        return insights
    
    async def get_insights_by_type(self, learning_type: LearningType) -> List[LearningInsight]:
        """Get insights by learning type"""
        return [insight for insight in self.learning_insights.values() if insight.type == learning_type]
    
    async def validate_insight(self, insight_id: str, success: bool) -> bool:
        """Validate a learning insight based on real-world application"""
        try:
            if insight_id not in self.learning_insights:
                return False
            
            insight = self.learning_insights[insight_id]
            insight.validation_count += 1
            insight.last_validated = datetime.utcnow()
            
            # Update success rate
            if insight.validation_count == 1:
                insight.success_rate = 1.0 if success else 0.0
            else:
                # Running average
                current_successes = insight.success_rate * (insight.validation_count - 1)
                new_successes = current_successes + (1 if success else 0)
                insight.success_rate = new_successes / insight.validation_count
            
            # Update confidence based on validation
            if insight.success_rate > 0.8:
                insight.confidence = min(0.95, insight.confidence + 0.1)
            elif insight.success_rate < 0.3:
                insight.confidence = max(0.1, insight.confidence - 0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate insight: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get learning system statistics"""
        try:
            total_insights = len(self.learning_insights)
            
            # Count by type
            type_counts = defaultdict(int)
            confidence_sum = 0
            validated_count = 0
            
            for insight in self.learning_insights.values():
                type_counts[insight.type.value] += 1
                confidence_sum += insight.confidence
                if insight.validation_count > 0:
                    validated_count += 1
            
            avg_confidence = confidence_sum / total_insights if total_insights > 0 else 0
            validation_rate = validated_count / total_insights if total_insights > 0 else 0
            
            return {
                "agent_id": self.agent_id,
                "total_insights": total_insights,
                "type_distribution": dict(type_counts),
                "average_confidence": avg_confidence,
                "validation_rate": validation_rate,
                "learning_interval_hours": self.learning_interval_hours,
                "patterns_identified": len(self.identified_patterns)
            }
            
        except Exception as e:
            logger.error(f"Failed to get learning stats: {e}")
            return {"error": str(e)}
    
    async def shutdown(self) -> None:
        """Shutdown the learning system"""
        if self.learning_task:
            self.learning_task.cancel()
            try:
                await self.learning_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Self-learning system shutdown complete")
