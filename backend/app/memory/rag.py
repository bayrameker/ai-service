"""
Retrieval Augmented Generation (RAG) System
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import uuid

from app.memory.vector_store import BaseVectorStore, VectorDocument, SearchResult
from app.memory.episodic import EpisodicMemory, Experience
from app.memory.semantic import SemanticMemory, Knowledge
from app.memory.working import WorkingMemory
from app.llm_providers.manager import LLMProviderManager
from app.llm_providers.base import LLMRequest

logger = logging.getLogger(__name__)


class RetrievalStrategy(str, Enum):
    """Retrieval strategies for RAG"""
    SEMANTIC_ONLY = "semantic_only"
    EPISODIC_ONLY = "episodic_only"
    HYBRID = "hybrid"
    CONTEXTUAL = "contextual"
    ADAPTIVE = "adaptive"


class RAGQuery(BaseModel):
    """RAG query model"""
    query: str
    context: Optional[str] = None
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    max_results: int = 5
    include_episodic: bool = True
    include_semantic: bool = True
    include_working: bool = True
    confidence_threshold: float = 0.3
    time_weight: float = 0.1  # Weight for recency in scoring
    importance_weight: float = 0.3  # Weight for importance in scoring


class RetrievedItem(BaseModel):
    """Retrieved item with metadata"""
    id: str
    content: str
    source_type: str  # episodic, semantic, working
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[datetime] = None


class RAGResponse(BaseModel):
    """RAG response model"""
    query: str
    retrieved_items: List[RetrievedItem]
    generated_response: str
    context_used: str
    total_items_found: int
    retrieval_time_ms: int
    generation_time_ms: int
    strategy_used: RetrievalStrategy


class RAGSystem:
    """Retrieval Augmented Generation system"""
    
    def __init__(
        self,
        agent_id: str,
        llm_manager: LLMProviderManager,
        episodic_memory: Optional[EpisodicMemory] = None,
        semantic_memory: Optional[SemanticMemory] = None,
        working_memory: Optional[WorkingMemory] = None
    ):
        self.agent_id = agent_id
        self.llm_manager = llm_manager
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        self.working_memory = working_memory
        
        # Configuration
        self.max_context_length = 4000  # Maximum context length in tokens
        self.default_model = "gpt-3.5-turbo"
        
    async def query(self, rag_query: RAGQuery) -> RAGResponse:
        """Execute RAG query"""
        start_time = datetime.utcnow()
        
        try:
            # Retrieve relevant information
            retrieved_items = await self._retrieve_information(rag_query)
            
            retrieval_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Generate context from retrieved items
            context = await self._build_context(retrieved_items, rag_query)
            
            # Generate response using LLM
            generation_start = datetime.utcnow()
            generated_response = await self._generate_response(rag_query.query, context)
            generation_time = (datetime.utcnow() - generation_start).total_seconds() * 1000
            
            return RAGResponse(
                query=rag_query.query,
                retrieved_items=retrieved_items,
                generated_response=generated_response,
                context_used=context,
                total_items_found=len(retrieved_items),
                retrieval_time_ms=int(retrieval_time),
                generation_time_ms=int(generation_time),
                strategy_used=rag_query.strategy
            )
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            raise
    
    async def _retrieve_information(self, rag_query: RAGQuery) -> List[RetrievedItem]:
        """Retrieve relevant information from all memory systems"""
        try:
            all_items = []
            
            # Retrieve from episodic memory
            if rag_query.include_episodic and self.episodic_memory:
                episodic_items = await self._retrieve_from_episodic(rag_query)
                all_items.extend(episodic_items)
            
            # Retrieve from semantic memory
            if rag_query.include_semantic and self.semantic_memory:
                semantic_items = await self._retrieve_from_semantic(rag_query)
                all_items.extend(semantic_items)
            
            # Retrieve from working memory
            if rag_query.include_working and self.working_memory:
                working_items = await self._retrieve_from_working(rag_query)
                all_items.extend(working_items)
            
            # Score and rank items
            scored_items = await self._score_and_rank_items(all_items, rag_query)
            
            # Filter by confidence threshold
            filtered_items = [
                item for item in scored_items 
                if item.score >= rag_query.confidence_threshold
            ]
            
            # Return top results
            return filtered_items[:rag_query.max_results]
            
        except Exception as e:
            logger.error(f"Failed to retrieve information: {e}")
            return []
    
    async def _retrieve_from_episodic(self, rag_query: RAGQuery) -> List[RetrievedItem]:
        """Retrieve from episodic memory"""
        try:
            experiences = await self.episodic_memory.retrieve_experiences(
                query=rag_query.query,
                limit=rag_query.max_results * 2
            )
            
            items = []
            for exp in experiences:
                content = f"Experience: {exp.title}\n{exp.description}"
                if exp.outcome:
                    content += f"\nOutcome: {exp.outcome}"
                if exp.lessons_learned:
                    content += f"\nLessons: {', '.join(exp.lessons_learned)}"
                
                item = RetrievedItem(
                    id=exp.id,
                    content=content,
                    source_type="episodic",
                    score=exp.importance,
                    metadata={
                        "type": exp.type.value,
                        "success": exp.success,
                        "importance": exp.importance,
                        "tags": exp.tags
                    },
                    timestamp=exp.timestamp
                )
                items.append(item)
            
            return items
            
        except Exception as e:
            logger.error(f"Failed to retrieve from episodic memory: {e}")
            return []
    
    async def _retrieve_from_semantic(self, rag_query: RAGQuery) -> List[RetrievedItem]:
        """Retrieve from semantic memory"""
        try:
            knowledge_items = await self.semantic_memory.retrieve_knowledge(
                query=rag_query.query,
                limit=rag_query.max_results * 2
            )
            
            items = []
            for knowledge in knowledge_items:
                content = f"Knowledge: {knowledge.subject}\n{knowledge.content}"
                if knowledge.description:
                    content += f"\nDescription: {knowledge.description}"
                if knowledge.evidence:
                    content += f"\nEvidence: {', '.join(knowledge.evidence)}"
                
                item = RetrievedItem(
                    id=knowledge.id,
                    content=content,
                    source_type="semantic",
                    score=knowledge.importance,
                    metadata={
                        "type": knowledge.type.value,
                        "confidence": knowledge.confidence.value,
                        "domain": knowledge.domain,
                        "verified": knowledge.verified,
                        "tags": knowledge.tags
                    },
                    timestamp=knowledge.created_at
                )
                items.append(item)
            
            return items
            
        except Exception as e:
            logger.error(f"Failed to retrieve from semantic memory: {e}")
            return []
    
    async def _retrieve_from_working(self, rag_query: RAGQuery) -> List[RetrievedItem]:
        """Retrieve from working memory"""
        try:
            # Search working memory items
            memory_items = await self.working_memory.search_items(
                query=rag_query.query,
                limit=rag_query.max_results
            )
            
            items = []
            for mem_item in memory_items:
                content = f"Context: {mem_item.key}\n{str(mem_item.value)}"
                
                item = RetrievedItem(
                    id=mem_item.id,
                    content=content,
                    source_type="working",
                    score=mem_item.importance,
                    metadata={
                        "type": mem_item.type.value,
                        "priority": mem_item.priority.value,
                        "access_count": mem_item.access_count,
                        "tags": mem_item.tags
                    },
                    timestamp=mem_item.created_at
                )
                items.append(item)
            
            # Also include recent conversation context
            if rag_query.query.lower() in ["conversation", "chat", "recent", "context"]:
                conversation_context = await self.working_memory.get_conversation_context(max_tokens=1000)
                if conversation_context:
                    item = RetrievedItem(
                        id="conversation_context",
                        content=f"Recent Conversation:\n{conversation_context}",
                        source_type="working",
                        score=0.8,
                        metadata={"type": "conversation"},
                        timestamp=datetime.utcnow()
                    )
                    items.append(item)
            
            return items
            
        except Exception as e:
            logger.error(f"Failed to retrieve from working memory: {e}")
            return []
    
    async def _score_and_rank_items(self, items: List[RetrievedItem], rag_query: RAGQuery) -> List[RetrievedItem]:
        """Score and rank retrieved items"""
        try:
            now = datetime.utcnow()
            
            for item in items:
                # Base score from retrieval
                base_score = item.score
                
                # Time decay factor (more recent items get higher scores)
                time_factor = 1.0
                if item.timestamp:
                    days_old = (now - item.timestamp).days
                    time_factor = max(0.1, 1.0 - (days_old * 0.01))  # Decay over time
                
                # Source type weighting
                source_weight = 1.0
                if item.source_type == "working":
                    source_weight = 1.2  # Working memory is more relevant for current context
                elif item.source_type == "episodic":
                    source_weight = 1.1  # Experiences are valuable
                elif item.source_type == "semantic":
                    source_weight = 1.0  # Base weight for knowledge
                
                # Metadata-based adjustments
                metadata_bonus = 0.0
                if item.metadata.get("verified"):
                    metadata_bonus += 0.1
                if item.metadata.get("success"):
                    metadata_bonus += 0.1
                if item.metadata.get("priority") == "high":
                    metadata_bonus += 0.1
                
                # Calculate final score
                final_score = (
                    base_score * source_weight +
                    time_factor * rag_query.time_weight +
                    metadata_bonus
                )
                
                item.score = min(1.0, final_score)  # Cap at 1.0
            
            # Sort by score (descending)
            items.sort(key=lambda x: x.score, reverse=True)
            
            return items
            
        except Exception as e:
            logger.error(f"Failed to score and rank items: {e}")
            return items
    
    async def _build_context(self, items: List[RetrievedItem], rag_query: RAGQuery) -> str:
        """Build context string from retrieved items"""
        try:
            context_parts = []
            total_length = 0
            max_length = self.max_context_length * 4  # Rough character estimate
            
            # Add query context if provided
            if rag_query.context:
                context_parts.append(f"Query Context: {rag_query.context}")
                total_length += len(rag_query.context)
            
            # Add retrieved items
            for item in items:
                item_text = f"[{item.source_type.upper()}] {item.content}"
                
                if total_length + len(item_text) > max_length:
                    break
                
                context_parts.append(item_text)
                total_length += len(item_text)
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Failed to build context: {e}")
            return ""
    
    async def _generate_response(self, query: str, context: str) -> str:
        """Generate response using LLM with retrieved context"""
        try:
            # Prepare system prompt
            system_prompt = """You are an AI assistant with access to relevant information from memory. 
Use the provided context to answer the user's question accurately and comprehensively. 
If the context doesn't contain enough information to fully answer the question, say so clearly.
Always cite which type of memory (episodic, semantic, or working) your information comes from when relevant."""
            
            # Prepare user prompt
            user_prompt = f"""Context from memory:
{context}

Question: {query}

Please provide a comprehensive answer based on the available context."""
            
            # Create LLM request
            llm_request = LLMRequest(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.default_model,
                max_tokens=1000,
                temperature=0.7
            )
            
            # Generate response
            response = await self.llm_manager.generate(llm_request)
            
            return response.content if response else "I apologize, but I couldn't generate a response at this time."
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return "I apologize, but I encountered an error while generating a response."
    
    async def ask_with_memory(self, question: str, context: Optional[str] = None) -> str:
        """Simple interface for asking questions with memory retrieval"""
        try:
            rag_query = RAGQuery(
                query=question,
                context=context,
                strategy=RetrievalStrategy.ADAPTIVE,
                max_results=5
            )
            
            response = await self.query(rag_query)
            return response.generated_response
            
        except Exception as e:
            logger.error(f"Failed to ask with memory: {e}")
            return "I apologize, but I couldn't process your question at this time."
    
    async def get_relevant_context(self, query: str, max_items: int = 3) -> List[RetrievedItem]:
        """Get relevant context items without generating a response"""
        try:
            rag_query = RAGQuery(
                query=query,
                max_results=max_items,
                strategy=RetrievalStrategy.HYBRID
            )
            
            return await self._retrieve_information(rag_query)
            
        except Exception as e:
            logger.error(f"Failed to get relevant context: {e}")
            return []
    
    async def update_memory_from_interaction(self, query: str, response: str, feedback: Optional[str] = None) -> None:
        """Update memory systems based on interaction"""
        try:
            # Store in working memory as recent interaction
            if self.working_memory:
                await self.working_memory.add_conversation_turn("user", query)
                await self.working_memory.add_conversation_turn("assistant", response)
            
            # If feedback is provided, create an experience
            if feedback and self.episodic_memory:
                from app.memory.episodic import Experience, ExperienceType
                
                experience = Experience(
                    agent_id=self.agent_id,
                    type=ExperienceType.INTERACTION,
                    title=f"Q&A Interaction",
                    description=f"Question: {query}\nResponse: {response}",
                    outcome=feedback,
                    success=True if "good" in feedback.lower() or "correct" in feedback.lower() else None,
                    context={"query": query, "response": response},
                    importance=0.6
                )
                
                await self.episodic_memory.store_experience(experience)
            
        except Exception as e:
            logger.error(f"Failed to update memory from interaction: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        try:
            stats = {
                "agent_id": self.agent_id,
                "max_context_length": self.max_context_length,
                "default_model": self.default_model,
                "memory_systems": {
                    "episodic": self.episodic_memory is not None,
                    "semantic": self.semantic_memory is not None,
                    "working": self.working_memory is not None
                }
            }
            
            # Add memory system stats if available
            if self.episodic_memory:
                stats["episodic_stats"] = await self.episodic_memory.get_stats()
            
            if self.semantic_memory:
                stats["semantic_stats"] = await self.semantic_memory.get_stats()
            
            if self.working_memory:
                stats["working_stats"] = await self.working_memory.get_stats()
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get RAG stats: {e}")
            return {"error": str(e)}
