"""
Semantic Memory System - Stores knowledge, facts, and concepts
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum
import uuid
import json

from app.memory.vector_store import BaseVectorStore, VectorDocument, create_vector_store
from app.llm_providers.manager import LLMProviderManager
from app.core.database import get_redis

logger = logging.getLogger(__name__)


class KnowledgeType(str, Enum):
    """Types of knowledge"""
    FACT = "fact"
    CONCEPT = "concept"
    PROCEDURE = "procedure"
    RULE = "rule"
    RELATIONSHIP = "relationship"
    DEFINITION = "definition"
    EXAMPLE = "example"
    PATTERN = "pattern"


class ConfidenceLevel(str, Enum):
    """Confidence levels for knowledge"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class Knowledge(BaseModel):
    """Individual piece of knowledge"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    type: KnowledgeType
    subject: str
    content: str
    description: Optional[str] = None
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    source: Optional[str] = None
    evidence: List[str] = Field(default_factory=list)
    related_knowledge: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    domain: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = 0
    importance: float = 0.5  # 0.0 to 1.0
    verified: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConceptGraph(BaseModel):
    """Graph representation of concepts and their relationships"""
    nodes: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    edges: List[Tuple[str, str, str]] = Field(default_factory=list)  # (from, to, relationship_type)


class SemanticMemory:
    """Semantic memory system for storing and organizing knowledge"""
    
    def __init__(self, agent_id: str, vector_store: BaseVectorStore, llm_manager: LLMProviderManager):
        self.agent_id = agent_id
        self.vector_store = vector_store
        self.llm_manager = llm_manager
        self.knowledge_base: Dict[str, Knowledge] = {}
        self.concept_graph = ConceptGraph()
        self.redis_client = None
        
        # Memory configuration
        self.max_knowledge_items = 50000
        self.confidence_threshold = ConfidenceLevel.LOW
        self.retention_days = 730  # 2 years
        
    async def initialize(self) -> None:
        """Initialize semantic memory"""
        try:
            # Initialize vector store
            if not self.vector_store.initialized:
                await self.vector_store.initialize()
            
            # Get Redis client
            self.redis_client = await get_redis()
            
            # Load recent knowledge from Redis
            await self._load_recent_knowledge()
            
            # Build concept graph
            await self._build_concept_graph()
            
            logger.info(f"Semantic memory initialized for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize semantic memory: {e}")
            raise
    
    async def store_knowledge(self, knowledge: Knowledge) -> str:
        """Store new knowledge"""
        try:
            # Set agent ID
            knowledge.agent_id = self.agent_id
            
            # Check for existing similar knowledge
            existing = await self._find_similar_knowledge(knowledge)
            if existing:
                # Update existing knowledge instead of creating duplicate
                return await self._merge_knowledge(existing, knowledge)
            
            # Generate embedding for the knowledge
            knowledge_text = self._knowledge_to_text(knowledge)
            embedding = await self._get_embedding(knowledge_text)
            
            # Create vector document
            vector_doc = VectorDocument(
                id=knowledge.id,
                content=knowledge_text,
                embedding=embedding,
                metadata={
                    "agent_id": self.agent_id,
                    "type": knowledge.type.value,
                    "subject": knowledge.subject,
                    "confidence": knowledge.confidence.value,
                    "domain": knowledge.domain,
                    "importance": knowledge.importance,
                    "verified": knowledge.verified,
                    "tags": knowledge.tags,
                    "created_at": knowledge.created_at.isoformat(),
                    **knowledge.metadata
                }
            )
            
            # Store in vector database
            await self.vector_store.add_documents([vector_doc])
            
            # Store in local cache
            self.knowledge_base[knowledge.id] = knowledge
            
            # Store in Redis for quick access
            await self._store_knowledge_in_redis(knowledge)
            
            # Update concept graph
            await self._update_concept_graph(knowledge)
            
            # Update related knowledge
            await self._update_related_knowledge(knowledge)
            
            logger.info(f"Stored knowledge {knowledge.id} for agent {self.agent_id}")
            return knowledge.id
            
        except Exception as e:
            logger.error(f"Failed to store knowledge: {e}")
            raise
    
    async def retrieve_knowledge(
        self,
        query: str,
        limit: int = 10,
        knowledge_type: Optional[KnowledgeType] = None,
        domain: Optional[str] = None,
        min_confidence: Optional[ConfidenceLevel] = None,
        verified_only: bool = False
    ) -> List[Knowledge]:
        """Retrieve relevant knowledge based on query"""
        try:
            # Get query embedding
            query_embedding = await self._get_embedding(query)
            
            # Prepare metadata filter
            metadata_filter = {"agent_id": self.agent_id}
            if knowledge_type:
                metadata_filter["type"] = knowledge_type.value
            if domain:
                metadata_filter["domain"] = domain
            if verified_only:
                metadata_filter["verified"] = True
            
            # Search vector store
            search_results = await self.vector_store.search(
                query_embedding=query_embedding,
                k=limit * 2,  # Get more results for filtering
                filter_metadata=metadata_filter
            )
            
            # Filter and convert results
            knowledge_items = []
            confidence_levels = {
                ConfidenceLevel.VERY_LOW: 1,
                ConfidenceLevel.LOW: 2,
                ConfidenceLevel.MEDIUM: 3,
                ConfidenceLevel.HIGH: 4,
                ConfidenceLevel.VERY_HIGH: 5
            }
            
            min_conf_value = confidence_levels.get(min_confidence, 0) if min_confidence else 0
            
            for result in search_results:
                knowledge_id = result.document.id
                
                # Get full knowledge from cache or reconstruct
                knowledge = self.knowledge_base.get(knowledge_id)
                if not knowledge:
                    knowledge = await self._reconstruct_knowledge_from_vector_doc(result.document)
                
                if not knowledge:
                    continue
                
                # Apply confidence filter
                if min_confidence and confidence_levels.get(knowledge.confidence, 0) < min_conf_value:
                    continue
                
                # Update access statistics
                knowledge.last_accessed = datetime.utcnow()
                knowledge.access_count += 1
                
                knowledge_items.append(knowledge)
                
                if len(knowledge_items) >= limit:
                    break
            
            return knowledge_items
            
        except Exception as e:
            logger.error(f"Failed to retrieve knowledge: {e}")
            return []
    
    async def get_knowledge(self, knowledge_id: str) -> Optional[Knowledge]:
        """Get specific knowledge by ID"""
        # Try local cache first
        if knowledge_id in self.knowledge_base:
            knowledge = self.knowledge_base[knowledge_id]
            knowledge.last_accessed = datetime.utcnow()
            knowledge.access_count += 1
            return knowledge
        
        # Try Redis
        knowledge = await self._get_knowledge_from_redis(knowledge_id)
        if knowledge:
            self.knowledge_base[knowledge_id] = knowledge
            knowledge.last_accessed = datetime.utcnow()
            knowledge.access_count += 1
            return knowledge
        
        # Try vector store
        vector_doc = await self.vector_store.get_document(knowledge_id)
        if vector_doc:
            knowledge = await self._reconstruct_knowledge_from_vector_doc(vector_doc)
            if knowledge:
                knowledge.last_accessed = datetime.utcnow()
                knowledge.access_count += 1
            return knowledge
        
        return None
    
    async def update_knowledge(self, knowledge_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing knowledge"""
        try:
            knowledge = await self.get_knowledge(knowledge_id)
            if not knowledge:
                return False
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(knowledge, key):
                    setattr(knowledge, key, value)
            
            knowledge.updated_at = datetime.utcnow()
            
            # Re-store the knowledge
            await self.store_knowledge(knowledge)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update knowledge {knowledge_id}: {e}")
            return False
    
    async def verify_knowledge(self, knowledge_id: str, verified: bool = True) -> bool:
        """Mark knowledge as verified or unverified"""
        return await self.update_knowledge(knowledge_id, {"verified": verified})
    
    async def get_knowledge_by_subject(self, subject: str, limit: int = 10) -> List[Knowledge]:
        """Get knowledge about a specific subject"""
        return await self.retrieve_knowledge(f"knowledge about {subject}", limit=limit)
    
    async def get_knowledge_by_domain(self, domain: str, limit: int = 10) -> List[Knowledge]:
        """Get knowledge from a specific domain"""
        return await self.retrieve_knowledge(
            query=f"knowledge in {domain} domain",
            limit=limit,
            domain=domain
        )
    
    async def get_related_knowledge(self, knowledge_id: str, limit: int = 5) -> List[Knowledge]:
        """Get knowledge related to a specific knowledge item"""
        knowledge = await self.get_knowledge(knowledge_id)
        if not knowledge:
            return []
        
        # Get explicitly related knowledge
        related = []
        for related_id in knowledge.related_knowledge:
            related_knowledge = await self.get_knowledge(related_id)
            if related_knowledge:
                related.append(related_knowledge)
        
        # Get semantically similar knowledge
        if len(related) < limit:
            knowledge_text = self._knowledge_to_text(knowledge)
            similar = await self.retrieve_knowledge(knowledge_text, limit=limit - len(related))
            
            # Filter out the original knowledge and already related ones
            for sim_knowledge in similar:
                if sim_knowledge.id != knowledge_id and sim_knowledge.id not in [r.id for r in related]:
                    related.append(sim_knowledge)
                    if len(related) >= limit:
                        break
        
        return related[:limit]
    
    async def get_concept_graph(self) -> ConceptGraph:
        """Get the concept graph"""
        return self.concept_graph
    
    async def find_knowledge_gaps(self, domain: str) -> List[str]:
        """Identify potential knowledge gaps in a domain"""
        try:
            # Get all knowledge in the domain
            domain_knowledge = await self.get_knowledge_by_domain(domain, limit=100)
            
            # Analyze for gaps (simplified implementation)
            subjects = set()
            for knowledge in domain_knowledge:
                subjects.add(knowledge.subject)
            
            # This is a simplified gap analysis
            # In a real system, you might use more sophisticated techniques
            gaps = []
            
            # Look for incomplete concept coverage
            if len(subjects) < 5:  # Arbitrary threshold
                gaps.append(f"Limited coverage in {domain} domain")
            
            # Look for unverified knowledge
            unverified_count = sum(1 for k in domain_knowledge if not k.verified)
            if unverified_count > len(domain_knowledge) * 0.5:
                gaps.append(f"Many unverified facts in {domain}")
            
            return gaps
            
        except Exception as e:
            logger.error(f"Failed to find knowledge gaps: {e}")
            return []
    
    def _knowledge_to_text(self, knowledge: Knowledge) -> str:
        """Convert knowledge to text for embedding"""
        text_parts = [
            f"Subject: {knowledge.subject}",
            f"Type: {knowledge.type.value}",
            f"Content: {knowledge.content}"
        ]
        
        if knowledge.description:
            text_parts.append(f"Description: {knowledge.description}")
        
        if knowledge.domain:
            text_parts.append(f"Domain: {knowledge.domain}")
        
        if knowledge.tags:
            text_parts.append(f"Tags: {', '.join(knowledge.tags)}")
        
        if knowledge.evidence:
            text_parts.append(f"Evidence: {', '.join(knowledge.evidence)}")
        
        return "\n".join(text_parts)
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text"""
        try:
            return await self.llm_manager.get_embedding(text)
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 1536
    
    async def _find_similar_knowledge(self, knowledge: Knowledge) -> Optional[Knowledge]:
        """Find existing similar knowledge"""
        try:
            similar_items = await self.retrieve_knowledge(
                query=self._knowledge_to_text(knowledge),
                limit=5
            )
            
            # Check for exact or very similar matches
            for item in similar_items:
                if (item.subject.lower() == knowledge.subject.lower() and 
                    item.type == knowledge.type and
                    item.content.lower() == knowledge.content.lower()):
                    return item
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to find similar knowledge: {e}")
            return None
    
    async def _merge_knowledge(self, existing: Knowledge, new: Knowledge) -> str:
        """Merge new knowledge with existing knowledge"""
        try:
            # Update confidence if new knowledge has higher confidence
            confidence_levels = {
                ConfidenceLevel.VERY_LOW: 1,
                ConfidenceLevel.LOW: 2,
                ConfidenceLevel.MEDIUM: 3,
                ConfidenceLevel.HIGH: 4,
                ConfidenceLevel.VERY_HIGH: 5
            }
            
            if confidence_levels.get(new.confidence, 0) > confidence_levels.get(existing.confidence, 0):
                existing.confidence = new.confidence
            
            # Merge evidence
            for evidence in new.evidence:
                if evidence not in existing.evidence:
                    existing.evidence.append(evidence)
            
            # Merge tags
            for tag in new.tags:
                if tag not in existing.tags:
                    existing.tags.append(tag)
            
            # Update source if new source is provided
            if new.source and not existing.source:
                existing.source = new.source
            
            # Update verification status
            if new.verified and not existing.verified:
                existing.verified = True
            
            existing.updated_at = datetime.utcnow()
            
            # Re-store the merged knowledge
            await self.store_knowledge(existing)
            
            return existing.id
            
        except Exception as e:
            logger.error(f"Failed to merge knowledge: {e}")
            return existing.id
    
    async def _store_knowledge_in_redis(self, knowledge: Knowledge) -> None:
        """Store knowledge in Redis for quick access"""
        if not self.redis_client:
            return
        
        try:
            key = f"knowledge:{self.agent_id}:{knowledge.id}"
            value = json.dumps(knowledge.dict(), default=str)
            await self.redis_client.set(key, value, ex=86400 * 30)  # 30 days
        except Exception as e:
            logger.error(f"Failed to store knowledge in Redis: {e}")
    
    async def _get_knowledge_from_redis(self, knowledge_id: str) -> Optional[Knowledge]:
        """Get knowledge from Redis"""
        if not self.redis_client:
            return None
        
        try:
            key = f"knowledge:{self.agent_id}:{knowledge_id}"
            value = await self.redis_client.get(key)
            if value:
                data = json.loads(value)
                return Knowledge(**data)
        except Exception as e:
            logger.error(f"Failed to get knowledge from Redis: {e}")
        
        return None
    
    async def _load_recent_knowledge(self) -> None:
        """Load recent knowledge from Redis"""
        if not self.redis_client:
            return
        
        try:
            pattern = f"knowledge:{self.agent_id}:*"
            keys = await self.redis_client.keys(pattern)
            
            for key in keys[:1000]:  # Limit to 1000 recent items
                value = await self.redis_client.get(key)
                if value:
                    data = json.loads(value)
                    knowledge = Knowledge(**data)
                    self.knowledge_base[knowledge.id] = knowledge
            
            logger.info(f"Loaded {len(self.knowledge_base)} knowledge items from Redis")
            
        except Exception as e:
            logger.error(f"Failed to load recent knowledge: {e}")
    
    async def _reconstruct_knowledge_from_vector_doc(self, vector_doc: VectorDocument) -> Optional[Knowledge]:
        """Reconstruct knowledge from vector document metadata"""
        try:
            metadata = vector_doc.metadata
            
            knowledge = Knowledge(
                id=vector_doc.id,
                agent_id=metadata.get("agent_id", self.agent_id),
                type=KnowledgeType(metadata.get("type", "fact")),
                subject=metadata.get("subject", "Unknown"),
                content=vector_doc.content,
                confidence=ConfidenceLevel(metadata.get("confidence", "medium")),
                domain=metadata.get("domain"),
                importance=metadata.get("importance", 0.5),
                verified=metadata.get("verified", False),
                tags=metadata.get("tags", []),
                created_at=datetime.fromisoformat(metadata.get("created_at", datetime.utcnow().isoformat())),
                metadata=metadata
            )
            
            return knowledge
            
        except Exception as e:
            logger.error(f"Failed to reconstruct knowledge: {e}")
            return None
    
    async def _build_concept_graph(self) -> None:
        """Build concept graph from knowledge base"""
        try:
            self.concept_graph = ConceptGraph()
            
            # Add nodes for each knowledge item
            for knowledge in self.knowledge_base.values():
                self.concept_graph.nodes[knowledge.id] = {
                    "subject": knowledge.subject,
                    "type": knowledge.type.value,
                    "domain": knowledge.domain,
                    "importance": knowledge.importance
                }
            
            # Add edges for related knowledge
            for knowledge in self.knowledge_base.values():
                for related_id in knowledge.related_knowledge:
                    if related_id in self.concept_graph.nodes:
                        self.concept_graph.edges.append((knowledge.id, related_id, "related"))
            
            logger.info(f"Built concept graph with {len(self.concept_graph.nodes)} nodes and {len(self.concept_graph.edges)} edges")
            
        except Exception as e:
            logger.error(f"Failed to build concept graph: {e}")
    
    async def _update_concept_graph(self, knowledge: Knowledge) -> None:
        """Update concept graph with new knowledge"""
        try:
            # Add node
            self.concept_graph.nodes[knowledge.id] = {
                "subject": knowledge.subject,
                "type": knowledge.type.value,
                "domain": knowledge.domain,
                "importance": knowledge.importance
            }
            
            # Add edges for related knowledge
            for related_id in knowledge.related_knowledge:
                if related_id in self.concept_graph.nodes:
                    self.concept_graph.edges.append((knowledge.id, related_id, "related"))
            
        except Exception as e:
            logger.error(f"Failed to update concept graph: {e}")
    
    async def _update_related_knowledge(self, knowledge: Knowledge) -> None:
        """Update related knowledge relationships"""
        try:
            # Find semantically similar knowledge
            similar_items = await self.retrieve_knowledge(
                query=self._knowledge_to_text(knowledge),
                limit=10
            )
            
            # Update bidirectional relationships
            for similar_item in similar_items:
                if similar_item.id != knowledge.id:
                    # Add to current knowledge
                    if similar_item.id not in knowledge.related_knowledge:
                        knowledge.related_knowledge.append(similar_item.id)
                    
                    # Add to similar knowledge
                    if knowledge.id not in similar_item.related_knowledge:
                        similar_item.related_knowledge.append(knowledge.id)
                        await self._store_knowledge_in_redis(similar_item)
            
        except Exception as e:
            logger.error(f"Failed to update related knowledge: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get semantic memory statistics"""
        try:
            total_knowledge = len(self.knowledge_base)
            
            # Count by type
            type_counts = {}
            domain_counts = {}
            confidence_counts = {}
            verified_count = 0
            importance_sum = 0
            
            for knowledge in self.knowledge_base.values():
                type_counts[knowledge.type.value] = type_counts.get(knowledge.type.value, 0) + 1
                
                if knowledge.domain:
                    domain_counts[knowledge.domain] = domain_counts.get(knowledge.domain, 0) + 1
                
                confidence_counts[knowledge.confidence.value] = confidence_counts.get(knowledge.confidence.value, 0) + 1
                
                if knowledge.verified:
                    verified_count += 1
                
                importance_sum += knowledge.importance
            
            avg_importance = importance_sum / total_knowledge if total_knowledge > 0 else 0
            verification_rate = verified_count / total_knowledge if total_knowledge > 0 else 0
            
            return {
                "agent_id": self.agent_id,
                "total_knowledge": total_knowledge,
                "type_distribution": type_counts,
                "domain_distribution": domain_counts,
                "confidence_distribution": confidence_counts,
                "verification_rate": verification_rate,
                "average_importance": avg_importance,
                "concept_graph": {
                    "nodes": len(self.concept_graph.nodes),
                    "edges": len(self.concept_graph.edges)
                },
                "vector_store_stats": await self.vector_store.get_stats()
            }
            
        except Exception as e:
            logger.error(f"Failed to get semantic memory stats: {e}")
            return {"error": str(e)}
