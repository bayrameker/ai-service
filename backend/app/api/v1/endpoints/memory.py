"""
Memory Management API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List, Optional, Dict, Any
import logging

from app.memory.manager import MemoryManager
from app.memory.episodic import Experience, ExperienceType
from app.memory.semantic import Knowledge, KnowledgeType, ConfidenceLevel
from app.memory.rag import RAGQuery, RetrievalStrategy
from app.memory.working import ContextType, Priority

logger = logging.getLogger(__name__)
router = APIRouter()


async def get_memory_manager(request: Request) -> MemoryManager:
    """Get memory manager from app state"""
    return request.app.state.memory_manager


@router.post("/experiences")
async def store_experience(
    experience: Experience,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """Store an experience in episodic memory"""
    try:
        experience_id = await memory_manager.store_experience(experience)
        return {"experience_id": experience_id, "status": "stored"}
    except Exception as e:
        logger.error(f"Failed to store experience: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiences")
async def get_experiences(
    query: str,
    limit: int = 10,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """Retrieve experiences from episodic memory"""
    try:
        experiences = await memory_manager.get_experiences(query, limit)
        return [exp.dict() for exp in experiences]
    except Exception as e:
        logger.error(f"Failed to get experiences: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge")
async def store_knowledge(
    knowledge: Knowledge,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """Store knowledge in semantic memory"""
    try:
        knowledge_id = await memory_manager.store_knowledge(knowledge)
        return {"knowledge_id": knowledge_id, "status": "stored"}
    except Exception as e:
        logger.error(f"Failed to store knowledge: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge")
async def get_knowledge(
    query: str,
    limit: int = 10,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """Retrieve knowledge from semantic memory"""
    try:
        knowledge_items = await memory_manager.get_knowledge(query, limit)
        return [knowledge.dict() for knowledge in knowledge_items]
    except Exception as e:
        logger.error(f"Failed to get knowledge: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/context")
async def store_context(
    key: str,
    value: Any,
    context_type: ContextType = ContextType.TEMPORARY,
    priority: Priority = Priority.NORMAL,
    ttl: Optional[int] = None,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """Store context in working memory"""
    try:
        context_id = await memory_manager.store_context(
            key, value, context_type=context_type, priority=priority, ttl=ttl
        )
        return {"context_id": context_id, "status": "stored"}
    except Exception as e:
        logger.error(f"Failed to store context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/context/{key}")
async def get_context(
    key: str,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """Get context from working memory"""
    try:
        value = await memory_manager.get_context(key)
        if value is None:
            raise HTTPException(status_code=404, detail="Context not found")
        return {"key": key, "value": value}
    except Exception as e:
        logger.error(f"Failed to get context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversation")
async def add_conversation_turn(
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """Add conversation turn to working memory"""
    try:
        turn_id = await memory_manager.add_conversation_turn(role, content, metadata)
        return {"turn_id": turn_id, "status": "added"}
    except Exception as e:
        logger.error(f"Failed to add conversation turn: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/query")
async def rag_query(
    rag_query: RAGQuery,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """Execute RAG query"""
    try:
        response = await memory_manager.query_with_rag(rag_query)
        return response.dict()
    except Exception as e:
        logger.error(f"Failed to execute RAG query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask")
async def ask_question(
    question: str,
    context: Optional[str] = None,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """Ask a question using RAG system"""
    try:
        answer = await memory_manager.ask_question(question, context)
        return {"question": question, "answer": answer}
    except Exception as e:
        logger.error(f"Failed to ask question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search_all_memories(
    query: str,
    limit: int = 10,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """Search across all memory systems"""
    try:
        results = await memory_manager.search_all_memories(query, limit)
        return results
    except Exception as e:
        logger.error(f"Failed to search memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/learning/insights")
async def get_learning_insights(
    limit: Optional[int] = None,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """Get learning insights"""
    try:
        insights = await memory_manager.get_learning_insights(limit)
        return [insight.dict() for insight in insights]
    except Exception as e:
        logger.error(f"Failed to get learning insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_memory_stats(
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """Get memory system statistics"""
    try:
        return await memory_manager.get_memory_stats()
    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup")
async def cleanup_memories(
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """Clean up expired memories"""
    try:
        results = await memory_manager.cleanup_expired_memories()
        return results
    except Exception as e:
        logger.error(f"Failed to cleanup memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export")
async def export_memories(
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """Export memories for backup"""
    try:
        export_data = await memory_manager.export_memories()
        return export_data
    except Exception as e:
        logger.error(f"Failed to export memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def memory_health_check(
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """Check health of memory systems"""
    try:
        health = await memory_manager.health_check()
        return health
    except Exception as e:
        logger.error(f"Failed to check memory health: {e}")
        raise HTTPException(status_code=500, detail=str(e))
