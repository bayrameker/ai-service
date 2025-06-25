"""
Vector Database Integration for AI Memory System
"""

import asyncio
import logging
import numpy as np
import pickle
import os
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod
from pydantic import BaseModel
from datetime import datetime
import uuid

# Vector database imports
import faiss
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorDocument(BaseModel):
    """Document stored in vector database"""
    id: str
    content: str
    metadata: Dict[str, Any] = {}
    embedding: Optional[List[float]] = None
    created_at: datetime = datetime.utcnow()
    updated_at: datetime = datetime.utcnow()


class SearchResult(BaseModel):
    """Search result from vector database"""
    document: VectorDocument
    score: float
    distance: float


class BaseVectorStore(ABC):
    """Base class for vector stores"""
    
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store"""
        pass
    
    @abstractmethod
    async def add_documents(self, documents: List[VectorDocument]) -> None:
        """Add documents to the vector store"""
        pass
    
    @abstractmethod
    async def search(self, query_embedding: List[float], k: int = 10, filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    async def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Get document by ID"""
        pass
    
    @abstractmethod
    async def update_document(self, doc_id: str, document: VectorDocument) -> bool:
        """Update document"""
        pass
    
    @abstractmethod
    async def delete_document(self, doc_id: str) -> bool:
        """Delete document"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        pass


class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store implementation"""
    
    def __init__(self, dimension: int = 1536, index_path: Optional[str] = None):
        super().__init__(dimension)
        self.index_path = index_path or os.path.join(settings.VECTOR_DB_PATH, "faiss_index")
        self.index: Optional[faiss.Index] = None
        self.documents: Dict[str, VectorDocument] = {}
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        self.next_index = 0
    
    async def initialize(self) -> None:
        """Initialize FAISS index"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Try to load existing index
            if os.path.exists(f"{self.index_path}.index"):
                await self._load_index()
            else:
                # Create new index
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity)
                await self._save_index()
            
            self.initialized = True
            logger.info(f"FAISS vector store initialized with {self.index.ntotal} documents")
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS vector store: {e}")
            raise
    
    async def _load_index(self) -> None:
        """Load FAISS index from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{self.index_path}.index")
            
            # Load metadata
            with open(f"{self.index_path}.metadata", 'rb') as f:
                metadata = pickle.load(f)
                self.documents = metadata['documents']
                self.id_to_index = metadata['id_to_index']
                self.index_to_id = metadata['index_to_id']
                self.next_index = metadata['next_index']
            
            logger.info(f"Loaded FAISS index with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            # Create new index if loading fails
            self.index = faiss.IndexFlatIP(self.dimension)
            self.documents = {}
            self.id_to_index = {}
            self.index_to_id = {}
            self.next_index = 0
    
    async def _save_index(self) -> None:
        """Save FAISS index to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{self.index_path}.index")
            
            # Save metadata
            metadata = {
                'documents': self.documents,
                'id_to_index': self.id_to_index,
                'index_to_id': self.index_to_id,
                'next_index': self.next_index
            }
            
            with open(f"{self.index_path}.metadata", 'wb') as f:
                pickle.dump(metadata, f)
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    async def add_documents(self, documents: List[VectorDocument]) -> None:
        """Add documents to FAISS index"""
        if not self.initialized:
            await self.initialize()
        
        try:
            embeddings = []
            for doc in documents:
                if not doc.embedding:
                    raise ValueError(f"Document {doc.id} has no embedding")
                
                # Store document
                self.documents[doc.id] = doc
                
                # Map ID to index
                self.id_to_index[doc.id] = self.next_index
                self.index_to_id[self.next_index] = doc.id
                
                embeddings.append(doc.embedding)
                self.next_index += 1
            
            # Add to FAISS index
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings_array)
            
            self.index.add(embeddings_array)
            
            # Save index
            await self._save_index()
            
            logger.info(f"Added {len(documents)} documents to FAISS index")
            
        except Exception as e:
            logger.error(f"Failed to add documents to FAISS index: {e}")
            raise
    
    async def search(self, query_embedding: List[float], k: int = 10, filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar documents in FAISS index"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Prepare query embedding
            query_array = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_array)
            
            # Search
            scores, indices = self.index.search(query_array, k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # No more results
                    break
                
                doc_id = self.index_to_id.get(idx)
                if not doc_id:
                    continue
                
                document = self.documents.get(doc_id)
                if not document:
                    continue
                
                # Apply metadata filter if provided
                if filter_metadata:
                    match = True
                    for key, value in filter_metadata.items():
                        if document.metadata.get(key) != value:
                            match = False
                            break
                    if not match:
                        continue
                
                results.append(SearchResult(
                    document=document,
                    score=float(score),
                    distance=1.0 - float(score)  # Convert similarity to distance
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search FAISS index: {e}")
            return []
    
    async def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Get document by ID"""
        return self.documents.get(doc_id)
    
    async def update_document(self, doc_id: str, document: VectorDocument) -> bool:
        """Update document (requires rebuilding index for FAISS)"""
        if doc_id not in self.documents:
            return False
        
        # For FAISS, we need to rebuild the index to update embeddings
        # This is a limitation of FAISS
        self.documents[doc_id] = document
        document.updated_at = datetime.utcnow()
        
        # Save metadata
        await self._save_index()
        
        return True
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete document (requires rebuilding index for FAISS)"""
        if doc_id not in self.documents:
            return False
        
        # Remove from documents
        del self.documents[doc_id]
        
        # For FAISS, we need to rebuild the index to remove embeddings
        # This is a limitation of FAISS
        await self._rebuild_index()
        
        return True
    
    async def _rebuild_index(self) -> None:
        """Rebuild FAISS index from scratch"""
        try:
            # Create new index
            new_index = faiss.IndexFlatIP(self.dimension)
            new_id_to_index = {}
            new_index_to_id = {}
            
            # Add all documents
            embeddings = []
            doc_ids = []
            
            for i, (doc_id, doc) in enumerate(self.documents.items()):
                if doc.embedding:
                    embeddings.append(doc.embedding)
                    doc_ids.append(doc_id)
                    new_id_to_index[doc_id] = i
                    new_index_to_id[i] = doc_id
            
            if embeddings:
                embeddings_array = np.array(embeddings, dtype=np.float32)
                faiss.normalize_L2(embeddings_array)
                new_index.add(embeddings_array)
            
            # Replace old index
            self.index = new_index
            self.id_to_index = new_id_to_index
            self.index_to_id = new_index_to_id
            self.next_index = len(embeddings)
            
            # Save
            await self._save_index()
            
            logger.info(f"Rebuilt FAISS index with {len(embeddings)} documents")
            
        except Exception as e:
            logger.error(f"Failed to rebuild FAISS index: {e}")
            raise
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get FAISS vector store statistics"""
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "index_type": "FAISS IndexFlatIP",
            "storage_path": self.index_path
        }


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB-based vector store implementation"""
    
    def __init__(self, dimension: int = 1536, collection_name: str = "ai_memory"):
        super().__init__(dimension)
        self.collection_name = collection_name
        self.client = None
        self.collection = None
    
    async def initialize(self) -> None:
        """Initialize ChromaDB"""
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is not available. Install with: pip install chromadb")
        
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=settings.VECTOR_DB_PATH,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"dimension": self.dimension}
            )
            
            self.initialized = True
            logger.info(f"ChromaDB vector store initialized with collection '{self.collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB vector store: {e}")
            raise
    
    async def add_documents(self, documents: List[VectorDocument]) -> None:
        """Add documents to ChromaDB"""
        if not self.initialized:
            await self.initialize()
        
        try:
            ids = []
            embeddings = []
            metadatas = []
            documents_content = []
            
            for doc in documents:
                if not doc.embedding:
                    raise ValueError(f"Document {doc.id} has no embedding")
                
                ids.append(doc.id)
                embeddings.append(doc.embedding)
                metadatas.append({
                    **doc.metadata,
                    "created_at": doc.created_at.isoformat(),
                    "updated_at": doc.updated_at.isoformat()
                })
                documents_content.append(doc.content)
            
            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_content
            )
            
            logger.info(f"Added {len(documents)} documents to ChromaDB")
            
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            raise
    
    async def search(self, query_embedding: List[float], k: int = 10, filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar documents in ChromaDB"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Prepare where clause for filtering
            where_clause = filter_metadata if filter_metadata else None
            
            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_clause
            )
            
            search_results = []
            
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    document = VectorDocument(
                        id=doc_id,
                        content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i],
                        embedding=results['embeddings'][0][i] if results['embeddings'] else None
                    )
                    
                    # ChromaDB returns distances, convert to similarity score
                    distance = results['distances'][0][i]
                    score = 1.0 / (1.0 + distance)  # Convert distance to similarity
                    
                    search_results.append(SearchResult(
                        document=document,
                        score=score,
                        distance=distance
                    ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search ChromaDB: {e}")
            return []
    
    async def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Get document by ID from ChromaDB"""
        if not self.initialized:
            await self.initialize()
        
        try:
            results = self.collection.get(ids=[doc_id])
            
            if results['ids'] and results['ids'][0]:
                return VectorDocument(
                    id=results['ids'][0],
                    content=results['documents'][0],
                    metadata=results['metadatas'][0],
                    embedding=results['embeddings'][0] if results['embeddings'] else None
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document from ChromaDB: {e}")
            return None
    
    async def update_document(self, doc_id: str, document: VectorDocument) -> bool:
        """Update document in ChromaDB"""
        if not self.initialized:
            await self.initialize()
        
        try:
            document.updated_at = datetime.utcnow()
            
            self.collection.update(
                ids=[doc_id],
                embeddings=[document.embedding] if document.embedding else None,
                metadatas=[{
                    **document.metadata,
                    "created_at": document.created_at.isoformat(),
                    "updated_at": document.updated_at.isoformat()
                }],
                documents=[document.content]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document in ChromaDB: {e}")
            return False
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete document from ChromaDB"""
        if not self.initialized:
            await self.initialize()
        
        try:
            self.collection.delete(ids=[doc_id])
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document from ChromaDB: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get ChromaDB vector store statistics"""
        if not self.initialized:
            await self.initialize()
        
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "dimension": self.dimension,
                "index_type": "ChromaDB",
                "collection_name": self.collection_name,
                "storage_path": settings.VECTOR_DB_PATH
            }
        except Exception as e:
            logger.error(f"Failed to get ChromaDB stats: {e}")
            return {
                "total_documents": 0,
                "dimension": self.dimension,
                "index_type": "ChromaDB",
                "collection_name": self.collection_name,
                "error": str(e)
            }


def create_vector_store(store_type: str = None, **kwargs) -> BaseVectorStore:
    """Factory function to create vector store"""
    store_type = store_type or settings.VECTOR_DB_TYPE.lower()
    
    if store_type == "faiss":
        return FAISSVectorStore(**kwargs)
    elif store_type == "chroma":
        return ChromaVectorStore(**kwargs)
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")
