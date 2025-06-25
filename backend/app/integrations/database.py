"""
Database Connectors for SQL and NoSQL Databases
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import json
import uuid

# SQL Database imports
try:
    import asyncpg
    import aiomysql
    import aiosqlite
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

# NoSQL Database imports
try:
    import motor.motor_asyncio
    MOTOR_AVAILABLE = True
except ImportError:
    MOTOR_AVAILABLE = False

from app.core.config import settings

logger = logging.getLogger(__name__)


class DatabaseType(str, Enum):
    """Supported database types"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"
    REDIS = "redis"


class QueryType(str, Enum):
    """Query types"""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"
    DROP = "drop"
    CUSTOM = "custom"


class DatabaseConfig(BaseModel):
    """Database configuration"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: DatabaseType
    host: str
    port: int
    database: str
    username: Optional[str] = None
    password: Optional[str] = None
    connection_string: Optional[str] = None
    ssl: bool = False
    pool_size: int = 10
    max_connections: int = 20
    timeout: int = 30
    enabled: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """Database query request"""
    database_id: str
    query_type: QueryType
    query: str
    parameters: Optional[Dict[str, Any]] = None
    timeout: Optional[int] = None


class QueryResult(BaseModel):
    """Database query result"""
    success: bool
    data: Optional[Any] = None
    rows_affected: Optional[int] = None
    error: Optional[str] = None
    execution_time_ms: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BaseDatabaseConnector:
    """Base class for database connectors"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection = None
        self.pool = None
        self.connected = False
    
    async def connect(self) -> bool:
        """Connect to database"""
        raise NotImplementedError
    
    async def disconnect(self) -> None:
        """Disconnect from database"""
        raise NotImplementedError
    
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute query"""
        raise NotImplementedError
    
    async def health_check(self) -> bool:
        """Check database health"""
        raise NotImplementedError


class PostgreSQLConnector(BaseDatabaseConnector):
    """PostgreSQL database connector"""
    
    async def connect(self) -> bool:
        """Connect to PostgreSQL"""
        if not ASYNCPG_AVAILABLE:
            raise ImportError("asyncpg is required for PostgreSQL connections")
        
        try:
            if self.config.connection_string:
                self.pool = await asyncpg.create_pool(
                    self.config.connection_string,
                    min_size=1,
                    max_size=self.config.max_connections
                )
            else:
                self.pool = await asyncpg.create_pool(
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    user=self.config.username,
                    password=self.config.password,
                    ssl=self.config.ssl,
                    min_size=1,
                    max_size=self.config.max_connections
                )
            
            self.connected = True
            logger.info(f"Connected to PostgreSQL: {self.config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL {self.config.name}: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from PostgreSQL"""
        if self.pool:
            await self.pool.close()
            self.connected = False
            logger.info(f"Disconnected from PostgreSQL: {self.config.name}")
    
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute PostgreSQL query"""
        start_time = datetime.utcnow()
        
        try:
            if not self.connected:
                await self.connect()
            
            async with self.pool.acquire() as connection:
                if parameters:
                    # Convert named parameters to positional
                    param_values = list(parameters.values())
                    result = await connection.fetch(query, *param_values)
                else:
                    result = await connection.fetch(query)
                
                execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                
                # Convert result to serializable format
                data = []
                for row in result:
                    data.append(dict(row))
                
                return QueryResult(
                    success=True,
                    data=data,
                    rows_affected=len(data),
                    execution_time_ms=execution_time
                )
                
        except Exception as e:
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            logger.error(f"PostgreSQL query failed: {e}")
            
            return QueryResult(
                success=False,
                error=str(e),
                execution_time_ms=execution_time
            )
    
    async def health_check(self) -> bool:
        """Check PostgreSQL health"""
        try:
            result = await self.execute_query("SELECT 1")
            return result.success
        except Exception:
            return False


class MySQLConnector(BaseDatabaseConnector):
    """MySQL database connector"""
    
    async def connect(self) -> bool:
        """Connect to MySQL"""
        try:
            self.pool = await aiomysql.create_pool(
                host=self.config.host,
                port=self.config.port,
                user=self.config.username,
                password=self.config.password,
                db=self.config.database,
                minsize=1,
                maxsize=self.config.max_connections
            )
            
            self.connected = True
            logger.info(f"Connected to MySQL: {self.config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MySQL {self.config.name}: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from MySQL"""
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            self.connected = False
            logger.info(f"Disconnected from MySQL: {self.config.name}")
    
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute MySQL query"""
        start_time = datetime.utcnow()
        
        try:
            if not self.connected:
                await self.connect()
            
            async with self.pool.acquire() as connection:
                async with connection.cursor() as cursor:
                    if parameters:
                        await cursor.execute(query, parameters)
                    else:
                        await cursor.execute(query)
                    
                    if query.strip().upper().startswith('SELECT'):
                        result = await cursor.fetchall()
                        # Get column names
                        columns = [desc[0] for desc in cursor.description]
                        # Convert to list of dicts
                        data = [dict(zip(columns, row)) for row in result]
                    else:
                        data = None
                        await connection.commit()
                    
                    execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                    
                    return QueryResult(
                        success=True,
                        data=data,
                        rows_affected=cursor.rowcount,
                        execution_time_ms=execution_time
                    )
                    
        except Exception as e:
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            logger.error(f"MySQL query failed: {e}")
            
            return QueryResult(
                success=False,
                error=str(e),
                execution_time_ms=execution_time
            )
    
    async def health_check(self) -> bool:
        """Check MySQL health"""
        try:
            result = await self.execute_query("SELECT 1")
            return result.success
        except Exception:
            return False


class SQLiteConnector(BaseDatabaseConnector):
    """SQLite database connector"""
    
    async def connect(self) -> bool:
        """Connect to SQLite"""
        try:
            # SQLite doesn't need explicit connection setup
            self.connected = True
            logger.info(f"Connected to SQLite: {self.config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to SQLite {self.config.name}: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from SQLite"""
        self.connected = False
        logger.info(f"Disconnected from SQLite: {self.config.name}")
    
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute SQLite query"""
        start_time = datetime.utcnow()
        
        try:
            async with aiosqlite.connect(self.config.database) as connection:
                if parameters:
                    cursor = await connection.execute(query, parameters)
                else:
                    cursor = await connection.execute(query)
                
                if query.strip().upper().startswith('SELECT'):
                    rows = await cursor.fetchall()
                    # Get column names
                    columns = [desc[0] for desc in cursor.description]
                    # Convert to list of dicts
                    data = [dict(zip(columns, row)) for row in rows]
                else:
                    data = None
                    await connection.commit()
                
                execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                
                return QueryResult(
                    success=True,
                    data=data,
                    rows_affected=cursor.rowcount,
                    execution_time_ms=execution_time
                )
                
        except Exception as e:
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            logger.error(f"SQLite query failed: {e}")
            
            return QueryResult(
                success=False,
                error=str(e),
                execution_time_ms=execution_time
            )
    
    async def health_check(self) -> bool:
        """Check SQLite health"""
        try:
            result = await self.execute_query("SELECT 1")
            return result.success
        except Exception:
            return False


class MongoDBConnector(BaseDatabaseConnector):
    """MongoDB database connector"""
    
    async def connect(self) -> bool:
        """Connect to MongoDB"""
        if not MOTOR_AVAILABLE:
            raise ImportError("motor is required for MongoDB connections")
        
        try:
            if self.config.connection_string:
                self.connection = motor.motor_asyncio.AsyncIOMotorClient(
                    self.config.connection_string
                )
            else:
                connection_string = f"mongodb://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
                self.connection = motor.motor_asyncio.AsyncIOMotorClient(
                    connection_string
                )
            
            # Test connection
            await self.connection.admin.command('ping')
            
            self.connected = True
            logger.info(f"Connected to MongoDB: {self.config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB {self.config.name}: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from MongoDB"""
        if self.connection:
            self.connection.close()
            self.connected = False
            logger.info(f"Disconnected from MongoDB: {self.config.name}")
    
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute MongoDB query (simplified - uses JSON query format)"""
        start_time = datetime.utcnow()
        
        try:
            if not self.connected:
                await self.connect()
            
            # Parse query as JSON
            query_doc = json.loads(query)
            operation = query_doc.get("operation")
            collection_name = query_doc.get("collection")
            filter_doc = query_doc.get("filter", {})
            document = query_doc.get("document", {})
            
            db = self.connection[self.config.database]
            collection = db[collection_name]
            
            if operation == "find":
                cursor = collection.find(filter_doc)
                data = await cursor.to_list(length=None)
                rows_affected = len(data)
                
            elif operation == "insert_one":
                result = await collection.insert_one(document)
                data = {"inserted_id": str(result.inserted_id)}
                rows_affected = 1
                
            elif operation == "update_one":
                update_doc = query_doc.get("update", {})
                result = await collection.update_one(filter_doc, update_doc)
                data = {"modified_count": result.modified_count}
                rows_affected = result.modified_count
                
            elif operation == "delete_one":
                result = await collection.delete_one(filter_doc)
                data = {"deleted_count": result.deleted_count}
                rows_affected = result.deleted_count
                
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return QueryResult(
                success=True,
                data=data,
                rows_affected=rows_affected,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            logger.error(f"MongoDB query failed: {e}")
            
            return QueryResult(
                success=False,
                error=str(e),
                execution_time_ms=execution_time
            )
    
    async def health_check(self) -> bool:
        """Check MongoDB health"""
        try:
            if self.connection:
                await self.connection.admin.command('ping')
                return True
            return False
        except Exception:
            return False


class DatabaseManager:
    """Database connection manager"""
    
    def __init__(self):
        self.connectors: Dict[str, BaseDatabaseConnector] = {}
        self.configs: Dict[str, DatabaseConfig] = {}
    
    def register_database(self, config: DatabaseConfig) -> str:
        """Register a database configuration"""
        try:
            # Create appropriate connector
            if config.type == DatabaseType.POSTGRESQL:
                connector = PostgreSQLConnector(config)
            elif config.type == DatabaseType.MYSQL:
                connector = MySQLConnector(config)
            elif config.type == DatabaseType.SQLITE:
                connector = SQLiteConnector(config)
            elif config.type == DatabaseType.MONGODB:
                connector = MongoDBConnector(config)
            else:
                raise ValueError(f"Unsupported database type: {config.type}")
            
            self.connectors[config.id] = connector
            self.configs[config.id] = config
            
            logger.info(f"Registered database: {config.name} ({config.type})")
            return config.id
            
        except Exception as e:
            logger.error(f"Failed to register database: {e}")
            raise
    
    async def connect_database(self, database_id: str) -> bool:
        """Connect to a specific database"""
        connector = self.connectors.get(database_id)
        if not connector:
            return False
        
        return await connector.connect()
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all registered databases"""
        results = {}
        
        for db_id, connector in self.connectors.items():
            config = self.configs[db_id]
            if config.enabled:
                results[db_id] = await connector.connect()
            else:
                results[db_id] = False
        
        return results
    
    async def execute_query(self, request: QueryRequest) -> QueryResult:
        """Execute query on specified database"""
        try:
            connector = self.connectors.get(request.database_id)
            if not connector:
                return QueryResult(
                    success=False,
                    error=f"Database {request.database_id} not found",
                    execution_time_ms=0
                )
            
            if not connector.connected:
                await connector.connect()
            
            return await connector.execute_query(request.query, request.parameters)
            
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            return QueryResult(
                success=False,
                error=str(e),
                execution_time_ms=0
            )
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all databases"""
        results = {}
        
        for db_id, connector in self.connectors.items():
            config = self.configs[db_id]
            if config.enabled:
                results[db_id] = await connector.health_check()
            else:
                results[db_id] = False
        
        return results
    
    def get_database_config(self, database_id: str) -> Optional[DatabaseConfig]:
        """Get database configuration"""
        return self.configs.get(database_id)
    
    def list_databases(self) -> List[DatabaseConfig]:
        """List all registered databases"""
        return list(self.configs.values())
    
    def remove_database(self, database_id: str) -> bool:
        """Remove database from manager"""
        try:
            if database_id in self.connectors:
                connector = self.connectors[database_id]
                asyncio.create_task(connector.disconnect())
                del self.connectors[database_id]
                del self.configs[database_id]
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove database: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown all database connections"""
        try:
            for connector in self.connectors.values():
                await connector.disconnect()
            
            logger.info("All database connections closed")
            
        except Exception as e:
            logger.error(f"Error during database shutdown: {e}")


# Global database manager instance
database_manager = DatabaseManager()
