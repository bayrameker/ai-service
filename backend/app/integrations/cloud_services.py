"""
Cloud Services Integration (AWS, Azure, GCP)
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import json
import uuid
import base64

# AWS imports
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

# Azure imports
try:
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient
    from azure.keyvault.secrets import SecretClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

# GCP imports
try:
    from google.cloud import storage as gcp_storage
    from google.cloud import secretmanager
    from google.oauth2 import service_account
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

from app.core.config import settings

logger = logging.getLogger(__name__)


class CloudProvider(str, Enum):
    """Supported cloud providers"""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"


class ServiceType(str, Enum):
    """Cloud service types"""
    STORAGE = "storage"
    COMPUTE = "compute"
    DATABASE = "database"
    SECRETS = "secrets"
    MESSAGING = "messaging"
    AI_ML = "ai_ml"
    MONITORING = "monitoring"
    CUSTOM = "custom"


class CloudConfig(BaseModel):
    """Cloud service configuration"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    provider: CloudProvider
    service_type: ServiceType
    region: Optional[str] = None
    credentials: Dict[str, Any] = Field(default_factory=dict)
    endpoint: Optional[str] = None
    enabled: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CloudRequest(BaseModel):
    """Cloud service request"""
    service_id: str
    operation: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timeout: Optional[int] = None


class CloudResponse(BaseModel):
    """Cloud service response"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BaseCloudConnector:
    """Base class for cloud service connectors"""
    
    def __init__(self, config: CloudConfig):
        self.config = config
        self.client = None
        self.connected = False
    
    async def connect(self) -> bool:
        """Connect to cloud service"""
        raise NotImplementedError
    
    async def disconnect(self) -> None:
        """Disconnect from cloud service"""
        raise NotImplementedError
    
    async def execute_operation(self, operation: str, parameters: Dict[str, Any]) -> CloudResponse:
        """Execute cloud operation"""
        raise NotImplementedError
    
    async def health_check(self) -> bool:
        """Check service health"""
        raise NotImplementedError


class AWSConnector(BaseCloudConnector):
    """AWS services connector"""
    
    async def connect(self) -> bool:
        """Connect to AWS"""
        if not AWS_AVAILABLE:
            raise ImportError("boto3 is required for AWS connections")
        
        try:
            # Initialize AWS session
            session = boto3.Session(
                aws_access_key_id=self.config.credentials.get("access_key_id"),
                aws_secret_access_key=self.config.credentials.get("secret_access_key"),
                region_name=self.config.region or "us-east-1"
            )
            
            # Create service client based on service type
            if self.config.service_type == ServiceType.STORAGE:
                self.client = session.client('s3')
            elif self.config.service_type == ServiceType.SECRETS:
                self.client = session.client('secretsmanager')
            elif self.config.service_type == ServiceType.COMPUTE:
                self.client = session.client('ec2')
            elif self.config.service_type == ServiceType.DATABASE:
                self.client = session.client('rds')
            elif self.config.service_type == ServiceType.AI_ML:
                self.client = session.client('bedrock-runtime')
            else:
                # Default to S3
                self.client = session.client('s3')
            
            self.connected = True
            logger.info(f"Connected to AWS {self.config.service_type}: {self.config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to AWS {self.config.name}: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from AWS"""
        self.client = None
        self.connected = False
        logger.info(f"Disconnected from AWS: {self.config.name}")
    
    async def execute_operation(self, operation: str, parameters: Dict[str, Any]) -> CloudResponse:
        """Execute AWS operation"""
        start_time = datetime.utcnow()
        
        try:
            if not self.connected:
                await self.connect()
            
            # Execute operation based on service type and operation
            if self.config.service_type == ServiceType.STORAGE:
                result = await self._execute_s3_operation(operation, parameters)
            elif self.config.service_type == ServiceType.SECRETS:
                result = await self._execute_secrets_operation(operation, parameters)
            else:
                raise ValueError(f"Unsupported operation {operation} for service type {self.config.service_type}")
            
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return CloudResponse(
                success=True,
                data=result,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            logger.error(f"AWS operation failed: {e}")
            
            return CloudResponse(
                success=False,
                error=str(e),
                execution_time_ms=execution_time
            )
    
    async def _execute_s3_operation(self, operation: str, parameters: Dict[str, Any]) -> Any:
        """Execute S3 operations"""
        if operation == "list_buckets":
            response = self.client.list_buckets()
            return [bucket['Name'] for bucket in response['Buckets']]
        
        elif operation == "list_objects":
            bucket = parameters.get("bucket")
            prefix = parameters.get("prefix", "")
            response = self.client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            return [obj['Key'] for obj in response.get('Contents', [])]
        
        elif operation == "upload_object":
            bucket = parameters.get("bucket")
            key = parameters.get("key")
            data = parameters.get("data")
            
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            self.client.put_object(Bucket=bucket, Key=key, Body=data)
            return {"uploaded": key}
        
        elif operation == "download_object":
            bucket = parameters.get("bucket")
            key = parameters.get("key")
            
            response = self.client.get_object(Bucket=bucket, Key=key)
            return response['Body'].read().decode('utf-8')
        
        else:
            raise ValueError(f"Unsupported S3 operation: {operation}")
    
    async def _execute_secrets_operation(self, operation: str, parameters: Dict[str, Any]) -> Any:
        """Execute Secrets Manager operations"""
        if operation == "get_secret":
            secret_id = parameters.get("secret_id")
            response = self.client.get_secret_value(SecretId=secret_id)
            return response['SecretString']
        
        elif operation == "create_secret":
            name = parameters.get("name")
            secret_string = parameters.get("secret_string")
            description = parameters.get("description", "")
            
            response = self.client.create_secret(
                Name=name,
                SecretString=secret_string,
                Description=description
            )
            return {"arn": response['ARN']}
        
        else:
            raise ValueError(f"Unsupported Secrets Manager operation: {operation}")
    
    async def health_check(self) -> bool:
        """Check AWS service health"""
        try:
            if self.config.service_type == ServiceType.STORAGE:
                self.client.list_buckets()
            elif self.config.service_type == ServiceType.SECRETS:
                self.client.list_secrets(MaxResults=1)
            return True
        except Exception:
            return False


class AzureConnector(BaseCloudConnector):
    """Azure services connector"""
    
    async def connect(self) -> bool:
        """Connect to Azure"""
        if not AZURE_AVAILABLE:
            raise ImportError("Azure SDK is required for Azure connections")
        
        try:
            # Initialize Azure credential
            if self.config.credentials.get("client_id"):
                # Service principal authentication
                from azure.identity import ClientSecretCredential
                credential = ClientSecretCredential(
                    tenant_id=self.config.credentials.get("tenant_id"),
                    client_id=self.config.credentials.get("client_id"),
                    client_secret=self.config.credentials.get("client_secret")
                )
            else:
                # Default credential chain
                credential = DefaultAzureCredential()
            
            # Create service client based on service type
            if self.config.service_type == ServiceType.STORAGE:
                account_url = f"https://{self.config.credentials.get('storage_account')}.blob.core.windows.net"
                self.client = BlobServiceClient(account_url=account_url, credential=credential)
            elif self.config.service_type == ServiceType.SECRETS:
                vault_url = f"https://{self.config.credentials.get('key_vault')}.vault.azure.net/"
                self.client = SecretClient(vault_url=vault_url, credential=credential)
            else:
                raise ValueError(f"Unsupported Azure service type: {self.config.service_type}")
            
            self.connected = True
            logger.info(f"Connected to Azure {self.config.service_type}: {self.config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Azure {self.config.name}: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Azure"""
        if hasattr(self.client, 'close'):
            await self.client.close()
        self.client = None
        self.connected = False
        logger.info(f"Disconnected from Azure: {self.config.name}")
    
    async def execute_operation(self, operation: str, parameters: Dict[str, Any]) -> CloudResponse:
        """Execute Azure operation"""
        start_time = datetime.utcnow()
        
        try:
            if not self.connected:
                await self.connect()
            
            # Execute operation based on service type
            if self.config.service_type == ServiceType.STORAGE:
                result = await self._execute_blob_operation(operation, parameters)
            elif self.config.service_type == ServiceType.SECRETS:
                result = await self._execute_keyvault_operation(operation, parameters)
            else:
                raise ValueError(f"Unsupported operation {operation} for service type {self.config.service_type}")
            
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return CloudResponse(
                success=True,
                data=result,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            logger.error(f"Azure operation failed: {e}")
            
            return CloudResponse(
                success=False,
                error=str(e),
                execution_time_ms=execution_time
            )
    
    async def _execute_blob_operation(self, operation: str, parameters: Dict[str, Any]) -> Any:
        """Execute Azure Blob Storage operations"""
        if operation == "list_containers":
            containers = self.client.list_containers()
            return [container.name for container in containers]
        
        elif operation == "upload_blob":
            container = parameters.get("container")
            blob_name = parameters.get("blob_name")
            data = parameters.get("data")
            
            blob_client = self.client.get_blob_client(container=container, blob=blob_name)
            blob_client.upload_blob(data, overwrite=True)
            return {"uploaded": blob_name}
        
        elif operation == "download_blob":
            container = parameters.get("container")
            blob_name = parameters.get("blob_name")
            
            blob_client = self.client.get_blob_client(container=container, blob=blob_name)
            return blob_client.download_blob().readall().decode('utf-8')
        
        else:
            raise ValueError(f"Unsupported Blob operation: {operation}")
    
    async def _execute_keyvault_operation(self, operation: str, parameters: Dict[str, Any]) -> Any:
        """Execute Azure Key Vault operations"""
        if operation == "get_secret":
            secret_name = parameters.get("secret_name")
            secret = self.client.get_secret(secret_name)
            return secret.value
        
        elif operation == "set_secret":
            secret_name = parameters.get("secret_name")
            secret_value = parameters.get("secret_value")
            
            secret = self.client.set_secret(secret_name, secret_value)
            return {"name": secret.name, "version": secret.properties.version}
        
        else:
            raise ValueError(f"Unsupported Key Vault operation: {operation}")
    
    async def health_check(self) -> bool:
        """Check Azure service health"""
        try:
            if self.config.service_type == ServiceType.STORAGE:
                list(self.client.list_containers(max_results=1))
            elif self.config.service_type == ServiceType.SECRETS:
                list(self.client.list_properties_of_secrets(max_results=1))
            return True
        except Exception:
            return False


class GCPConnector(BaseCloudConnector):
    """Google Cloud Platform services connector"""
    
    async def connect(self) -> bool:
        """Connect to GCP"""
        if not GCP_AVAILABLE:
            raise ImportError("Google Cloud SDK is required for GCP connections")
        
        try:
            # Initialize GCP credentials
            if self.config.credentials.get("service_account_key"):
                credentials = service_account.Credentials.from_service_account_info(
                    self.config.credentials["service_account_key"]
                )
            else:
                credentials = None  # Use default credentials
            
            # Create service client based on service type
            if self.config.service_type == ServiceType.STORAGE:
                self.client = gcp_storage.Client(credentials=credentials)
            elif self.config.service_type == ServiceType.SECRETS:
                self.client = secretmanager.SecretManagerServiceClient(credentials=credentials)
            else:
                raise ValueError(f"Unsupported GCP service type: {self.config.service_type}")
            
            self.connected = True
            logger.info(f"Connected to GCP {self.config.service_type}: {self.config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to GCP {self.config.name}: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from GCP"""
        self.client = None
        self.connected = False
        logger.info(f"Disconnected from GCP: {self.config.name}")
    
    async def execute_operation(self, operation: str, parameters: Dict[str, Any]) -> CloudResponse:
        """Execute GCP operation"""
        start_time = datetime.utcnow()
        
        try:
            if not self.connected:
                await self.connect()
            
            # Execute operation based on service type
            if self.config.service_type == ServiceType.STORAGE:
                result = await self._execute_storage_operation(operation, parameters)
            elif self.config.service_type == ServiceType.SECRETS:
                result = await self._execute_secret_operation(operation, parameters)
            else:
                raise ValueError(f"Unsupported operation {operation} for service type {self.config.service_type}")
            
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return CloudResponse(
                success=True,
                data=result,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            logger.error(f"GCP operation failed: {e}")
            
            return CloudResponse(
                success=False,
                error=str(e),
                execution_time_ms=execution_time
            )
    
    async def _execute_storage_operation(self, operation: str, parameters: Dict[str, Any]) -> Any:
        """Execute Google Cloud Storage operations"""
        if operation == "list_buckets":
            buckets = self.client.list_buckets()
            return [bucket.name for bucket in buckets]
        
        elif operation == "upload_blob":
            bucket_name = parameters.get("bucket")
            blob_name = parameters.get("blob_name")
            data = parameters.get("data")
            
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_string(data)
            return {"uploaded": blob_name}
        
        elif operation == "download_blob":
            bucket_name = parameters.get("bucket")
            blob_name = parameters.get("blob_name")
            
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            return blob.download_as_text()
        
        else:
            raise ValueError(f"Unsupported Storage operation: {operation}")
    
    async def _execute_secret_operation(self, operation: str, parameters: Dict[str, Any]) -> Any:
        """Execute Google Secret Manager operations"""
        if operation == "get_secret":
            project_id = parameters.get("project_id")
            secret_id = parameters.get("secret_id")
            version_id = parameters.get("version_id", "latest")
            
            name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
            response = self.client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        
        elif operation == "create_secret":
            project_id = parameters.get("project_id")
            secret_id = parameters.get("secret_id")
            secret_data = parameters.get("secret_data")
            
            parent = f"projects/{project_id}"
            secret = {"replication": {"automatic": {}}}
            
            response = self.client.create_secret(
                request={"parent": parent, "secret_id": secret_id, "secret": secret}
            )
            
            # Add secret version
            version_response = self.client.add_secret_version(
                request={"parent": response.name, "payload": {"data": secret_data.encode("UTF-8")}}
            )
            
            return {"name": response.name, "version": version_response.name}
        
        else:
            raise ValueError(f"Unsupported Secret Manager operation: {operation}")
    
    async def health_check(self) -> bool:
        """Check GCP service health"""
        try:
            if self.config.service_type == ServiceType.STORAGE:
                list(self.client.list_buckets(max_results=1))
            return True
        except Exception:
            return False


class CloudServiceManager:
    """Cloud services manager"""
    
    def __init__(self):
        self.connectors: Dict[str, BaseCloudConnector] = {}
        self.configs: Dict[str, CloudConfig] = {}
    
    def register_service(self, config: CloudConfig) -> str:
        """Register a cloud service"""
        try:
            # Create appropriate connector
            if config.provider == CloudProvider.AWS:
                connector = AWSConnector(config)
            elif config.provider == CloudProvider.AZURE:
                connector = AzureConnector(config)
            elif config.provider == CloudProvider.GCP:
                connector = GCPConnector(config)
            else:
                raise ValueError(f"Unsupported cloud provider: {config.provider}")
            
            self.connectors[config.id] = connector
            self.configs[config.id] = config
            
            logger.info(f"Registered cloud service: {config.name} ({config.provider})")
            return config.id
            
        except Exception as e:
            logger.error(f"Failed to register cloud service: {e}")
            raise
    
    async def connect_service(self, service_id: str) -> bool:
        """Connect to a specific cloud service"""
        connector = self.connectors.get(service_id)
        if not connector:
            return False
        
        return await connector.connect()
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all registered cloud services"""
        results = {}
        
        for service_id, connector in self.connectors.items():
            config = self.configs[service_id]
            if config.enabled:
                results[service_id] = await connector.connect()
            else:
                results[service_id] = False
        
        return results
    
    async def execute_operation(self, request: CloudRequest) -> CloudResponse:
        """Execute operation on cloud service"""
        try:
            connector = self.connectors.get(request.service_id)
            if not connector:
                return CloudResponse(
                    success=False,
                    error=f"Service {request.service_id} not found",
                    execution_time_ms=0
                )
            
            if not connector.connected:
                await connector.connect()
            
            return await connector.execute_operation(request.operation, request.parameters)
            
        except Exception as e:
            logger.error(f"Failed to execute cloud operation: {e}")
            return CloudResponse(
                success=False,
                error=str(e),
                execution_time_ms=0
            )
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all cloud services"""
        results = {}
        
        for service_id, connector in self.connectors.items():
            config = self.configs[service_id]
            if config.enabled:
                results[service_id] = await connector.health_check()
            else:
                results[service_id] = False
        
        return results
    
    def get_service_config(self, service_id: str) -> Optional[CloudConfig]:
        """Get cloud service configuration"""
        return self.configs.get(service_id)
    
    def list_services(self) -> List[CloudConfig]:
        """List all registered cloud services"""
        return list(self.configs.values())
    
    def remove_service(self, service_id: str) -> bool:
        """Remove cloud service from manager"""
        try:
            if service_id in self.connectors:
                connector = self.connectors[service_id]
                asyncio.create_task(connector.disconnect())
                del self.connectors[service_id]
                del self.configs[service_id]
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove cloud service: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown all cloud service connections"""
        try:
            for connector in self.connectors.values():
                await connector.disconnect()
            
            logger.info("All cloud service connections closed")
            
        except Exception as e:
            logger.error(f"Error during cloud services shutdown: {e}")


# Global cloud service manager instance
cloud_service_manager = CloudServiceManager()
