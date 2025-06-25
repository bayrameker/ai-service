"""
Workflow Engine for Task Queue and Agent Orchestration
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field
import uuid

from app.agents.base import Task, TaskPriority, AgentState
from app.core.database import get_redis

logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TaskDependency(BaseModel):
    """Task dependency definition"""
    task_id: str
    dependency_type: str = "completion"  # completion, success, failure
    required: bool = True


class WorkflowStep(BaseModel):
    """Individual step in a workflow"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    task: Task
    dependencies: List[TaskDependency] = Field(default_factory=list)
    parallel_group: Optional[str] = None
    timeout_seconds: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    status: WorkflowStatus = WorkflowStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class Workflow(BaseModel):
    """Workflow definition"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    steps: List[WorkflowStep] = Field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskQueue:
    """Advanced task queue with priority and scheduling"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.priority_queues: Dict[TaskPriority, List[Task]] = {
            TaskPriority.URGENT: [],
            TaskPriority.HIGH: [],
            TaskPriority.NORMAL: [],
            TaskPriority.LOW: []
        }
        self.scheduled_tasks: List[tuple[datetime, Task]] = []
        self.processing_tasks: Dict[str, Task] = {}
        
    async def enqueue(self, task: Task, schedule_at: Optional[datetime] = None) -> None:
        """Add task to queue"""
        if schedule_at and schedule_at > datetime.utcnow():
            # Schedule for later
            self.scheduled_tasks.append((schedule_at, task))
            self.scheduled_tasks.sort(key=lambda x: x[0])
        else:
            # Add to priority queue
            self.priority_queues[task.priority].append(task)
        
        # Persist to Redis if available
        if self.redis_client:
            await self._persist_task(task)
        
        logger.info(f"Enqueued task {task.id} with priority {task.priority}")
    
    async def dequeue(self) -> Optional[Task]:
        """Get next task from queue"""
        # Check scheduled tasks first
        await self._process_scheduled_tasks()
        
        # Get highest priority task
        for priority in [TaskPriority.URGENT, TaskPriority.HIGH, TaskPriority.NORMAL, TaskPriority.LOW]:
            if self.priority_queues[priority]:
                task = self.priority_queues[priority].pop(0)
                self.processing_tasks[task.id] = task
                return task
        
        return None
    
    async def complete_task(self, task_id: str) -> None:
        """Mark task as completed"""
        if task_id in self.processing_tasks:
            task = self.processing_tasks.pop(task_id)
            
            # Remove from Redis if available
            if self.redis_client:
                await self._remove_task(task_id)
            
            logger.info(f"Completed task {task_id}")
    
    async def fail_task(self, task_id: str, error: str) -> None:
        """Mark task as failed"""
        if task_id in self.processing_tasks:
            task = self.processing_tasks.pop(task_id)
            task.error = error
            
            # Could implement retry logic here
            logger.error(f"Failed task {task_id}: {error}")
    
    async def _process_scheduled_tasks(self) -> None:
        """Move scheduled tasks to priority queues when ready"""
        now = datetime.utcnow()
        ready_tasks = []
        
        for schedule_time, task in self.scheduled_tasks:
            if schedule_time <= now:
                ready_tasks.append((schedule_time, task))
            else:
                break  # List is sorted, so we can break early
        
        # Remove ready tasks from scheduled list and add to priority queues
        for schedule_time, task in ready_tasks:
            self.scheduled_tasks.remove((schedule_time, task))
            self.priority_queues[task.priority].append(task)
    
    async def _persist_task(self, task: Task) -> None:
        """Persist task to Redis"""
        try:
            key = f"task:{task.id}"
            await self.redis_client.hset(key, mapping=task.dict())
            await self.redis_client.expire(key, 86400)  # 24 hours
        except Exception as e:
            logger.error(f"Failed to persist task {task.id}: {e}")
    
    async def _remove_task(self, task_id: str) -> None:
        """Remove task from Redis"""
        try:
            await self.redis_client.delete(f"task:{task_id}")
        except Exception as e:
            logger.error(f"Failed to remove task {task_id}: {e}")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status"""
        return {
            "urgent": len(self.priority_queues[TaskPriority.URGENT]),
            "high": len(self.priority_queues[TaskPriority.HIGH]),
            "normal": len(self.priority_queues[TaskPriority.NORMAL]),
            "low": len(self.priority_queues[TaskPriority.LOW]),
            "scheduled": len(self.scheduled_tasks),
            "processing": len(self.processing_tasks),
            "total_pending": sum(len(queue) for queue in self.priority_queues.values()) + len(self.scheduled_tasks)
        }


class WorkflowEngine:
    """Workflow execution engine"""
    
    def __init__(self, agent_manager=None):
        self.agent_manager = agent_manager
        self.workflows: Dict[str, Workflow] = {}
        self.running_workflows: Dict[str, asyncio.Task] = {}
        
    async def create_workflow(self, workflow: Workflow) -> str:
        """Create a new workflow"""
        self.workflows[workflow.id] = workflow
        logger.info(f"Created workflow {workflow.name} with ID {workflow.id}")
        return workflow.id
    
    async def start_workflow(self, workflow_id: str) -> bool:
        """Start workflow execution"""
        if workflow_id not in self.workflows:
            return False
        
        workflow = self.workflows[workflow_id]
        if workflow.status != WorkflowStatus.PENDING:
            return False
        
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.utcnow()
        
        # Start workflow execution task
        execution_task = asyncio.create_task(self._execute_workflow(workflow))
        self.running_workflows[workflow_id] = execution_task
        
        logger.info(f"Started workflow {workflow.name}")
        return True
    
    async def pause_workflow(self, workflow_id: str) -> bool:
        """Pause workflow execution"""
        if workflow_id not in self.workflows:
            return False
        
        workflow = self.workflows[workflow_id]
        workflow.status = WorkflowStatus.PAUSED
        
        # Cancel execution task
        if workflow_id in self.running_workflows:
            self.running_workflows[workflow_id].cancel()
            del self.running_workflows[workflow_id]
        
        logger.info(f"Paused workflow {workflow.name}")
        return True
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel workflow execution"""
        if workflow_id not in self.workflows:
            return False
        
        workflow = self.workflows[workflow_id]
        workflow.status = WorkflowStatus.CANCELLED
        workflow.completed_at = datetime.utcnow()
        
        # Cancel execution task
        if workflow_id in self.running_workflows:
            self.running_workflows[workflow_id].cancel()
            del self.running_workflows[workflow_id]
        
        logger.info(f"Cancelled workflow {workflow.name}")
        return True
    
    async def _execute_workflow(self, workflow: Workflow) -> None:
        """Execute workflow steps"""
        try:
            logger.info(f"Executing workflow {workflow.name}")
            
            # Build dependency graph
            dependency_graph = self._build_dependency_graph(workflow.steps)
            
            # Execute steps based on dependencies
            completed_steps = set()
            
            while len(completed_steps) < len(workflow.steps):
                # Find steps that can be executed
                ready_steps = []
                
                for step in workflow.steps:
                    if step.id in completed_steps:
                        continue
                    
                    if step.status != WorkflowStatus.PENDING:
                        continue
                    
                    # Check if all dependencies are satisfied
                    dependencies_satisfied = True
                    for dep in step.dependencies:
                        if dep.task_id not in completed_steps:
                            dependencies_satisfied = False
                            break
                    
                    if dependencies_satisfied:
                        ready_steps.append(step)
                
                if not ready_steps:
                    # Check if we're stuck
                    pending_steps = [s for s in workflow.steps if s.status == WorkflowStatus.PENDING]
                    if pending_steps:
                        raise Exception("Workflow stuck - circular dependencies or missing tasks")
                    break
                
                # Group steps by parallel group
                parallel_groups: Dict[Optional[str], List[WorkflowStep]] = {}
                for step in ready_steps:
                    group = step.parallel_group
                    if group not in parallel_groups:
                        parallel_groups[group] = []
                    parallel_groups[group].append(step)
                
                # Execute parallel groups
                for group, steps in parallel_groups.items():
                    if group is None:
                        # Execute sequentially
                        for step in steps:
                            await self._execute_step(step)
                            completed_steps.add(step.id)
                    else:
                        # Execute in parallel
                        tasks = [self._execute_step(step) for step in steps]
                        await asyncio.gather(*tasks)
                        for step in steps:
                            completed_steps.add(step.id)
            
            # Check final status
            failed_steps = [s for s in workflow.steps if s.status == WorkflowStatus.FAILED]
            if failed_steps:
                workflow.status = WorkflowStatus.FAILED
            else:
                workflow.status = WorkflowStatus.COMPLETED
            
            workflow.completed_at = datetime.utcnow()
            
            logger.info(f"Workflow {workflow.name} completed with status {workflow.status}")
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            workflow.completed_at = datetime.utcnow()
            logger.error(f"Workflow {workflow.name} failed: {e}")
        
        finally:
            # Clean up
            if workflow.id in self.running_workflows:
                del self.running_workflows[workflow.id]
    
    async def _execute_step(self, step: WorkflowStep) -> None:
        """Execute a single workflow step"""
        try:
            step.status = WorkflowStatus.RUNNING
            step.started_at = datetime.utcnow()
            
            logger.info(f"Executing workflow step {step.name}")
            
            # Submit task to agent manager
            if self.agent_manager:
                task_id = await self.agent_manager.submit_task(step.task)
                
                # Wait for task completion with timeout
                timeout = step.timeout_seconds or 300
                start_time = datetime.utcnow()
                
                while True:
                    # Check if task is completed
                    completed_task = None
                    for task in self.agent_manager.completed_tasks:
                        if task.id == task_id:
                            completed_task = task
                            break
                    
                    if completed_task:
                        if completed_task.status == "completed":
                            step.status = WorkflowStatus.COMPLETED
                        else:
                            step.status = WorkflowStatus.FAILED
                            step.error = completed_task.error
                        break
                    
                    # Check timeout
                    if (datetime.utcnow() - start_time).total_seconds() > timeout:
                        step.status = WorkflowStatus.FAILED
                        step.error = "Task timeout"
                        break
                    
                    await asyncio.sleep(1)
            else:
                # No agent manager, mark as completed
                step.status = WorkflowStatus.COMPLETED
            
            step.completed_at = datetime.utcnow()
            
        except Exception as e:
            step.status = WorkflowStatus.FAILED
            step.error = str(e)
            step.completed_at = datetime.utcnow()
            logger.error(f"Workflow step {step.name} failed: {e}")
    
    def _build_dependency_graph(self, steps: List[WorkflowStep]) -> Dict[str, List[str]]:
        """Build dependency graph for workflow steps"""
        graph = {}
        
        for step in steps:
            graph[step.id] = [dep.task_id for dep in step.dependencies]
        
        return graph
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get workflow by ID"""
        return self.workflows.get(workflow_id)
    
    def list_workflows(self) -> List[Workflow]:
        """List all workflows"""
        return list(self.workflows.values())
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow status"""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return None
        
        return {
            "id": workflow.id,
            "name": workflow.name,
            "status": workflow.status,
            "created_at": workflow.created_at.isoformat(),
            "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
            "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
            "total_steps": len(workflow.steps),
            "completed_steps": len([s for s in workflow.steps if s.status == WorkflowStatus.COMPLETED]),
            "failed_steps": len([s for s in workflow.steps if s.status == WorkflowStatus.FAILED]),
            "running_steps": len([s for s in workflow.steps if s.status == WorkflowStatus.RUNNING])
        }
