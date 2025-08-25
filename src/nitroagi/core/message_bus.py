"""Message bus for inter-module communication in NitroAGI."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import UUID, uuid4

from nitroagi.core.exceptions import MessageBusException


class MessagePriority(Enum):
    """Priority levels for messages."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class MessageType(Enum):
    """Types of messages that can be sent."""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    COMMAND = "command"
    NOTIFICATION = "notification"
    HEARTBEAT = "heartbeat"


@dataclass
class Message:
    """Message to be sent through the message bus."""
    id: UUID = field(default_factory=uuid4)
    type: MessageType = MessageType.REQUEST
    topic: str = ""
    sender: str = ""
    recipient: Optional[str] = None  # None means broadcast
    payload: Any = None
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[UUID] = None
    reply_to: Optional[str] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "id": str(self.id),
            "type": self.type.value,
            "topic": self.topic,
            "sender": self.sender,
            "recipient": self.recipient,
            "payload": self.payload,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": str(self.correlation_id) if self.correlation_id else None,
            "reply_to": self.reply_to,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata
        }
    
    def is_expired(self) -> bool:
        """Check if the message has expired."""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False


class MessageHandler:
    """Handler for processing messages."""
    
    def __init__(
        self,
        callback: Callable[[Message], Any],
        message_type: Optional[MessageType] = None,
        sender_filter: Optional[str] = None,
        priority_filter: Optional[MessagePriority] = None
    ):
        """Initialize message handler.
        
        Args:
            callback: Function to call when message matches filters
            message_type: Optional filter by message type
            sender_filter: Optional filter by sender
            priority_filter: Optional minimum priority filter
        """
        self.callback = callback
        self.message_type = message_type
        self.sender_filter = sender_filter
        self.priority_filter = priority_filter
    
    def matches(self, message: Message) -> bool:
        """Check if a message matches the handler's filters.
        
        Args:
            message: Message to check
            
        Returns:
            True if message matches all filters
        """
        if self.message_type and message.type != self.message_type:
            return False
        if self.sender_filter and message.sender != self.sender_filter:
            return False
        if self.priority_filter and message.priority.value < self.priority_filter.value:
            return False
        return True
    
    async def handle(self, message: Message) -> Any:
        """Handle a message.
        
        Args:
            message: Message to handle
            
        Returns:
            Result from the callback
        """
        if asyncio.iscoroutinefunction(self.callback):
            return await self.callback(message)
        else:
            return self.callback(message)


class MessageBus:
    """Central message bus for inter-module communication."""
    
    def __init__(self, max_queue_size: int = 1000):
        """Initialize the message bus.
        
        Args:
            max_queue_size: Maximum size for message queues
        """
        self.logger = logging.getLogger("nitroagi.core.message_bus")
        self._subscribers: Dict[str, List[MessageHandler]] = {}
        self._queues: Dict[str, asyncio.Queue] = {}
        self._workers: Dict[str, asyncio.Task] = {}
        self._max_queue_size = max_queue_size
        self._running = False
        self._metrics = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "messages_expired": 0,
            "messages_failed": 0,
        }
        self._message_history: List[Message] = []
        self._max_history_size = 100
    
    async def start(self) -> None:
        """Start the message bus."""
        self._running = True
        self.logger.info("Message bus started")
    
    async def stop(self) -> None:
        """Stop the message bus."""
        self._running = False
        
        # Cancel all workers
        for worker in self._workers.values():
            worker.cancel()
        
        # Wait for workers to finish
        if self._workers:
            await asyncio.gather(*self._workers.values(), return_exceptions=True)
        
        self.logger.info("Message bus stopped")
    
    def subscribe(
        self,
        topic: str,
        handler: MessageHandler,
        create_queue: bool = False
    ) -> None:
        """Subscribe to a topic.
        
        Args:
            topic: Topic to subscribe to
            handler: Handler for messages on this topic
            create_queue: If True, create a queue for async processing
        """
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        
        self._subscribers[topic].append(handler)
        
        if create_queue and topic not in self._queues:
            self._queues[topic] = asyncio.Queue(maxsize=self._max_queue_size)
            self._workers[topic] = asyncio.create_task(self._process_queue(topic))
        
        self.logger.debug(f"Subscribed handler to topic: {topic}")
    
    def unsubscribe(self, topic: str, handler: MessageHandler) -> None:
        """Unsubscribe from a topic.
        
        Args:
            topic: Topic to unsubscribe from
            handler: Handler to remove
        """
        if topic in self._subscribers:
            self._subscribers[topic].remove(handler)
            if not self._subscribers[topic]:
                del self._subscribers[topic]
                
                # Clean up queue and worker if no more subscribers
                if topic in self._queues:
                    del self._queues[topic]
                if topic in self._workers:
                    self._workers[topic].cancel()
                    del self._workers[topic]
        
        self.logger.debug(f"Unsubscribed handler from topic: {topic}")
    
    async def publish(self, message: Message) -> None:
        """Publish a message to a topic.
        
        Args:
            message: Message to publish
        """
        if not self._running:
            raise MessageBusException("Message bus is not running")
        
        # Check if message is expired
        if message.is_expired():
            self._metrics["messages_expired"] += 1
            self.logger.warning(f"Message {message.id} expired before sending")
            return
        
        self._metrics["messages_sent"] += 1
        self._add_to_history(message)
        
        # If specific recipient, only send to that recipient
        if message.recipient:
            await self._deliver_to_recipient(message)
        else:
            # Broadcast to all subscribers of the topic
            await self._broadcast_to_topic(message)
    
    async def _deliver_to_recipient(self, message: Message) -> None:
        """Deliver message to specific recipient.
        
        Args:
            message: Message to deliver
        """
        topic = f"{message.topic}.{message.recipient}"
        if topic in self._subscribers:
            await self._deliver_to_handlers(message, self._subscribers[topic])
    
    async def _broadcast_to_topic(self, message: Message) -> None:
        """Broadcast message to all subscribers of a topic.
        
        Args:
            message: Message to broadcast
        """
        # Check exact topic match
        if message.topic in self._subscribers:
            await self._deliver_to_handlers(message, self._subscribers[message.topic])
        
        # Check wildcard subscriptions
        for topic, handlers in self._subscribers.items():
            if self._matches_wildcard(message.topic, topic):
                await self._deliver_to_handlers(message, handlers)
    
    async def _deliver_to_handlers(
        self,
        message: Message,
        handlers: List[MessageHandler]
    ) -> None:
        """Deliver message to a list of handlers.
        
        Args:
            message: Message to deliver
            handlers: List of handlers to deliver to
        """
        for handler in handlers:
            if handler.matches(message):
                try:
                    await handler.handle(message)
                    self._metrics["messages_delivered"] += 1
                except Exception as e:
                    self._metrics["messages_failed"] += 1
                    self.logger.error(
                        f"Error handling message {message.id}: {e}",
                        exc_info=True
                    )
    
    async def _process_queue(self, topic: str) -> None:
        """Process messages from a queue.
        
        Args:
            topic: Topic queue to process
        """
        queue = self._queues[topic]
        
        while self._running:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=1.0)
                
                if message.is_expired():
                    self._metrics["messages_expired"] += 1
                    continue
                
                if topic in self._subscribers:
                    await self._deliver_to_handlers(message, self._subscribers[topic])
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing queue for topic {topic}: {e}")
    
    def _matches_wildcard(self, topic: str, pattern: str) -> bool:
        """Check if a topic matches a wildcard pattern.
        
        Args:
            topic: Topic to check
            pattern: Pattern to match against (supports * and #)
            
        Returns:
            True if topic matches pattern
        """
        if pattern == "#":
            return True
        
        topic_parts = topic.split(".")
        pattern_parts = pattern.split(".")
        
        if len(pattern_parts) > len(topic_parts):
            return False
        
        for i, pattern_part in enumerate(pattern_parts):
            if pattern_part == "#":
                return True
            if pattern_part == "*":
                continue
            if i >= len(topic_parts) or pattern_part != topic_parts[i]:
                return False
        
        return len(pattern_parts) == len(topic_parts)
    
    def _add_to_history(self, message: Message) -> None:
        """Add message to history.
        
        Args:
            message: Message to add to history
        """
        self._message_history.append(message)
        if len(self._message_history) > self._max_history_size:
            self._message_history.pop(0)
    
    async def request_response(
        self,
        request: Message,
        timeout: float = 30.0
    ) -> Optional[Message]:
        """Send a request and wait for a response.
        
        Args:
            request: Request message to send
            timeout: Timeout in seconds
            
        Returns:
            Response message if received, None if timeout
        """
        response_future: asyncio.Future = asyncio.Future()
        response_topic = f"response.{request.id}"
        
        # Create handler for response
        async def response_handler(response: Message) -> None:
            if response.correlation_id == request.id:
                response_future.set_result(response)
        
        handler = MessageHandler(response_handler, MessageType.RESPONSE)
        self.subscribe(response_topic, handler)
        
        try:
            # Send request
            request.reply_to = response_topic
            await self.publish(request)
            
            # Wait for response
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Request {request.id} timed out")
            return None
            
        finally:
            self.unsubscribe(response_topic, handler)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get message bus metrics.
        
        Returns:
            Dictionary containing metrics
        """
        return {
            **self._metrics,
            "active_topics": len(self._subscribers),
            "active_queues": len(self._queues),
            "history_size": len(self._message_history),
        }
    
    def get_message_history(
        self,
        topic: Optional[str] = None,
        limit: int = 10
    ) -> List[Message]:
        """Get message history.
        
        Args:
            topic: Optional topic filter
            limit: Maximum number of messages to return
            
        Returns:
            List of messages from history
        """
        history = self._message_history
        
        if topic:
            history = [m for m in history if m.topic == topic]
        
        return history[-limit:]