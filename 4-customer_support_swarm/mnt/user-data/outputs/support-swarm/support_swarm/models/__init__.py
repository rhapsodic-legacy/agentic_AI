"""
Customer Support Swarm - Data Models

Models for customers, orders, tickets, and support context.
"""

from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import json


class TicketStatus(Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    PENDING_CUSTOMER = "pending_customer"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"


class TicketPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class QueryCategory(Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    SALES = "sales"
    REFUND = "refund"
    GENERAL = "general"
    ACCOUNT = "account"
    CANCELLATION = "cancellation"


class SentimentLevel(Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"


@dataclass
class Customer:
    """Customer information."""
    id: str
    name: str
    email: str
    phone: Optional[str] = None
    
    # Account info
    account_type: str = "standard"  # standard, premium, enterprise
    created_at: str = ""
    
    # Subscription
    subscription_plan: Optional[str] = None
    subscription_status: str = "active"  # active, cancelled, suspended
    billing_cycle: str = "monthly"
    next_billing_date: Optional[str] = None
    
    # Stats
    total_spent: float = 0.0
    lifetime_orders: int = 0
    support_tickets: int = 0
    
    # Preferences
    language: str = "en"
    timezone: str = "UTC"
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "account_type": self.account_type,
            "subscription_plan": self.subscription_plan,
            "subscription_status": self.subscription_status,
            "total_spent": self.total_spent,
            "lifetime_orders": self.lifetime_orders,
        }


@dataclass
class Order:
    """Order information."""
    id: str
    customer_id: str
    
    # Order details
    items: list[dict] = field(default_factory=list)
    subtotal: float = 0.0
    tax: float = 0.0
    total: float = 0.0
    
    # Status
    status: str = "pending"  # pending, confirmed, processing, shipped, delivered, cancelled
    payment_status: str = "pending"  # pending, paid, failed, refunded
    
    # Dates
    created_at: str = ""
    shipped_at: Optional[str] = None
    delivered_at: Optional[str] = None
    
    # Shipping
    tracking_number: Optional[str] = None
    shipping_address: Optional[dict] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "items": self.items,
            "total": self.total,
            "status": self.status,
            "payment_status": self.payment_status,
            "created_at": self.created_at,
            "tracking_number": self.tracking_number,
        }


@dataclass
class Invoice:
    """Invoice information."""
    id: str
    customer_id: str
    order_id: Optional[str] = None
    
    # Amount
    amount: float = 0.0
    tax: float = 0.0
    total: float = 0.0
    
    # Status
    status: str = "pending"  # pending, paid, overdue, cancelled
    
    # Dates
    created_at: str = ""
    due_date: str = ""
    paid_at: Optional[str] = None
    
    # Payment
    payment_method: Optional[str] = None
    transaction_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "amount": self.amount,
            "total": self.total,
            "status": self.status,
            "created_at": self.created_at,
            "due_date": self.due_date,
        }


@dataclass
class Refund:
    """Refund information."""
    id: str
    customer_id: str
    order_id: Optional[str] = None
    invoice_id: Optional[str] = None
    
    # Amount
    amount: float = 0.0
    reason: str = ""
    
    # Status
    status: str = "pending"  # pending, approved, processed, rejected
    
    # Dates
    created_at: str = ""
    processed_at: Optional[str] = None
    
    # Reference
    reference_number: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "amount": self.amount,
            "reason": self.reason,
            "status": self.status,
            "reference_number": self.reference_number,
        }


@dataclass
class Ticket:
    """Support ticket."""
    id: str
    customer_id: str
    
    # Details
    subject: str = ""
    description: str = ""
    category: QueryCategory = QueryCategory.GENERAL
    
    # Status
    status: TicketStatus = TicketStatus.OPEN
    priority: TicketPriority = TicketPriority.MEDIUM
    
    # Assignment
    assigned_agent: Optional[str] = None
    escalated_to: Optional[str] = None
    
    # Dates
    created_at: str = ""
    updated_at: str = ""
    resolved_at: Optional[str] = None
    
    # Conversation
    messages: list[dict] = field(default_factory=list)
    
    # Resolution
    resolution: Optional[str] = None
    satisfaction_rating: Optional[int] = None
    
    # Metadata
    tags: list[str] = field(default_factory=list)
    related_order_id: Optional[str] = None
    related_invoice_id: Optional[str] = None
    
    def add_message(self, role: str, content: str, agent: Optional[str] = None):
        """Add a message to the ticket."""
        self.messages.append({
            "role": role,
            "content": content,
            "agent": agent,
            "timestamp": datetime.now().isoformat(),
        })
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "subject": self.subject,
            "category": self.category.value,
            "status": self.status.value,
            "priority": self.priority.value,
            "assigned_agent": self.assigned_agent,
            "created_at": self.created_at,
            "message_count": len(self.messages),
        }


@dataclass
class SupportContext:
    """
    Context that flows between agents during handoffs.
    
    Preserves all relevant information for seamless handoffs.
    """
    # Session
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Customer
    customer: Optional[Customer] = None
    customer_id: Optional[str] = None
    
    # Current state
    current_agent: str = "triage"
    previous_agents: list[str] = field(default_factory=list)
    handoff_count: int = 0
    
    # Query analysis
    original_query: str = ""
    query_category: Optional[QueryCategory] = None
    sentiment: SentimentLevel = SentimentLevel.NEUTRAL
    
    # Ticket
    ticket: Optional[Ticket] = None
    
    # Related data
    related_orders: list[Order] = field(default_factory=list)
    related_invoices: list[Invoice] = field(default_factory=list)
    related_refunds: list[Refund] = field(default_factory=list)
    
    # Conversation
    conversation_history: list[dict] = field(default_factory=list)
    
    # Flags
    is_escalated: bool = False
    needs_human: bool = False
    is_vip: bool = False
    
    # Notes from agents
    internal_notes: list[str] = field(default_factory=list)
    
    # Actions taken
    actions_taken: list[dict] = field(default_factory=list)
    
    def record_handoff(self, from_agent: str, to_agent: str, reason: str):
        """Record a handoff between agents."""
        self.previous_agents.append(from_agent)
        self.current_agent = to_agent
        self.handoff_count += 1
        self.internal_notes.append(f"Handoff from {from_agent} to {to_agent}: {reason}")
    
    def add_action(self, action_type: str, details: dict):
        """Record an action taken."""
        self.actions_taken.append({
            "type": action_type,
            "details": details,
            "agent": self.current_agent,
            "timestamp": datetime.now().isoformat(),
        })
    
    def add_message(self, role: str, content: str):
        """Add to conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "agent": self.current_agent,
            "timestamp": datetime.now().isoformat(),
        })
    
    def get_summary(self) -> str:
        """Get a summary of the context for agent handoffs."""
        summary = f"Session: {self.session_id[:8]}\n"
        
        if self.customer:
            summary += f"Customer: {self.customer.name} ({self.customer.email})\n"
            summary += f"Account: {self.customer.account_type}, Plan: {self.customer.subscription_plan}\n"
        
        summary += f"Category: {self.query_category.value if self.query_category else 'unknown'}\n"
        summary += f"Sentiment: {self.sentiment.value}\n"
        summary += f"Agents: {' → '.join(self.previous_agents + [self.current_agent])}\n"
        
        if self.related_orders:
            summary += f"Related Orders: {len(self.related_orders)}\n"
        
        if self.internal_notes:
            summary += f"Notes: {'; '.join(self.internal_notes[-3:])}\n"
        
        return summary
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "customer_id": self.customer_id,
            "current_agent": self.current_agent,
            "query_category": self.query_category.value if self.query_category else None,
            "sentiment": self.sentiment.value,
            "handoff_count": self.handoff_count,
            "is_escalated": self.is_escalated,
            "actions_count": len(self.actions_taken),
        }


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID."""
    short_uuid = str(uuid.uuid4())[:8].upper()
    return f"{prefix}{short_uuid}" if prefix else short_uuid
