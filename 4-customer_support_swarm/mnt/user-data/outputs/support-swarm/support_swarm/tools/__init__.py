"""
Customer Support Swarm - Tools and Database

Mock database and tools for:
- Customer lookup
- Order management
- Invoice handling
- Refund processing
- Ticket management
- Knowledge base search
"""

from typing import Optional
from datetime import datetime, timedelta
import random
import json

from .models import (
    Customer, Order, Invoice, Refund, Ticket,
    SupportContext, QueryCategory, TicketStatus, TicketPriority,
    SentimentLevel, generate_id
)


# =============================================================================
# Mock Database
# =============================================================================

class MockDatabase:
    """
    Mock database for demo purposes.
    In production, replace with real database calls.
    """
    
    def __init__(self):
        self._customers = {}
        self._orders = {}
        self._invoices = {}
        self._refunds = {}
        self._tickets = {}
        self._knowledge_base = {}
        
        # Initialize with sample data
        self._seed_data()
    
    def _seed_data(self):
        """Create sample data."""
        # Sample customers
        customers = [
            Customer(
                id="CUST-001",
                name="John Smith",
                email="john.smith@email.com",
                phone="+1-555-0101",
                account_type="premium",
                subscription_plan="Pro Monthly",
                subscription_status="active",
                billing_cycle="monthly",
                next_billing_date=(datetime.now() + timedelta(days=15)).strftime("%Y-%m-%d"),
                total_spent=599.94,
                lifetime_orders=6,
                created_at="2023-06-15",
            ),
            Customer(
                id="CUST-002",
                name="Sarah Johnson",
                email="sarah.j@email.com",
                phone="+1-555-0102",
                account_type="enterprise",
                subscription_plan="Enterprise Annual",
                subscription_status="active",
                billing_cycle="annual",
                next_billing_date=(datetime.now() + timedelta(days=180)).strftime("%Y-%m-%d"),
                total_spent=4999.99,
                lifetime_orders=1,
                created_at="2023-01-10",
            ),
            Customer(
                id="CUST-003",
                name="Mike Wilson",
                email="mike.w@email.com",
                account_type="standard",
                subscription_plan="Basic Monthly",
                subscription_status="active",
                billing_cycle="monthly",
                next_billing_date=(datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d"),
                total_spent=149.95,
                lifetime_orders=5,
                created_at="2024-01-20",
            ),
        ]
        
        for customer in customers:
            self._customers[customer.id] = customer
            self._customers[customer.email.lower()] = customer
        
        # Sample orders
        orders = [
            Order(
                id="ORD-10001",
                customer_id="CUST-001",
                items=[
                    {"name": "Pro Plan - Monthly", "quantity": 1, "price": 29.99},
                ],
                subtotal=29.99,
                tax=2.40,
                total=32.39,
                status="confirmed",
                payment_status="paid",
                created_at="2024-01-15",
            ),
            Order(
                id="ORD-10002",
                customer_id="CUST-001",
                items=[
                    {"name": "Pro Plan - Monthly", "quantity": 1, "price": 29.99},
                    {"name": "Pro Plan - Monthly", "quantity": 1, "price": 29.99},  # Duplicate!
                ],
                subtotal=59.98,
                tax=4.80,
                total=64.78,
                status="confirmed",
                payment_status="paid",
                created_at="2024-01-15",
            ),
            Order(
                id="ORD-10003",
                customer_id="CUST-003",
                items=[
                    {"name": "Basic Plan - Monthly", "quantity": 1, "price": 9.99},
                ],
                subtotal=9.99,
                tax=0.80,
                total=10.79,
                status="confirmed",
                payment_status="paid",
                created_at="2024-01-20",
            ),
        ]
        
        for order in orders:
            self._orders[order.id] = order
        
        # Sample invoices
        invoices = [
            Invoice(
                id="INV-2024-001",
                customer_id="CUST-001",
                order_id="ORD-10001",
                amount=29.99,
                tax=2.40,
                total=32.39,
                status="paid",
                created_at="2024-01-15",
                due_date="2024-01-30",
                paid_at="2024-01-15",
                payment_method="credit_card",
                transaction_id="TXN-ABC123",
            ),
            Invoice(
                id="INV-2024-002",
                customer_id="CUST-001",
                order_id="ORD-10002",
                amount=59.98,
                tax=4.80,
                total=64.78,
                status="paid",
                created_at="2024-01-15",
                due_date="2024-01-30",
                paid_at="2024-01-15",
                payment_method="credit_card",
                transaction_id="TXN-DEF456",
            ),
        ]
        
        for invoice in invoices:
            self._invoices[invoice.id] = invoice
        
        # Knowledge base articles
        self._knowledge_base = {
            "billing": [
                {
                    "id": "KB-001",
                    "title": "Understanding Your Bill",
                    "content": "Your bill includes your subscription fee plus any applicable taxes. Bills are generated on your billing cycle date.",
                    "tags": ["billing", "invoice", "charges"],
                },
                {
                    "id": "KB-002",
                    "title": "Payment Methods",
                    "content": "We accept Visa, Mastercard, American Express, and PayPal. You can update your payment method in Account Settings.",
                    "tags": ["payment", "billing", "credit card"],
                },
                {
                    "id": "KB-003",
                    "title": "Refund Policy",
                    "content": "Refunds are processed within 5-7 business days. Pro-rated refunds are available for annual subscriptions.",
                    "tags": ["refund", "cancellation", "money back"],
                },
            ],
            "technical": [
                {
                    "id": "KB-010",
                    "title": "Login Issues",
                    "content": "If you can't log in, try resetting your password. Clear your browser cache and cookies. Try a different browser.",
                    "tags": ["login", "password", "access"],
                },
                {
                    "id": "KB-011",
                    "title": "API Rate Limits",
                    "content": "Basic: 100 req/min, Pro: 1000 req/min, Enterprise: Unlimited. Rate limits reset every minute.",
                    "tags": ["api", "rate limit", "technical"],
                },
                {
                    "id": "KB-012",
                    "title": "Integration Guide",
                    "content": "To integrate our API, generate an API key in Settings > API. Use Bearer token authentication.",
                    "tags": ["api", "integration", "setup"],
                },
            ],
            "sales": [
                {
                    "id": "KB-020",
                    "title": "Plan Comparison",
                    "content": "Basic ($9.99/mo): 100 API calls. Pro ($29.99/mo): 1000 API calls + priority support. Enterprise (custom): Unlimited + dedicated support.",
                    "tags": ["pricing", "plans", "comparison"],
                },
                {
                    "id": "KB-021",
                    "title": "Enterprise Features",
                    "content": "Enterprise includes: SSO, custom SLA, dedicated account manager, 24/7 phone support, and custom integrations.",
                    "tags": ["enterprise", "features", "custom"],
                },
            ],
        }
    
    # Customer operations
    def get_customer(self, identifier: str) -> Optional[Customer]:
        """Get customer by ID or email."""
        return self._customers.get(identifier) or self._customers.get(identifier.lower())
    
    def update_customer(self, customer: Customer) -> bool:
        """Update customer."""
        if customer.id in self._customers:
            self._customers[customer.id] = customer
            self._customers[customer.email.lower()] = customer
            return True
        return False
    
    # Order operations
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)
    
    def get_customer_orders(self, customer_id: str) -> list[Order]:
        """Get all orders for a customer."""
        return [o for o in self._orders.values() if o.customer_id == customer_id]
    
    # Invoice operations
    def get_invoice(self, invoice_id: str) -> Optional[Invoice]:
        """Get invoice by ID."""
        return self._invoices.get(invoice_id)
    
    def get_customer_invoices(self, customer_id: str) -> list[Invoice]:
        """Get all invoices for a customer."""
        return [i for i in self._invoices.values() if i.customer_id == customer_id]
    
    # Refund operations
    def create_refund(self, refund: Refund) -> Refund:
        """Create a refund."""
        refund.reference_number = f"REF-{datetime.now().strftime('%Y')}-{generate_id()}"
        self._refunds[refund.id] = refund
        return refund
    
    def get_refund(self, refund_id: str) -> Optional[Refund]:
        """Get refund by ID."""
        return self._refunds.get(refund_id)
    
    # Ticket operations
    def create_ticket(self, ticket: Ticket) -> Ticket:
        """Create a support ticket."""
        self._tickets[ticket.id] = ticket
        return ticket
    
    def get_ticket(self, ticket_id: str) -> Optional[Ticket]:
        """Get ticket by ID."""
        return self._tickets.get(ticket_id)
    
    def update_ticket(self, ticket: Ticket) -> bool:
        """Update ticket."""
        if ticket.id in self._tickets:
            self._tickets[ticket.id] = ticket
            return True
        return False
    
    # Knowledge base
    def search_knowledge_base(self, query: str, category: str = None) -> list[dict]:
        """Search knowledge base articles."""
        results = []
        query_lower = query.lower()
        
        categories = [category] if category else self._knowledge_base.keys()
        
        for cat in categories:
            if cat not in self._knowledge_base:
                continue
            
            for article in self._knowledge_base[cat]:
                # Simple keyword matching
                if (query_lower in article["title"].lower() or
                    query_lower in article["content"].lower() or
                    any(query_lower in tag for tag in article["tags"])):
                    results.append(article)
        
        return results[:5]  # Return top 5


# Global database instance
db = MockDatabase()


# =============================================================================
# Tool Functions (for Swarm agents)
# =============================================================================

def lookup_customer(identifier: str) -> str:
    """
    Look up a customer by email or customer ID.
    
    Args:
        identifier: Customer email or ID (e.g., "john@email.com" or "CUST-001")
    
    Returns:
        Customer information or "not found" message
    """
    customer = db.get_customer(identifier)
    
    if customer:
        return f"""Customer Found:
- ID: {customer.id}
- Name: {customer.name}
- Email: {customer.email}
- Account Type: {customer.account_type}
- Subscription: {customer.subscription_plan} ({customer.subscription_status})
- Next Billing: {customer.next_billing_date}
- Total Spent: ${customer.total_spent:.2f}
- Lifetime Orders: {customer.lifetime_orders}"""
    
    return f"Customer not found with identifier: {identifier}"


def lookup_orders(customer_id: str) -> str:
    """
    Look up orders for a customer.
    
    Args:
        customer_id: Customer ID (e.g., "CUST-001")
    
    Returns:
        List of orders or "no orders" message
    """
    orders = db.get_customer_orders(customer_id)
    
    if not orders:
        return f"No orders found for customer {customer_id}"
    
    result = f"Orders for {customer_id}:\n"
    for order in orders:
        result += f"""
- Order {order.id}:
  Date: {order.created_at}
  Items: {', '.join(item['name'] for item in order.items)}
  Total: ${order.total:.2f}
  Status: {order.status}
  Payment: {order.payment_status}
"""
    
    return result


def lookup_invoices(customer_id: str) -> str:
    """
    Look up invoices for a customer.
    
    Args:
        customer_id: Customer ID (e.g., "CUST-001")
    
    Returns:
        List of invoices
    """
    invoices = db.get_customer_invoices(customer_id)
    
    if not invoices:
        return f"No invoices found for customer {customer_id}"
    
    result = f"Invoices for {customer_id}:\n"
    for inv in invoices:
        result += f"""
- Invoice {inv.id}:
  Date: {inv.created_at}
  Amount: ${inv.total:.2f}
  Status: {inv.status}
  Due: {inv.due_date}
  {"Paid: " + inv.paid_at if inv.paid_at else ""}
"""
    
    return result


def check_duplicate_charges(customer_id: str) -> str:
    """
    Check for duplicate charges on a customer's account.
    
    Args:
        customer_id: Customer ID
    
    Returns:
        Information about any duplicate charges found
    """
    invoices = db.get_customer_invoices(customer_id)
    
    # Group by date
    by_date = {}
    for inv in invoices:
        date = inv.created_at
        if date not in by_date:
            by_date[date] = []
        by_date[date].append(inv)
    
    duplicates = []
    for date, invs in by_date.items():
        if len(invs) > 1:
            duplicates.append({
                "date": date,
                "invoices": invs,
                "total_charged": sum(i.total for i in invs),
            })
    
    if duplicates:
        result = "⚠️ Duplicate charges detected:\n"
        for dup in duplicates:
            result += f"\nDate: {dup['date']}\n"
            for inv in dup['invoices']:
                result += f"  - Invoice {inv.id}: ${inv.total:.2f}\n"
            result += f"Total charged: ${dup['total_charged']:.2f}\n"
        return result
    
    return "No duplicate charges found."


def process_refund(customer_id: str, amount: float, reason: str, invoice_id: str = None) -> str:
    """
    Process a refund for a customer.
    
    Args:
        customer_id: Customer ID
        amount: Refund amount
        reason: Reason for refund
        invoice_id: Optional invoice ID to refund
    
    Returns:
        Refund confirmation with reference number
    """
    refund = Refund(
        id=f"RFD-{generate_id()}",
        customer_id=customer_id,
        invoice_id=invoice_id,
        amount=amount,
        reason=reason,
        status="approved",
        created_at=datetime.now().isoformat(),
    )
    
    refund = db.create_refund(refund)
    
    return f"""✅ Refund Processed:
- Reference: {refund.reference_number}
- Amount: ${refund.amount:.2f}
- Reason: {refund.reason}
- Status: {refund.status}
- Estimated arrival: 3-5 business days

The customer will receive an email confirmation."""


def update_subscription(customer_id: str, new_plan: str) -> str:
    """
    Update a customer's subscription plan.
    
    Args:
        customer_id: Customer ID
        new_plan: New plan name
    
    Returns:
        Confirmation of subscription change
    """
    customer = db.get_customer(customer_id)
    
    if not customer:
        return f"Customer {customer_id} not found"
    
    old_plan = customer.subscription_plan
    customer.subscription_plan = new_plan
    db.update_customer(customer)
    
    return f"""✅ Subscription Updated:
- Customer: {customer.name}
- Previous Plan: {old_plan}
- New Plan: {new_plan}
- Effective: Immediately
- Next billing will reflect the new rate."""


def cancel_subscription(customer_id: str, reason: str) -> str:
    """
    Cancel a customer's subscription.
    
    Args:
        customer_id: Customer ID
        reason: Cancellation reason
    
    Returns:
        Cancellation confirmation
    """
    customer = db.get_customer(customer_id)
    
    if not customer:
        return f"Customer {customer_id} not found"
    
    customer.subscription_status = "cancelled"
    db.update_customer(customer)
    
    return f"""⚠️ Subscription Cancelled:
- Customer: {customer.name}
- Plan: {customer.subscription_plan}
- Status: Cancelled
- Reason: {reason}
- Access until: End of current billing period ({customer.next_billing_date})

A confirmation email has been sent."""


def create_support_ticket(
    customer_id: str,
    subject: str,
    description: str,
    category: str,
    priority: str = "medium"
) -> str:
    """
    Create a support ticket.
    
    Args:
        customer_id: Customer ID
        subject: Ticket subject
        description: Issue description
        category: Category (billing, technical, sales, general)
        priority: Priority (low, medium, high, urgent)
    
    Returns:
        Ticket confirmation
    """
    ticket = Ticket(
        id=f"TKT-{generate_id()}",
        customer_id=customer_id,
        subject=subject,
        description=description,
        category=QueryCategory(category),
        priority=TicketPriority(priority),
        status=TicketStatus.OPEN,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
    )
    
    ticket = db.create_ticket(ticket)
    
    return f"""📋 Ticket Created:
- Ticket ID: {ticket.id}
- Subject: {ticket.subject}
- Category: {ticket.category.value}
- Priority: {ticket.priority.value}
- Status: {ticket.status.value}

You can reference this ticket ID for follow-ups."""


def search_knowledge_base(query: str, category: str = None) -> str:
    """
    Search the knowledge base for relevant articles.
    
    Args:
        query: Search query
        category: Optional category filter (billing, technical, sales)
    
    Returns:
        Relevant knowledge base articles
    """
    articles = db.search_knowledge_base(query, category)
    
    if not articles:
        return "No relevant articles found in the knowledge base."
    
    result = "📚 Knowledge Base Results:\n"
    for article in articles:
        result += f"""
**{article['title']}** (ID: {article['id']})
{article['content'][:200]}...
Tags: {', '.join(article['tags'])}
"""
    
    return result


def escalate_to_human(customer_id: str, reason: str, priority: str = "high") -> str:
    """
    Escalate the issue to a human agent.
    
    Args:
        customer_id: Customer ID
        reason: Reason for escalation
        priority: Priority level
    
    Returns:
        Escalation confirmation
    """
    ticket = Ticket(
        id=f"ESC-{generate_id()}",
        customer_id=customer_id,
        subject=f"Escalation: {reason}",
        description=reason,
        category=QueryCategory.GENERAL,
        priority=TicketPriority(priority),
        status=TicketStatus.ESCALATED,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
    )
    
    db.create_ticket(ticket)
    
    return f"""🚨 Escalated to Human Agent:
- Escalation ID: {ticket.id}
- Reason: {reason}
- Priority: {priority}
- Status: Awaiting human agent

A specialist will review this within:
- Urgent: 15 minutes
- High: 1 hour
- Medium: 4 hours

The customer has been notified."""


def check_system_status() -> str:
    """
    Check the current system status.
    
    Returns:
        System status information
    """
    return """🟢 System Status: All Systems Operational

- API: ✅ Operational (99.9% uptime)
- Dashboard: ✅ Operational
- Authentication: ✅ Operational
- Payments: ✅ Operational
- Database: ✅ Operational

No incidents reported in the last 24 hours.
Last updated: Just now"""


def get_pricing_info() -> str:
    """
    Get current pricing information.
    
    Returns:
        Pricing details for all plans
    """
    return """💰 Current Pricing Plans:

**Basic Plan** - $9.99/month
- 100 API calls/month
- Email support
- Basic analytics

**Pro Plan** - $29.99/month (Most Popular!)
- 1,000 API calls/month
- Priority email support
- Advanced analytics
- Custom integrations

**Enterprise Plan** - Custom pricing
- Unlimited API calls
- 24/7 phone & email support
- Dedicated account manager
- Custom SLA
- SSO & advanced security

Annual billing: Save 20%

Contact sales for Enterprise quotes."""


def apply_discount(customer_id: str, discount_code: str) -> str:
    """
    Apply a discount code to customer account.
    
    Args:
        customer_id: Customer ID
        discount_code: Discount code
    
    Returns:
        Discount application result
    """
    # Mock discount codes
    discounts = {
        "SAVE20": {"percent": 20, "description": "20% off next renewal"},
        "LOYALTY10": {"percent": 10, "description": "Loyalty discount"},
        "WINBACK30": {"percent": 30, "description": "Win-back offer"},
    }
    
    if discount_code.upper() not in discounts:
        return f"Invalid discount code: {discount_code}"
    
    discount = discounts[discount_code.upper()]
    
    return f"""✅ Discount Applied:
- Code: {discount_code.upper()}
- Discount: {discount['percent']}% off
- Description: {discount['description']}
- Applied to: Customer {customer_id}

This will be applied to the next billing cycle."""
