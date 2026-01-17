"""
Customer Support Swarm - Agent Definitions

Defines all support agents with their instructions and handoff functions.
Uses OpenAI Swarm pattern for seamless agent transitions.
"""

from typing import Callable, Optional
from dataclasses import dataclass, field

from ..models import SupportContext, QueryCategory, SentimentLevel
from ..tools import (
    lookup_customer,
    lookup_orders,
    lookup_invoices,
    check_duplicate_charges,
    process_refund,
    update_subscription,
    cancel_subscription,
    create_support_ticket,
    search_knowledge_base,
    escalate_to_human,
    check_system_status,
    get_pricing_info,
    apply_discount,
)


@dataclass
class Agent:
    """
    Swarm Agent definition.
    
    Compatible with OpenAI Swarm framework pattern.
    """
    name: str
    instructions: str
    functions: list[Callable] = field(default_factory=list)
    model: str = "gpt-4o-mini"
    
    # For handoffs
    handoff_agents: dict = field(default_factory=dict)
    
    def add_handoff(self, name: str, agent: "Agent"):
        """Register a handoff target."""
        self.handoff_agents[name] = agent


@dataclass
class Response:
    """Agent response."""
    agent: Optional[Agent] = None
    messages: list = field(default_factory=list)
    context_variables: dict = field(default_factory=dict)


# =============================================================================
# Handoff Functions
# =============================================================================

def transfer_to_billing():
    """Transfer the conversation to the Billing Agent."""
    return billing_agent


def transfer_to_technical():
    """Transfer the conversation to Technical Support."""
    return technical_agent


def transfer_to_sales():
    """Transfer the conversation to the Sales Agent."""
    return sales_agent


def transfer_to_refunds():
    """Transfer the conversation to the Refunds Specialist."""
    return refunds_agent


def transfer_to_escalation():
    """Escalate to the Escalation Agent for complex issues."""
    return escalation_agent


def transfer_to_retention():
    """Transfer to Retention Agent for cancellation requests."""
    return retention_agent


def transfer_to_triage():
    """Transfer back to Triage for re-routing."""
    return triage_agent


# =============================================================================
# Intent Classification
# =============================================================================

def classify_intent(message: str) -> str:
    """
    Classify the customer's intent.
    
    Args:
        message: Customer message
    
    Returns:
        Classification result with recommended routing
    """
    message_lower = message.lower()
    
    # Billing keywords
    billing_keywords = ["bill", "invoice", "charge", "payment", "subscription", "price", "cost"]
    
    # Technical keywords
    technical_keywords = ["error", "bug", "not working", "crash", "slow", "api", "login", "password", "technical"]
    
    # Sales keywords
    sales_keywords = ["upgrade", "plan", "pricing", "enterprise", "discount", "buy", "purchase", "demo"]
    
    # Refund keywords
    refund_keywords = ["refund", "money back", "charged twice", "duplicate", "cancel charge"]
    
    # Cancellation keywords
    cancel_keywords = ["cancel", "stop", "terminate", "end subscription", "close account"]
    
    # Check for matches
    if any(kw in message_lower for kw in refund_keywords):
        return "Category: REFUND\nRecommended: Transfer to Billing Agent, who may escalate to Refunds Specialist"
    
    if any(kw in message_lower for kw in cancel_keywords):
        return "Category: CANCELLATION\nRecommended: Transfer to Retention Agent"
    
    if any(kw in message_lower for kw in billing_keywords):
        return "Category: BILLING\nRecommended: Transfer to Billing Agent"
    
    if any(kw in message_lower for kw in technical_keywords):
        return "Category: TECHNICAL\nRecommended: Transfer to Technical Support"
    
    if any(kw in message_lower for kw in sales_keywords):
        return "Category: SALES\nRecommended: Transfer to Sales Agent"
    
    return "Category: GENERAL\nRecommended: Handle directly or gather more information"


def analyze_sentiment(message: str) -> str:
    """
    Analyze customer sentiment.
    
    Args:
        message: Customer message
    
    Returns:
        Sentiment analysis result
    """
    message_lower = message.lower()
    
    # Angry indicators
    angry_indicators = ["furious", "outraged", "terrible", "worst", "sue", "lawyer", "fraud", "scam", "!!!"]
    
    # Frustrated indicators
    frustrated_indicators = ["frustrated", "annoyed", "disappointed", "unacceptable", "again", "still", "multiple times"]
    
    # Positive indicators
    positive_indicators = ["thank", "great", "love", "excellent", "happy", "appreciate"]
    
    sentiment = "NEUTRAL"
    escalation_needed = False
    
    if any(ind in message_lower for ind in angry_indicators) or message.count('!') > 2:
        sentiment = "ANGRY"
        escalation_needed = True
    elif any(ind in message_lower for ind in frustrated_indicators):
        sentiment = "FRUSTRATED"
        escalation_needed = message_lower.count("again") > 0 or "multiple" in message_lower
    elif any(ind in message_lower for ind in positive_indicators):
        sentiment = "POSITIVE"
    
    result = f"Sentiment: {sentiment}"
    if escalation_needed:
        result += "\n⚠️ Consider escalation due to customer frustration level"
    
    return result


# =============================================================================
# Agent Definitions
# =============================================================================

# Triage Agent - First point of contact
triage_agent = Agent(
    name="Triage Agent",
    instructions="""You are the first point of contact for customer support. Your role is to:

1. Greet the customer warmly
2. Classify their query type using the classify_intent function
3. Analyze their sentiment using analyze_sentiment function
4. Route them to the appropriate specialist

Categories and routing:
- BILLING issues → Transfer to Billing Agent
- TECHNICAL issues → Transfer to Technical Support
- SALES inquiries → Transfer to Sales Agent
- REFUND requests → Transfer to Billing Agent (they'll escalate to Refunds if needed)
- CANCELLATION → Transfer to Retention Agent
- GENERAL → Handle yourself or ask clarifying questions

Before transferring:
- Acknowledge the customer's concern
- Briefly explain you're connecting them with a specialist
- Never leave the customer hanging

If you can't classify the issue, ask clarifying questions.

Always be empathetic and professional.""",
    functions=[
        classify_intent,
        analyze_sentiment,
        lookup_customer,
        transfer_to_billing,
        transfer_to_technical,
        transfer_to_sales,
        transfer_to_retention,
    ],
)


# Billing Agent
billing_agent = Agent(
    name="Billing Specialist",
    instructions="""You are a Billing Specialist. You handle:

- Invoice questions and clarifications
- Payment issues and failures
- Subscription changes (upgrades/downgrades)
- Billing cycle questions
- Charge disputes

Available actions:
1. Look up customer information
2. Check invoices and orders
3. Check for duplicate charges
4. Update subscription plans
5. Transfer to Refunds Specialist for refund processing

When handling billing issues:
1. First, look up the customer's account
2. Review their recent invoices and orders
3. Identify any issues (duplicates, wrong charges)
4. Explain findings clearly
5. Take appropriate action or escalate

For refund requests:
- Verify the issue exists
- If refund is warranted, transfer to Refunds Specialist
- Explain you're connecting them with a specialist who can process the refund

Always be transparent about charges and policies.
If an issue is complex or the customer is frustrated, consider escalating.""",
    functions=[
        lookup_customer,
        lookup_orders,
        lookup_invoices,
        check_duplicate_charges,
        update_subscription,
        search_knowledge_base,
        create_support_ticket,
        transfer_to_refunds,
        transfer_to_escalation,
        transfer_to_triage,
    ],
)


# Technical Support Agent
technical_agent = Agent(
    name="Technical Support",
    instructions="""You are a Technical Support specialist. You handle:

- Login and authentication issues
- API problems and errors
- Integration troubleshooting
- Performance issues
- Bug reports
- Feature questions

Troubleshooting approach:
1. Understand the specific issue
2. Check system status first
3. Search knowledge base for solutions
4. Guide step-by-step troubleshooting
5. Create ticket if issue needs further investigation
6. Escalate if issue is critical or unresolved

Common solutions:
- Login issues: Password reset, clear cache, check email
- API errors: Check rate limits, verify API key, check endpoint
- Performance: Check system status, verify connection

When creating tickets:
- Include all relevant technical details
- Note troubleshooting steps already taken
- Set appropriate priority

For billing-related issues that come up during technical support, transfer to Billing.
For complex issues requiring engineering, escalate.""",
    functions=[
        lookup_customer,
        check_system_status,
        search_knowledge_base,
        create_support_ticket,
        escalate_to_human,
        transfer_to_billing,
        transfer_to_escalation,
        transfer_to_triage,
    ],
)


# Sales Agent
sales_agent = Agent(
    name="Sales Agent",
    instructions="""You are a Sales Agent. You handle:

- Plan inquiries and comparisons
- Upgrade requests
- Enterprise inquiries
- Pricing questions
- Discount requests
- Demo requests

Sales approach:
1. Understand customer's needs and current situation
2. Explain relevant plan features
3. Highlight value and benefits
4. Handle objections professionally
5. Guide upgrade process or schedule demo

When handling upgrades:
1. Look up current plan
2. Understand why they want to upgrade
3. Recommend appropriate plan
4. Explain pricing and benefits
5. Process upgrade or transfer to billing

For discounts:
- Check if customer qualifies for any promotions
- Apply discount codes when appropriate
- For special pricing, may need to escalate

For Enterprise inquiries:
- Gather requirements
- Explain enterprise features
- Schedule demo or call with sales team

Always be helpful, not pushy. Focus on solving their needs.""",
    functions=[
        lookup_customer,
        get_pricing_info,
        apply_discount,
        update_subscription,
        search_knowledge_base,
        create_support_ticket,
        transfer_to_billing,
        transfer_to_retention,
        transfer_to_triage,
    ],
)


# Refunds Specialist
refunds_agent = Agent(
    name="Refunds Specialist",
    instructions="""You are a Refunds Specialist. You have authority to process refunds.

You handle:
- Refund processing
- Duplicate charge corrections
- Billing error corrections
- Pro-rated refunds for cancellations

Refund policy:
- Full refunds available within 30 days
- Pro-rated refunds for annual plans
- Duplicate charges: Always refund
- Billing errors: Always correct

When processing refunds:
1. Verify the customer's identity
2. Confirm the issue (duplicate charge, etc.)
3. Calculate refund amount
4. Process the refund
5. Provide reference number
6. Explain timeline (3-5 business days)

For complex situations:
- Large refunds (>$500): May need approval
- Disputed charges: Create detailed ticket
- Repeated issues: Escalate to prevent future problems

Always be empathetic. Refund situations are often frustrating for customers.
Document everything for accounting purposes.""",
    functions=[
        lookup_customer,
        lookup_invoices,
        check_duplicate_charges,
        process_refund,
        create_support_ticket,
        transfer_to_billing,
        transfer_to_escalation,
        transfer_to_triage,
    ],
)


# Escalation Agent
escalation_agent = Agent(
    name="Escalation Agent",
    instructions="""You are the Escalation Agent. You handle complex or sensitive issues that other agents couldn't resolve.

You handle:
- Complex technical issues
- Frustrated or angry customers
- Policy exceptions
- VIP customers
- Issues requiring management approval

Approach:
1. Review the full conversation history
2. Acknowledge the customer's frustration
3. Take ownership of the issue
4. Find a resolution, even if unconventional
5. Follow up to ensure satisfaction

You have authority to:
- Offer extended refunds
- Apply special discounts
- Expedite requests
- Create VIP tickets
- Escalate to human management

De-escalation techniques:
- Validate their feelings
- Apologize sincerely for the experience
- Focus on solutions, not blame
- Offer something extra when appropriate
- Ensure they feel heard

If you cannot resolve:
- Create a high-priority ticket
- Escalate to human agent
- Provide clear timeline for resolution""",
    functions=[
        lookup_customer,
        lookup_orders,
        lookup_invoices,
        process_refund,
        apply_discount,
        escalate_to_human,
        create_support_ticket,
        transfer_to_billing,
        transfer_to_technical,
        transfer_to_triage,
    ],
)


# Retention Agent
retention_agent = Agent(
    name="Retention Agent",
    instructions="""You are a Retention Specialist. Your goal is to retain customers who want to cancel.

You handle:
- Cancellation requests
- Downgrade requests
- Customer complaints leading to churn
- Win-back opportunities

Retention approach:
1. Understand WHY they want to cancel
2. Address their specific concern
3. Offer alternatives to full cancellation
4. Present retention offers if appropriate
5. If they insist, process cancellation gracefully

Common reasons and responses:
- Too expensive → Offer discount or downgrade
- Not using it → Pause subscription option
- Missing features → Explain roadmap, gather feedback
- Bad experience → Apologize, offer to fix, provide credit
- Switching to competitor → Understand what's better, match if possible

Retention offers available:
- LOYALTY10: 10% off for loyal customers
- SAVE20: 20% off next 3 months
- WINBACK30: 30% for returning customers

If customer insists on canceling:
- Process it professionally
- Offer to downgrade instead of full cancel
- Leave door open for return
- Create feedback ticket

Never make the customer feel guilty or pressured.
A graceful goodbye is better than a frustrated ex-customer.""",
    functions=[
        lookup_customer,
        lookup_orders,
        apply_discount,
        update_subscription,
        cancel_subscription,
        create_support_ticket,
        transfer_to_billing,
        transfer_to_sales,
        transfer_to_escalation,
        transfer_to_triage,
    ],
)


# =============================================================================
# Register Handoffs
# =============================================================================

# Triage can hand off to all specialists
triage_agent.add_handoff("billing", billing_agent)
triage_agent.add_handoff("technical", technical_agent)
triage_agent.add_handoff("sales", sales_agent)
triage_agent.add_handoff("retention", retention_agent)

# Billing can hand off to refunds and escalation
billing_agent.add_handoff("refunds", refunds_agent)
billing_agent.add_handoff("escalation", escalation_agent)
billing_agent.add_handoff("triage", triage_agent)

# Technical can escalate
technical_agent.add_handoff("billing", billing_agent)
technical_agent.add_handoff("escalation", escalation_agent)
technical_agent.add_handoff("triage", triage_agent)

# Sales
sales_agent.add_handoff("billing", billing_agent)
sales_agent.add_handoff("retention", retention_agent)
sales_agent.add_handoff("triage", triage_agent)

# Refunds
refunds_agent.add_handoff("billing", billing_agent)
refunds_agent.add_handoff("escalation", escalation_agent)
refunds_agent.add_handoff("triage", triage_agent)

# Escalation can route anywhere
escalation_agent.add_handoff("billing", billing_agent)
escalation_agent.add_handoff("technical", technical_agent)
escalation_agent.add_handoff("triage", triage_agent)

# Retention
retention_agent.add_handoff("billing", billing_agent)
retention_agent.add_handoff("sales", sales_agent)
retention_agent.add_handoff("escalation", escalation_agent)
retention_agent.add_handoff("triage", triage_agent)


# =============================================================================
# All Agents Export
# =============================================================================

ALL_AGENTS = {
    "triage": triage_agent,
    "billing": billing_agent,
    "technical": technical_agent,
    "sales": sales_agent,
    "refunds": refunds_agent,
    "escalation": escalation_agent,
    "retention": retention_agent,
}


def get_agent(name: str) -> Agent:
    """Get an agent by name."""
    return ALL_AGENTS.get(name.lower())
