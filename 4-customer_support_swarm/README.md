# 🎧 Customer Support Swarm

A swarm of specialized support agents using **OpenAI Swarm** pattern with seamless handoffs between agents based on query type.

![Swarm](https://img.shields.io/badge/Framework-OpenAI%20Swarm-blue)
![Architecture](https://img.shields.io/badge/Architecture-Flat%20with%20Handoffs-green)
![Agents](https://img.shields.io/badge/Agents-7-purple)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Intent Classification** | Auto-routes to the right specialist |
| 🔄 **Seamless Handoffs** | Context preserved across transfers |
| 😊 **Sentiment Analysis** | Escalates frustrated customers |
| 📚 **Knowledge Base** | Integrated FAQ and documentation |
| 🔍 **Account Lookup** | Customer, order, invoice data |
| 📋 **Ticket Creation** | Automatic issue tracking |
| 🚨 **Human Escalation** | Complex issues go to humans |
| 🌍 **Context Memory** | Full conversation history |

## 🏗️ Architecture

```
                         ┌─────────────┐
                         │   TRIAGE    │
                         │   AGENT     │
                         └──────┬──────┘
                                │
       ┌────────────────────────┼────────────────────────┐
       │                        │                        │
       ▼                        ▼                        ▼
┌─────────────┐          ┌─────────────┐          ┌─────────────┐
│   BILLING   │◄────────►│  TECHNICAL  │◄────────►│   SALES     │
│   AGENT     │          │   SUPPORT   │          │   AGENT     │
└──────┬──────┘          └──────┬──────┘          └──────┬──────┘
       │                        │                        │
       ▼                        ▼                        ▼
┌─────────────┐          ┌─────────────┐          ┌─────────────┐
│  REFUNDS    │          │  ESCALATION │          │  RETENTION  │
│  SPECIALIST │          │    AGENT    │          │   AGENT     │
└─────────────┘          └─────────────┘          └─────────────┘
```

## 🤖 Agents

| Agent | Role | Handoffs To |
|-------|------|-------------|
| 🎯 **Triage** | Routes queries | Billing, Technical, Sales, Retention |
| 💳 **Billing** | Payments & invoices | Refunds, Escalation |
| 🔧 **Technical** | Tech issues | Billing, Escalation |
| 💼 **Sales** | Upgrades & pricing | Billing, Retention |
| 💰 **Refunds** | Process refunds | Billing, Escalation |
| 🚨 **Escalation** | Complex issues | All agents |
| 🤝 **Retention** | Prevents churn | Billing, Sales, Escalation |

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt

# Set API key (optional - mock mode works without keys)
export OPENAI_API_KEY="your-key"
```

### CLI Usage

```bash
# Interactive chat (mock mode)
python main.py chat

# Chat with real LLM
python main.py chat --provider openai

# With customer context
python main.py chat --customer CUST-001

# Run demo
python main.py demo

# Web server
python main.py serve
```

### Python API

```python
from support_swarm import SupportSwarm, SwarmConfig

# Create swarm
swarm = SupportSwarm()

# Set customer context (optional)
swarm.set_customer("CUST-001")

# Chat
response = swarm.chat("I was charged twice for my subscription")
print(response)
# [Triage routes to Billing]
# "I can see you were charged twice on Jan 15th..."

# Continue conversation
response = swarm.chat("Can I get a refund?")
print(response)
# [Billing routes to Refunds]
# "I've initiated a refund of $29.99..."

# Check current agent
print(f"Agent: {swarm.get_current_agent()}")
# "Refunds Specialist"

# Get handoff count
print(f"Handoffs: {swarm.handoff_count}")
# 2
```

## 📁 Project Structure

```
support-swarm/
├── support_swarm/
│   ├── __init__.py           # Package exports
│   ├── swarm.py              # Main SupportSwarm orchestrator
│   ├── agents/
│   │   └── __init__.py       # Agent definitions & handoff functions
│   ├── models/
│   │   └── __init__.py       # Customer, Order, Ticket models
│   ├── tools/
│   │   └── __init__.py       # Mock DB & tool functions
│   └── knowledge/
├── api.py                     # FastAPI backend
├── frontend/
│   └── index.html            # React web UI
├── main.py                    # CLI
├── requirements.txt
└── README.md
```

## 🔧 Tool Functions

Each agent has access to specific tools:

### Customer Tools
```python
lookup_customer(identifier)     # Find by ID or email
lookup_orders(customer_id)      # Get order history
lookup_invoices(customer_id)    # Get invoices
```

### Billing Tools
```python
check_duplicate_charges(customer_id)  # Detect double charges
update_subscription(customer_id, plan) # Change plan
cancel_subscription(customer_id, reason)
```

### Refund Tools
```python
process_refund(customer_id, amount, reason)  # Issue refund
```

### Support Tools
```python
create_support_ticket(customer_id, subject, description, category)
search_knowledge_base(query, category)
escalate_to_human(customer_id, reason, priority)
check_system_status()
```

### Sales Tools
```python
get_pricing_info()
apply_discount(customer_id, discount_code)
```

## 💬 Handoff Example

```
Customer: "I was charged twice for my subscription"

[Triage Agent]: Classifying query... Category: BILLING
                Transferring to Billing Specialist.

[Billing Specialist]: "I can see two charges of $29.99 on Jan 15th.
                      Let me verify this is a duplicate...
                      Yes, I found a duplicate charge.
                      I'll transfer you to our Refunds Specialist."

[Refunds Specialist]: "I've processed a refund of $29.99.
                      Reference: REF-2024-ABC123
                      It will appear in 3-5 business days.
                      Is there anything else I can help with?"
```

## 📊 Context Preservation

The `SupportContext` object flows between agents:

```python
context = SupportContext(
    session_id="abc-123",
    customer=customer_object,
    current_agent="Billing Specialist",
    previous_agents=["Triage Agent"],
    query_category=QueryCategory.BILLING,
    sentiment=SentimentLevel.FRUSTRATED,
    related_orders=[...],
    internal_notes=["Customer has been charged twice"],
    actions_taken=[{"type": "lookup", "details": {...}}],
)
```

## 🎭 Sentiment Analysis

The system detects customer sentiment:

| Sentiment | Indicators | Action |
|-----------|------------|--------|
| **Positive** | "thank", "great", "love" | Continue normally |
| **Neutral** | Standard queries | Continue normally |
| **Frustrated** | "frustrated", "disappointed" | Extra care |
| **Angry** | "furious", "sue", "!!!" | Consider escalation |

## 🔒 Mock Database

For demo purposes, includes mock data:

**Customers:**
- CUST-001: John Smith (Premium, Pro Monthly)
- CUST-002: Sarah Johnson (Enterprise Annual)
- CUST-003: Mike Wilson (Standard, Basic Monthly)

**Sample Issues:**
- Duplicate charges on CUST-001's account
- Various orders and invoices

## ⚙️ Configuration

```python
from support_swarm import SupportSwarm, SwarmConfig

config = SwarmConfig(
    llm_provider="openai",    # "openai", "gemini", "anthropic", "mock"
    llm_model=None,           # Uses provider default
    temperature=0.3,
    max_handoffs=5,           # Prevent infinite loops
    debug=False,
    preserve_context=True,
)

swarm = SupportSwarm(config)
```

## 🌐 Web API

```bash
python main.py serve
```

Endpoints:
- `GET /api/status` - Current status
- `POST /api/chat` - Send message
- `POST /api/set-customer` - Set customer context
- `GET /api/customers` - List customers
- `POST /api/reset` - Reset conversation
- `WS /ws` - WebSocket for real-time chat

## 📈 Best Practices

1. **Set Customer Context** - Better responses with customer data
2. **Check Sentiment** - Escalate frustrated customers early
3. **Monitor Handoffs** - Too many handoffs = poor routing
4. **Use Knowledge Base** - Search before custom responses
5. **Create Tickets** - Track unresolved issues

## 🧪 Testing Without API Keys

The system works in mock mode without API keys:

```bash
python main.py chat  # Uses mock provider by default
```

Mock mode simulates:
- Intent classification
- Appropriate handoffs
- Tool execution
- Realistic responses

## 📝 License

MIT License
