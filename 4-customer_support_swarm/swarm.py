"""
Customer Support Swarm - Main Orchestrator 

Implements the Swarm pattern for multi-agent customer support
with seamless handoffs between specialized agents.
"""

from typing import Optional, Callable, Any
from dataclasses import dataclass, field
import json
import os
import inspect

from .agents import (
    Agent, Response,
    triage_agent, billing_agent, technical_agent, sales_agent,
    refunds_agent, escalation_agent, retention_agent,
    ALL_AGENTS, get_agent,
)
from .models import SupportContext, QueryCategory, SentimentLevel
from .tools import db


@dataclass
class SwarmConfig:
    """Configuration for the Support Swarm."""
    
    # LLM Settings
    llm_provider: str = "openai"  # openai, gemini, anthropic
    llm_model: Optional[str] = None
    temperature: float = 0.3
    
    # Behavior
    max_handoffs: int = 5
    debug: bool = False
    
    # Context
    preserve_context: bool = True


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str  # user, assistant, system, tool
    content: str
    agent: Optional[str] = None
    tool_calls: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)


class SupportSwarm:
    """
    Customer Support Swarm Implementation.
    
    Manages a swarm of specialized support agents with seamless handoffs.
    
    Architecture:
        Triage → routes to specialists
        Billing ↔ Refunds
        Technical → Escalation
        Sales ↔ Retention
    
    Usage:
        swarm = SupportSwarm()
        response = swarm.chat("I was charged twice for my subscription")
        
        # Continue conversation
        response = swarm.chat("Can I get a refund?")
        
        # Get conversation history
        history = swarm.get_history()
    """
    
    def __init__(self, config: Optional[SwarmConfig] = None):
        self.config = config or SwarmConfig()
        
        # Initialize LLM
        self._init_llm()
        
        # State
        self.current_agent = triage_agent
        self.context = SupportContext()
        self.conversation_history: list[ConversationTurn] = []
        self.handoff_count = 0
    
    def _init_llm(self):
        """Initialize the LLM client."""
        if self.config.llm_provider == "openai":
            from openai import OpenAI
            self.client = OpenAI()
            self.model = self.config.llm_model or "gpt-4o-mini"
        
        elif self.config.llm_provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.client = ChatGoogleGenerativeAI(
                model=self.config.llm_model or "gemini-1.5-flash",
                temperature=self.config.temperature,
            )
            self.model = self.config.llm_model or "gemini-1.5-flash"
        
        elif self.config.llm_provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic()
            self.model = self.config.llm_model or "claude-sonnet-4-20250514"
        
        else:
            # Fallback - use mock for demo
            self.client = None
            self.model = "mock"
    
    def _get_function_schema(self, func: Callable) -> dict:
        """Get OpenAI function schema from a Python function."""
        sig = inspect.signature(func)
        doc = func.__doc__ or ""
        
        # Parse docstring for description and args
        description = doc.split("\n")[0].strip() if doc else func.__name__
        
        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        
        for name, param in sig.parameters.items():
            param_type = "string"  # Default type
            
            # Try to get type from annotation
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
            
            parameters["properties"][name] = {
                "type": param_type,
                "description": f"Parameter: {name}",
            }
            
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(name)
        
        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": parameters,
            }
        }
    
    def _call_llm(self, messages: list, tools: list = None) -> dict:
        """Call the LLM with messages and optional tools."""
        
        if self.config.llm_provider == "openai":
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.config.temperature,
            }
            
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            response = self.client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            
            return {
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments),
                    }
                    for tc in (message.tool_calls or [])
                ],
            }
        
        elif self.config.llm_provider == "anthropic":
            # Convert to Anthropic format
            system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
            conv_messages = [m for m in messages if m["role"] != "system"]
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_msg,
                messages=conv_messages,
            )
            
            return {
                "content": response.content[0].text,
                "tool_calls": [],  # Simplified for demo
            }
        
        else:
            # Mock response for demo
            return self._mock_response(messages, tools)
    
    def _mock_response(self, messages: list, tools: list) -> dict:
        """Generate mock responses for demo without API keys."""
        last_message = messages[-1]["content"] if messages else ""
        agent_name = self.current_agent.name
        
        # Simple rule-based mock
        response_content = f"[{agent_name}] I understand you're asking about: {last_message[:100]}..."
        tool_calls = []
        
        # Check for keywords and simulate appropriate responses
        msg_lower = last_message.lower()
        
        if "charged twice" in msg_lower or "duplicate" in msg_lower:
            if "billing" in agent_name.lower():
                tool_calls = [{"id": "1", "name": "check_duplicate_charges", "arguments": {"customer_id": "CUST-001"}}]
            else:
                tool_calls = [{"id": "1", "name": "transfer_to_billing", "arguments": {}}]
        
        elif "refund" in msg_lower:
            if "refund" in agent_name.lower():
                tool_calls = [{"id": "1", "name": "process_refund", "arguments": {"customer_id": "CUST-001", "amount": 29.99, "reason": "Duplicate charge"}}]
            elif "billing" in agent_name.lower():
                tool_calls = [{"id": "1", "name": "transfer_to_refunds", "arguments": {}}]
        
        elif "cancel" in msg_lower:
            tool_calls = [{"id": "1", "name": "transfer_to_retention", "arguments": {}}]
        
        return {
            "content": response_content,
            "tool_calls": tool_calls,
        }
    
    def _execute_function(self, func_name: str, args: dict) -> Any:
        """Execute a function by name."""
        # Find function in current agent's functions
        for func in self.current_agent.functions:
            if func.__name__ == func_name:
                try:
                    result = func(**args)
                    
                    # Check if result is an agent (handoff)
                    if isinstance(result, Agent):
                        return {"handoff": result}
                    
                    return {"result": str(result)}
                    
                except Exception as e:
                    return {"error": str(e)}
        
        return {"error": f"Function {func_name} not found"}
    
    def _perform_handoff(self, new_agent: Agent, reason: str = ""):
        """Perform a handoff to a new agent."""
        old_agent = self.current_agent
        
        # Record handoff in context
        self.context.record_handoff(
            from_agent=old_agent.name,
            to_agent=new_agent.name,
            reason=reason,
        )
        
        self.current_agent = new_agent
        self.handoff_count += 1
        
        if self.config.debug:
            print(f"🔄 Handoff: {old_agent.name} → {new_agent.name}")
        
        return f"[Transferred to {new_agent.name}]"
    
    def chat(self, user_message: str, customer_id: str = None) -> str:
        """
        Process a user message through the swarm.
        
        Args:
            user_message: The user's message
            customer_id: Optional customer ID for context
        
        Returns:
            The agent's response
        """
        # Add user message to history
        self.conversation_history.append(ConversationTurn(
            role="user",
            content=user_message,
        ))
        self.context.add_message("user", user_message)
        
        # Look up customer if ID provided
        if customer_id and not self.context.customer:
            customer = db.get_customer(customer_id)
            if customer:
                self.context.customer = customer
                self.context.customer_id = customer_id
        
        # Build messages for LLM
        messages = self._build_messages()
        
        # Get tools for current agent
        tools = [self._get_function_schema(f) for f in self.current_agent.functions]
        
        # Call LLM
        response = self._call_llm(messages, tools)
        
        # Process tool calls
        if response.get("tool_calls"):
            tool_results = []
            
            for tool_call in response["tool_calls"]:
                result = self._execute_function(
                    tool_call["name"],
                    tool_call.get("arguments", {}),
                )
                
                # Check for handoff
                if "handoff" in result:
                    handoff_msg = self._perform_handoff(
                        result["handoff"],
                        reason=tool_call["name"],
                    )
                    
                    # Check handoff limit
                    if self.handoff_count > self.config.max_handoffs:
                        return "I apologize, but I'm having trouble routing your request. Let me create a ticket for a human agent to follow up."
                    
                    # Recursive call with new agent
                    return self.chat(user_message, customer_id)
                
                tool_results.append({
                    "name": tool_call["name"],
                    "result": result.get("result", result.get("error", "")),
                })
            
            # Add tool results to messages and call again
            messages.append({
                "role": "assistant",
                "content": response.get("content", ""),
                "tool_calls": response["tool_calls"],
            })
            
            for i, tr in enumerate(tool_results):
                messages.append({
                    "role": "tool",
                    "tool_call_id": response["tool_calls"][i]["id"],
                    "content": tr["result"],
                })
            
            # Get final response
            final_response = self._call_llm(messages, tools)
            assistant_message = final_response.get("content", "")
        else:
            assistant_message = response.get("content", "")
        
        # Record response
        self.conversation_history.append(ConversationTurn(
            role="assistant",
            content=assistant_message,
            agent=self.current_agent.name,
        ))
        self.context.add_message("assistant", assistant_message)
        
        return assistant_message
    
    def _build_messages(self) -> list:
        """Build the message list for the LLM."""
        messages = []
        
        # System message with agent instructions
        system_content = self.current_agent.instructions
        
        # Add context if available
        if self.context.customer:
            system_content += f"\n\nCustomer Context:\n{self.context.get_summary()}"
        
        messages.append({
            "role": "system",
            "content": system_content,
        })
        
        # Add conversation history
        for turn in self.conversation_history[-10:]:  # Last 10 turns
            messages.append({
                "role": turn.role,
                "content": turn.content,
            })
        
        return messages
    
    def get_history(self) -> list[dict]:
        """Get conversation history."""
        return [
            {
                "role": turn.role,
                "content": turn.content,
                "agent": turn.agent,
            }
            for turn in self.conversation_history
        ]
    
    def get_context(self) -> dict:
        """Get current context."""
        return self.context.to_dict()
    
    def get_current_agent(self) -> str:
        """Get current agent name."""
        return self.current_agent.name
    
    def reset(self):
        """Reset the conversation."""
        self.current_agent = triage_agent
        self.context = SupportContext()
        self.conversation_history = []
        self.handoff_count = 0
    
    def set_customer(self, customer_id: str) -> bool:
        """Set the current customer."""
        customer = db.get_customer(customer_id)
        if customer:
            self.context.customer = customer
            self.context.customer_id = customer_id
            return True
        return False


# =============================================================================
# Convenience Functions
# =============================================================================

def create_support_swarm(provider: str = "openai") -> SupportSwarm:
    """Create a configured support swarm."""
    config = SwarmConfig(llm_provider=provider)
    return SupportSwarm(config)


def handle_support_query(query: str, customer_id: str = None) -> dict:
    """
    Handle a single support query.
    
    Args:
        query: Customer's query
        customer_id: Optional customer ID
    
    Returns:
        Response with message and metadata
    """
    swarm = SupportSwarm()
    
    if customer_id:
        swarm.set_customer(customer_id)
    
    response = swarm.chat(query)
    
    return {
        "response": response,
        "agent": swarm.get_current_agent(),
        "context": swarm.get_context(),
    }
