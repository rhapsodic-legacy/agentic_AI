"""
DevOps Incident Response System - AutoGen Agents

Hierarchical agent organization:
- Incident Commander (coordinates all response)
- Monitor Agent (alerts and metrics)
- Diagnose Agent (root cause analysis)
- Infrastructure Fixer (K8s operations)
- Application Fixer (rollbacks, restarts)
- Communications Agent (Slack, status page)
"""

from typing import Optional, Callable
import os

try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent
except ImportError:
    raise ImportError("Install autogen: pip install pyautogen")

from ..tools import get_tools_for_role, TOOLS


def get_llm_config(provider: str = "gemini") -> dict:
    """Get LLM configuration for AutoGen."""
    
    if provider == "gemini":
        return {
            "config_list": [{
                "model": "gemini-1.5-flash",
                "api_key": os.environ.get("GOOGLE_API_KEY", ""),
                "api_type": "google",
            }],
            "temperature": 0.3,
        }
    elif provider == "openai":
        return {
            "config_list": [{
                "model": "gpt-4o-mini",
                "api_key": os.environ.get("OPENAI_API_KEY", ""),
            }],
            "temperature": 0.3,
        }
    elif provider == "anthropic":
        return {
            "config_list": [{
                "model": "claude-sonnet-4-20250514",
                "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
                "api_type": "anthropic",
            }],
            "temperature": 0.3,
        }
    else:
        # Default fallback
        return {
            "config_list": [{"model": "gpt-4o-mini"}],
            "temperature": 0.3,
        }


# =============================================================================
# Agent Definitions
# =============================================================================

def create_incident_commander(llm_config: dict) -> AssistantAgent:
    """
    Create the Incident Commander agent.
    
    The IC coordinates all incident response activities:
    - Triages incoming alerts
    - Assigns tasks to responders
    - Makes escalation decisions
    - Tracks incident progress
    """
    return AssistantAgent(
        name="Incident_Commander",
        system_message="""You are the Incident Commander for a DevOps team. Your responsibilities:

1. **TRIAGE**: When an incident is detected:
   - Assess severity (SEV1-4) based on impact
   - Identify affected services and customers
   - Declare the incident officially

2. **COORDINATE**: Assign tasks to your team:
   - Monitor Agent: Track metrics and validate alerts
   - Diagnose Agent: Investigate root cause
   - Infra Fixer: Execute infrastructure changes
   - App Fixer: Handle application-level fixes
   - Comms Agent: Manage communications

3. **DECIDE**: Make critical decisions:
   - Approve destructive actions (rollbacks, failovers)
   - Escalate to humans when needed
   - Determine when incident is resolved

4. **COMMUNICATE**: Keep stakeholders informed:
   - Coordinate status updates
   - Notify leadership for SEV1/SEV2
   - Initiate post-mortem process

Severity Guide:
- SEV1: Major outage, all users affected, immediate response
- SEV2: Significant impact, subset of users, urgent response
- SEV3: Minor impact, workaround available, standard response
- SEV4: Minimal impact, can be scheduled

Always follow the incident response flow:
DETECT → TRIAGE → DIAGNOSE → MITIGATE → VERIFY → COMMUNICATE → RESOLVE

When the incident is resolved, generate a summary and initiate post-mortem.""",
        llm_config=llm_config,
    )


def create_monitor_agent(llm_config: dict) -> AssistantAgent:
    """
    Create the Monitor Agent.
    
    Responsible for:
    - Monitoring system health
    - Validating alerts
    - Tracking metrics during incidents
    - Confirming recovery
    """
    return AssistantAgent(
        name="Monitor_Agent",
        system_message="""You are the Monitor Agent responsible for system observability.

Your responsibilities:
1. **DETECT**: Monitor incoming alerts and metrics
   - Validate alert legitimacy (not false positive)
   - Check affected services and scope
   - Report findings to Incident Commander

2. **TRACK**: During incidents
   - Continuously monitor affected services
   - Track metric trends (improving/worsening)
   - Watch for cascading failures

3. **VERIFY**: After mitigation
   - Confirm metrics return to normal
   - Verify no new alerts
   - Report recovery status

Available tools:
- get_service_metrics(service): Get current metrics
- check_service_health(service): Health check
- get_all_services_health(): System-wide health
- get_recent_alerts(): Recent alerts

When checking health, report:
- Current metric values
- Threshold violations
- Trend direction (up/down/stable)
- Dependencies status""",
        llm_config=llm_config,
    )


def create_diagnose_agent(llm_config: dict) -> AssistantAgent:
    """
    Create the Diagnose Agent.
    
    Responsible for:
    - Log analysis
    - Root cause investigation
    - Dependency mapping
    - Pattern recognition
    """
    return AssistantAgent(
        name="Diagnose_Agent",
        system_message="""You are the Diagnose Agent responsible for root cause analysis.

Your responsibilities:
1. **INVESTIGATE**: When assigned an incident
   - Search error logs for relevant entries
   - Identify error patterns and frequencies
   - Check recent changes (deployments, configs)

2. **ANALYZE**: Determine root cause
   - Correlate errors with timing of issues
   - Check dependency health
   - Review recent deployment history
   - Identify contributing factors

3. **REPORT**: Share findings with team
   - Probable root cause with confidence level
   - Supporting evidence (logs, metrics)
   - Recommended remediation actions

Available tools:
- search_error_logs(service, time_range): Search logs
- get_error_summary(service): Error breakdown
- find_root_cause(service): Pattern analysis
- check_dependencies(service): Dependency health
- get_service_info(service): Service details

Investigation checklist:
1. What changed recently?
2. When did errors start?
3. Which dependencies are affected?
4. What's the error pattern?
5. Is there a common cause?""",
        llm_config=llm_config,
    )


def create_infra_fixer(llm_config: dict) -> AssistantAgent:
    """
    Create the Infrastructure Fixer Agent.
    
    Responsible for:
    - Kubernetes operations (scale, restart)
    - Infrastructure changes
    - Runbook execution
    """
    return AssistantAgent(
        name="Infra_Fixer",
        system_message="""You are the Infrastructure Fixer responsible for infrastructure remediation.

Your responsibilities:
1. **EXECUTE**: Perform infrastructure changes
   - Scale services up/down
   - Restart pods/services
   - Execute runbook steps

2. **VALIDATE**: Ensure changes are safe
   - Check current state before changes
   - Verify changes took effect
   - Report results to Incident Commander

3. **ESCALATE**: For destructive actions
   - Request approval for rollbacks
   - Flag risky operations
   - Suggest alternatives if possible

Available tools:
- scale_service(service, replicas): Scale service
- restart_service(service): Rolling restart
- rollback_service(service, version): Rollback
- get_service_info(service): Current state
- list_available_runbooks(): Available runbooks
- get_runbook_for_alert(alert_type): Find runbook

Safety rules:
- Always check current state before changes
- Request IC approval for destructive actions
- Document all changes made
- Have rollback plan ready""",
        llm_config=llm_config,
    )


def create_app_fixer(llm_config: dict) -> AssistantAgent:
    """
    Create the Application Fixer Agent.
    
    Responsible for:
    - Application-level fixes
    - Rollbacks and restarts
    - Configuration changes
    """
    return AssistantAgent(
        name="App_Fixer",
        system_message="""You are the Application Fixer responsible for application-level remediation.

Your responsibilities:
1. **REMEDIATE**: Fix application issues
   - Rollback bad deployments
   - Restart failing services
   - Clear caches if needed
   - Update configurations

2. **COORDINATE**: Work with Infra Fixer
   - Request infrastructure changes
   - Verify application behavior
   - Report success/failure

3. **VERIFY**: Confirm fixes work
   - Check error rates after changes
   - Monitor application logs
   - Report to Incident Commander

Available tools:
- restart_service(service): Restart application
- rollback_service(service, version): Rollback deployment
- get_service_info(service): Service details
- search_error_logs(service, time_range): Check logs

Common fixes:
1. Recent deployment? → Rollback
2. Memory issues? → Restart
3. Config problem? → Update and restart
4. Dependency issue? → Coordinate with Infra Fixer""",
        llm_config=llm_config,
    )


def create_comms_agent(llm_config: dict) -> AssistantAgent:
    """
    Create the Communications Agent.
    
    Responsible for:
    - Slack notifications
    - Status page updates
    - Stakeholder communication
    - Post-incident documentation
    """
    return AssistantAgent(
        name="Comms_Agent",
        system_message="""You are the Communications Agent responsible for incident communication.

Your responsibilities:
1. **NOTIFY**: Alert relevant parties
   - Post to #incidents Slack channel
   - Page on-call for SEV1/SEV2
   - Email stakeholders as needed

2. **UPDATE**: Keep everyone informed
   - Update status page for customer-facing issues
   - Send regular updates during incident
   - Announce when resolved

3. **DOCUMENT**: Create records
   - Create incident tickets
   - Document timeline
   - Support post-mortem process

Available tools:
- send_slack_alert(channel, message, severity): Slack message
- page_oncall_team(team, message, severity): Page on-call
- update_status_page(status, message, components): Status page
- create_incident_ticket(title, description, priority): Jira ticket
- notify_stakeholders(incident_id, message): Email update

Status page statuses:
- investigating: Just detected, looking into it
- identified: Root cause found
- monitoring: Fix applied, watching
- resolved: Issue fixed, normal operation

Communication templates:
- Initial: "[SEV{X}] {Service} experiencing issues. Investigating."
- Update: "Root cause identified. Implementing fix."
- Resolved: "Issue resolved. {Service} operating normally."
- Postmortem: "Post-incident review scheduled."

Remember: Be clear, concise, and factual in all communications.""",
        llm_config=llm_config,
    )


def create_human_proxy() -> UserProxyAgent:
    """
    Create a human proxy for approvals and escalations.
    """
    return UserProxyAgent(
        name="Human_Operator",
        human_input_mode="NEVER",  # Auto-approve for demo
        code_execution_config=False,
        default_auto_reply="Approved. Proceed with the action.",
        max_consecutive_auto_reply=5,
    )


# =============================================================================
# Agent Team Factory
# =============================================================================

class IncidentResponseTeam:
    """
    Factory for creating the incident response team.
    """
    
    def __init__(self, llm_provider: str = "gemini"):
        self.llm_config = get_llm_config(llm_provider)
        self._agents = {}
    
    @property
    def incident_commander(self) -> AssistantAgent:
        if "incident_commander" not in self._agents:
            self._agents["incident_commander"] = create_incident_commander(self.llm_config)
        return self._agents["incident_commander"]
    
    @property
    def monitor_agent(self) -> AssistantAgent:
        if "monitor_agent" not in self._agents:
            self._agents["monitor_agent"] = create_monitor_agent(self.llm_config)
        return self._agents["monitor_agent"]
    
    @property
    def diagnose_agent(self) -> AssistantAgent:
        if "diagnose_agent" not in self._agents:
            self._agents["diagnose_agent"] = create_diagnose_agent(self.llm_config)
        return self._agents["diagnose_agent"]
    
    @property
    def infra_fixer(self) -> AssistantAgent:
        if "infra_fixer" not in self._agents:
            self._agents["infra_fixer"] = create_infra_fixer(self.llm_config)
        return self._agents["infra_fixer"]
    
    @property
    def app_fixer(self) -> AssistantAgent:
        if "app_fixer" not in self._agents:
            self._agents["app_fixer"] = create_app_fixer(self.llm_config)
        return self._agents["app_fixer"]
    
    @property
    def comms_agent(self) -> AssistantAgent:
        if "comms_agent" not in self._agents:
            self._agents["comms_agent"] = create_comms_agent(self.llm_config)
        return self._agents["comms_agent"]
    
    @property
    def human_proxy(self) -> UserProxyAgent:
        if "human_proxy" not in self._agents:
            self._agents["human_proxy"] = create_human_proxy()
        return self._agents["human_proxy"]
    
    def get_all_agents(self) -> list:
        """Get all agents."""
        return [
            self.incident_commander,
            self.monitor_agent,
            self.diagnose_agent,
            self.infra_fixer,
            self.app_fixer,
            self.comms_agent,
        ]
    
    def get_agent_by_name(self, name: str) -> Optional[AssistantAgent]:
        """Get agent by name."""
        name_map = {
            "Incident_Commander": self.incident_commander,
            "Monitor_Agent": self.monitor_agent,
            "Diagnose_Agent": self.diagnose_agent,
            "Infra_Fixer": self.infra_fixer,
            "App_Fixer": self.app_fixer,
            "Comms_Agent": self.comms_agent,
        }
        return name_map.get(name)
