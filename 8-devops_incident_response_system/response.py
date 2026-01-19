"""
DevOps Incident Response System - Main Orchestration 

AutoGen-based hierarchical incident response system.

Organization Structure:
┌─────────────────────────────────────────────────────────────────┐
│                    INCIDENT COMMANDER                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
      ┌─────────────────────┼─────────────────────────────────────┐
      │                     │                                     │
      ▼                     ▼                                     ▼
  MONITOR               DIAGNOSE                               COMMS
  AGENT                 AGENT                                  AGENT
      │                     │
      │              ┌──────┴──────┐
      │              │             │
      │              ▼             ▼
      │          INFRA          APP
      │          FIXER         FIXER
      │              │             │
      └──────────────┴─────────────┘
              (Feedback Loop)

Incident Response Flow:
1. DETECTION - Alert received, Monitor Agent validates
2. TRIAGE - Incident Commander assigns severity
3. DIAGNOSIS - Diagnose Agent investigates root cause
4. MITIGATION - Fixer Agents implement remediation
5. VERIFICATION - Monitor Agent confirms recovery
6. COMMUNICATION - Comms Agent updates stakeholders
7. POST-MORTEM - Generate incident report
"""

from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import json

try:
    import autogen
    from autogen import GroupChat, GroupChatManager
except ImportError:
    raise ImportError("Install autogen: pip install pyautogen")

from .models import (
    Alert, Incident, RemediationAction, PostMortem,
    Severity, IncidentStatus, ActionType
)
from .agents import IncidentResponseTeam, get_llm_config
from .tools import TOOLS, get_tools_for_role
from .integrations import monitoring, logging_system, infrastructure, communication


@dataclass
class ResponseConfig:
    """Configuration for incident response system."""
    llm_provider: str = "gemini"
    auto_approve_sev3_sev4: bool = True
    max_conversation_turns: int = 20
    verbose: bool = True


@dataclass
class IncidentContext:
    """Context for an active incident."""
    incident: Incident
    alerts: list[Alert] = field(default_factory=list)
    actions_taken: list[RemediationAction] = field(default_factory=list)
    conversation_history: list[dict] = field(default_factory=list)
    
    def add_action(self, action: RemediationAction):
        self.actions_taken.append(action)
        self.incident.actions.append(action)
        self.incident.add_timeline_event(
            event_type="remediation",
            description=f"{action.action_type.value}: {action.description}",
            actor=action.executed_by,
        )


class IncidentResponseSystem:
    """
    DevOps Incident Response System
    
    An autonomous incident response system that detects issues, diagnoses
    root causes, implements fixes, and coordinates communication.
    
    Usage:
        system = IncidentResponseSystem()
        
        # Simulate an incident
        alert = system.simulate_incident("api-gateway", "high_error_rate")
        
        # Respond to the incident
        result = system.respond_to_incident(alert)
        
        # Check incident status
        print(result.incident.to_summary())
    """
    
    def __init__(self, config: Optional[ResponseConfig] = None):
        self.config = config or ResponseConfig()
        
        # Initialize agent team
        self.team = IncidentResponseTeam(self.config.llm_provider)
        
        # Active incidents
        self.active_incidents: dict[str, IncidentContext] = {}
        self.resolved_incidents: list[Incident] = []
        
        # Register tool functions
        self._register_tools()
    
    def _register_tools(self):
        """Register tools with agents."""
        # Tools are available through the TOOLS dict
        # In production, these would be registered with AutoGen's function calling
        pass
    
    def simulate_incident(self, service: str, incident_type: str) -> Alert:
        """
        Simulate an incident for testing.
        
        Args:
            service: Service name (e.g., 'api-gateway')
            incident_type: Type of incident:
                - 'high_error_rate'
                - 'high_latency'
                - 'service_down'
                - 'high_cpu'
                - 'database_connections'
        
        Returns:
            Generated alert
        """
        # Also simulate infrastructure failure
        if incident_type == "high_cpu":
            infrastructure.simulate_failure(service, "high_cpu")
        elif incident_type == "service_down":
            infrastructure.simulate_failure(service, "pod_crash")
        
        return monitoring.simulate_incident(service, incident_type)
    
    def respond_to_incident(self, alert: Alert) -> IncidentContext:
        """
        Respond to an incident alert.
        
        This orchestrates the full incident response flow:
        1. Create incident from alert
        2. Triage and assign severity
        3. Diagnose root cause
        4. Execute remediation
        5. Verify recovery
        6. Communicate updates
        
        Args:
            alert: The triggering alert
        
        Returns:
            IncidentContext with full response details
        """
        # Create incident
        incident = self._create_incident(alert)
        context = IncidentContext(incident=incident, alerts=[alert])
        
        self.active_incidents[incident.incident_id] = context
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"🚨 INCIDENT DETECTED: {incident.incident_id}")
            print(f"   Severity: {incident.severity.value}")
            print(f"   Service: {', '.join(incident.affected_services)}")
            print(f"   Alert: {alert.title}")
            print(f"{'='*60}\n")
        
        # Execute response flow
        try:
            # Phase 1: Triage
            self._phase_triage(context)
            
            # Phase 2: Diagnosis
            self._phase_diagnose(context)
            
            # Phase 3: Mitigation
            self._phase_mitigate(context)
            
            # Phase 4: Verification
            self._phase_verify(context)
            
            # Phase 5: Communication
            self._phase_communicate(context)
            
            # Phase 6: Resolution
            self._phase_resolve(context)
            
        except Exception as e:
            print(f"Error during incident response: {e}")
            incident.add_timeline_event("error", f"Response error: {str(e)}")
        
        return context
    
    def _create_incident(self, alert: Alert) -> Incident:
        """Create an incident from an alert."""
        incident_id = f"INC-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        
        incident = Incident(
            incident_id=incident_id,
            title=alert.title,
            description=alert.description,
            severity=alert.severity,
            status=IncidentStatus.DETECTED,
            source_alerts=[alert],
            affected_services=[alert.service],
            affected_regions=[alert.region],
            detected_at=datetime.now().isoformat(),
            incident_commander="Incident_Commander",
            responders=["Monitor_Agent", "Diagnose_Agent", "Infra_Fixer", "App_Fixer", "Comms_Agent"],
        )
        
        incident.add_timeline_event("alert", f"Alert received: {alert.title}", "system")
        
        return incident
    
    def _phase_triage(self, context: IncidentContext):
        """Phase 1: Triage the incident."""
        incident = context.incident
        
        if self.config.verbose:
            print("📋 Phase 1: TRIAGE")
        
        incident.status = IncidentStatus.TRIAGED
        incident.triaged_at = datetime.now().isoformat()
        
        # Determine customer impact
        if incident.severity == Severity.SEV1:
            incident.customer_impact = "All users experiencing service disruption"
        elif incident.severity == Severity.SEV2:
            incident.customer_impact = "Significant subset of users affected"
        else:
            incident.customer_impact = "Minor or limited impact"
        
        incident.add_timeline_event(
            "triage",
            f"Incident triaged as {incident.severity.value}. Impact: {incident.customer_impact}",
            "Incident_Commander"
        )
        
        if self.config.verbose:
            print(f"   ✓ Severity: {incident.severity.value}")
            print(f"   ✓ Impact: {incident.customer_impact}")
    
    def _phase_diagnose(self, context: IncidentContext):
        """Phase 2: Diagnose root cause."""
        incident = context.incident
        
        if self.config.verbose:
            print("\n🔍 Phase 2: DIAGNOSIS")
        
        incident.status = IncidentStatus.INVESTIGATING
        
        service = incident.affected_services[0] if incident.affected_services else ""
        
        # Get health check
        health = TOOLS["check_service_health"](service)
        if self.config.verbose:
            print(f"   Health Check:\n{health}")
        
        # Search logs
        error_summary = TOOLS["get_error_summary"](service)
        if self.config.verbose:
            print(f"\n   Error Summary:\n{error_summary}")
        
        # Find root cause
        root_cause_analysis = TOOLS["find_root_cause"](service)
        if self.config.verbose:
            print(f"\n   Root Cause Analysis:\n{root_cause_analysis}")
        
        # Check dependencies
        dep_check = TOOLS["check_dependencies"](service)
        if self.config.verbose:
            print(f"\n   Dependencies:\n{dep_check}")
        
        # Set root cause (would be determined by agent in full implementation)
        if "connection_error" in error_summary.lower():
            incident.root_cause = "Database connection issues causing service failures"
            incident.root_cause_category = "database"
        elif "timeout" in error_summary.lower():
            incident.root_cause = "Upstream service timeouts causing cascading failures"
            incident.root_cause_category = "dependency"
        else:
            incident.root_cause = "Elevated error rates due to recent deployment or traffic spike"
            incident.root_cause_category = "application"
        
        incident.status = IncidentStatus.IDENTIFIED
        incident.add_timeline_event(
            "diagnosis",
            f"Root cause identified: {incident.root_cause}",
            "Diagnose_Agent"
        )
        
        if self.config.verbose:
            print(f"\n   ✓ Root Cause: {incident.root_cause}")
    
    def _phase_mitigate(self, context: IncidentContext):
        """Phase 3: Mitigate the incident."""
        incident = context.incident
        
        if self.config.verbose:
            print("\n🔧 Phase 3: MITIGATION")
        
        incident.status = IncidentStatus.MITIGATING
        
        service = incident.affected_services[0] if incident.affected_services else ""
        
        # Determine remediation actions based on root cause
        actions_taken = []
        
        # Get service info
        svc = infrastructure.get_service(service)
        
        # Action 1: Scale up if possible
        if svc and svc.desired_replicas < 10:
            new_replicas = min(svc.desired_replicas * 2, 10)
            result = TOOLS["scale_service"](service, new_replicas)
            
            action = RemediationAction(
                action_id=f"action-{uuid.uuid4().hex[:8]}",
                action_type=ActionType.SCALE_UP,
                target_service=service,
                description=f"Scaled service from {svc.desired_replicas} to {new_replicas} replicas",
                executed_by="Infra_Fixer",
                executed_at=datetime.now().isoformat(),
                success="success" in result.lower() or "✅" in result,
                result_message=result,
            )
            context.add_action(action)
            actions_taken.append(result)
            
            if self.config.verbose:
                print(f"   {result}")
        
        # Action 2: Restart service
        result = TOOLS["restart_service"](service)
        
        action = RemediationAction(
            action_id=f"action-{uuid.uuid4().hex[:8]}",
            action_type=ActionType.RESTART,
            target_service=service,
            description="Rolling restart initiated",
            executed_by="App_Fixer",
            executed_at=datetime.now().isoformat(),
            success="success" in result.lower() or "✅" in result,
            result_message=result,
        )
        context.add_action(action)
        actions_taken.append(result)
        
        if self.config.verbose:
            print(f"   {result}")
        
        # For SEV1/SEV2, consider rollback
        if incident.severity in [Severity.SEV1, Severity.SEV2]:
            if self.config.verbose:
                print("\n   ⚠️ High severity - considering rollback...")
            
            # Simulate approval for demo
            result = TOOLS["rollback_service"](service)
            
            action = RemediationAction(
                action_id=f"action-{uuid.uuid4().hex[:8]}",
                action_type=ActionType.ROLLBACK,
                target_service=service,
                description="Rolled back to previous version",
                executed_by="App_Fixer",
                executed_at=datetime.now().isoformat(),
                success="success" in result.lower() or "✅" in result,
                result_message=result,
            )
            context.add_action(action)
            actions_taken.append(result)
            
            if self.config.verbose:
                print(f"   {result}")
        
        incident.mitigated_at = datetime.now().isoformat()
        incident.add_timeline_event(
            "mitigation",
            f"Mitigation actions completed: {len(context.actions_taken)} actions taken",
            "Infra_Fixer"
        )
    
    def _phase_verify(self, context: IncidentContext):
        """Phase 4: Verify recovery."""
        incident = context.incident
        
        if self.config.verbose:
            print("\n✅ Phase 4: VERIFICATION")
        
        incident.status = IncidentStatus.MONITORING
        
        service = incident.affected_services[0] if incident.affected_services else ""
        
        # Check metrics
        health = TOOLS["check_service_health"](service)
        
        if self.config.verbose:
            print(f"   Post-mitigation health:\n{health}")
        
        # Simulate recovery
        # In production, would actually wait and verify metrics
        
        # Reset mock data to healthy state
        if service in monitoring.services:
            monitoring.services[service]["error_rate"] = 0.5
            monitoring.services[service]["latency_p99"] = 150
            monitoring.services[service]["cpu"] = 45
        
        if service in infrastructure.services:
            svc = infrastructure.services[service]
            svc.unhealthy_pods = 0
            svc.healthy_pods = svc.desired_replicas
            svc.cpu_usage = 45
            svc.memory_usage = 60
        
        # Verify again
        health_after = TOOLS["check_service_health"](service)
        
        if self.config.verbose:
            print(f"\n   After recovery:\n{health_after}")
        
        incident.add_timeline_event(
            "verification",
            "Service metrics returning to normal",
            "Monitor_Agent"
        )
    
    def _phase_communicate(self, context: IncidentContext):
        """Phase 5: Communicate updates."""
        incident = context.incident
        
        if self.config.verbose:
            print("\n📢 Phase 5: COMMUNICATION")
        
        # Send Slack notification
        result = TOOLS["send_slack_alert"](
            "#incidents",
            f"[{incident.severity.value}] {incident.title} - Mitigation in progress",
            "high" if incident.severity in [Severity.SEV1, Severity.SEV2] else "normal"
        )
        if self.config.verbose:
            print(f"   {result}")
        
        # Page on-call for high severity
        if incident.severity in [Severity.SEV1, Severity.SEV2]:
            result = TOOLS["page_oncall_team"]("platform", incident.title, incident.severity)
            if self.config.verbose:
                print(f"   {result}")
        
        # Update status page
        result = TOOLS["update_status_page"](
            "monitoring",
            f"We identified and resolved an issue with {incident.affected_services[0]}. Monitoring recovery.",
            incident.affected_services
        )
        incident.status_page_updated = True
        if self.config.verbose:
            print(f"   {result}")
        
        # Create ticket
        result = TOOLS["create_incident_ticket"](
            f"[{incident.severity.value}] {incident.title}",
            incident.description,
            "high" if incident.severity in [Severity.SEV1, Severity.SEV2] else "medium"
        )
        if self.config.verbose:
            print(f"   {result}")
        
        # Notify stakeholders for SEV1
        if incident.severity == Severity.SEV1:
            result = TOOLS["notify_stakeholders"](
                incident.incident_id,
                f"SEV1 Incident Update: {incident.title}\nStatus: Mitigating\nImpact: {incident.customer_impact}"
            )
            incident.stakeholders_notified = True
            if self.config.verbose:
                print(f"   {result}")
        
        incident.add_timeline_event(
            "communication",
            "Stakeholders notified, status page updated",
            "Comms_Agent"
        )
    
    def _phase_resolve(self, context: IncidentContext):
        """Phase 6: Resolve incident."""
        incident = context.incident
        
        if self.config.verbose:
            print("\n🎉 Phase 6: RESOLUTION")
        
        incident.status = IncidentStatus.RESOLVED
        incident.resolved_at = datetime.now().isoformat()
        
        # Send resolution notification
        result = TOOLS["send_slack_alert"](
            "#incidents",
            f"✅ [{incident.severity.value}] {incident.title} - RESOLVED\nDuration: {incident.get_duration_minutes() or 0} minutes",
            "normal"
        )
        if self.config.verbose:
            print(f"   {result}")
        
        # Update status page to resolved
        result = TOOLS["update_status_page"](
            "resolved",
            f"The issue has been resolved. {incident.affected_services[0]} is operating normally.",
            incident.affected_services
        )
        if self.config.verbose:
            print(f"   {result}")
        
        incident.add_timeline_event(
            "resolved",
            f"Incident resolved after {incident.get_duration_minutes() or 0} minutes",
            "Incident_Commander"
        )
        
        # Move to resolved
        if incident.incident_id in self.active_incidents:
            del self.active_incidents[incident.incident_id]
        self.resolved_incidents.append(incident)
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"✅ INCIDENT RESOLVED: {incident.incident_id}")
            print(f"   Duration: {incident.get_duration_minutes() or 0} minutes")
            print(f"   Actions taken: {len(context.actions_taken)}")
            print(f"{'='*60}")
    
    def generate_postmortem(self, incident_id: str) -> str:
        """
        Generate post-incident report.
        
        Args:
            incident_id: Incident ID
        
        Returns:
            Post-mortem report in markdown
        """
        # Find incident
        incident = None
        for inc in self.resolved_incidents:
            if inc.incident_id == incident_id:
                incident = inc
                break
        
        if not incident:
            return f"Incident not found: {incident_id}"
        
        return TOOLS["generate_postmortem"](incident)
    
    def get_system_status(self) -> dict:
        """Get current system status."""
        return {
            "active_incidents": len(self.active_incidents),
            "resolved_today": len(self.resolved_incidents),
            "system_health": TOOLS["get_all_services_health"](),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_incident_system(provider: str = "gemini") -> IncidentResponseSystem:
    """Create an incident response system."""
    config = ResponseConfig(llm_provider=provider)
    return IncidentResponseSystem(config)


def simulate_and_respond(service: str, incident_type: str) -> IncidentContext:
    """
    Simulate an incident and respond to it.
    
    Args:
        service: Service name
        incident_type: Type of incident
    
    Returns:
        Incident context with response details
    """
    system = create_incident_system()
    alert = system.simulate_incident(service, incident_type)
    return system.respond_to_incident(alert)
