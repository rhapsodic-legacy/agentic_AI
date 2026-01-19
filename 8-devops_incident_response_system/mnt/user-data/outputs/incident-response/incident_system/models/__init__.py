"""
DevOps Incident Response System - Data Models

Models for:
- Incidents and alerts
- Runbooks and remediation actions
- Infrastructure state
- Communication and reporting
"""

from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import uuid


class Severity(Enum):
    """Incident severity levels."""
    SEV1 = "SEV1"  # Critical - Major outage affecting all users
    SEV2 = "SEV2"  # High - Significant impact on subset of users
    SEV3 = "SEV3"  # Medium - Minor impact, workaround available
    SEV4 = "SEV4"  # Low - Minimal impact, can be scheduled


class IncidentStatus(Enum):
    """Incident lifecycle status."""
    DETECTED = "detected"
    TRIAGED = "triaged"
    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"
    MITIGATING = "mitigating"
    MONITORING = "monitoring"
    RESOLVED = "resolved"
    POSTMORTEM = "postmortem"


class AlertSource(Enum):
    """Alert source systems."""
    DATADOG = "datadog"
    PROMETHEUS = "prometheus"
    PAGERDUTY = "pagerduty"
    CLOUDWATCH = "cloudwatch"
    CUSTOM = "custom"


class ServiceType(Enum):
    """Service types in the infrastructure."""
    API = "api"
    WEB = "web"
    DATABASE = "database"
    CACHE = "cache"
    QUEUE = "queue"
    WORKER = "worker"
    GATEWAY = "gateway"


class ActionType(Enum):
    """Remediation action types."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    RESTART = "restart"
    ROLLBACK = "rollback"
    CLEAR_CACHE = "clear_cache"
    FAILOVER = "failover"
    CONFIG_UPDATE = "config_update"
    DNS_UPDATE = "dns_update"
    MANUAL = "manual"


@dataclass
class Alert:
    """Incoming alert from monitoring systems."""
    alert_id: str
    source: AlertSource
    
    # Alert details
    title: str
    description: str
    severity: Severity
    
    # Affected resources
    service: str
    environment: str = "production"
    region: str = "us-east-1"
    
    # Metrics
    metric_name: str = ""
    metric_value: float = 0.0
    threshold: float = 0.0
    
    # Timing
    triggered_at: str = ""
    
    # Metadata
    tags: dict = field(default_factory=dict)
    runbook_url: str = ""
    
    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "source": self.source.value,
            "title": self.title,
            "severity": self.severity.value,
            "service": self.service,
            "environment": self.environment,
            "triggered_at": self.triggered_at,
        }


@dataclass
class Metric:
    """System metric data point."""
    name: str
    value: float
    unit: str = ""
    timestamp: str = ""
    tags: dict = field(default_factory=dict)
    
    # Thresholds
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    
    @property
    def is_warning(self) -> bool:
        if self.warning_threshold:
            return self.value >= self.warning_threshold
        return False
    
    @property
    def is_critical(self) -> bool:
        if self.critical_threshold:
            return self.value >= self.critical_threshold
        return False


@dataclass
class LogEntry:
    """Log entry from services."""
    timestamp: str
    level: str  # DEBUG, INFO, WARN, ERROR, FATAL
    service: str
    message: str
    
    # Structured data
    trace_id: str = ""
    span_id: str = ""
    error_type: str = ""
    stack_trace: str = ""
    
    # Context
    host: str = ""
    pod: str = ""
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "service": self.service,
            "message": self.message[:200],
        }


@dataclass
class Service:
    """Service in the infrastructure."""
    name: str
    service_type: ServiceType
    
    # Deployment
    namespace: str = "default"
    replicas: int = 3
    desired_replicas: int = 3
    
    # Health
    healthy_pods: int = 3
    unhealthy_pods: int = 0
    
    # Resources
    cpu_usage: float = 0.0  # percentage
    memory_usage: float = 0.0  # percentage
    
    # Current version
    version: str = "1.0.0"
    last_deployed: str = ""
    
    # Dependencies
    dependencies: list[str] = field(default_factory=list)
    
    # Endpoints
    endpoint: str = ""
    health_check: str = "/health"
    
    @property
    def is_healthy(self) -> bool:
        return self.unhealthy_pods == 0 and self.healthy_pods >= self.desired_replicas
    
    @property
    def health_percentage(self) -> float:
        if self.desired_replicas == 0:
            return 100.0
        return (self.healthy_pods / self.desired_replicas) * 100
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.service_type.value,
            "replicas": f"{self.healthy_pods}/{self.desired_replicas}",
            "cpu": f"{self.cpu_usage:.1f}%",
            "memory": f"{self.memory_usage:.1f}%",
            "healthy": self.is_healthy,
        }


@dataclass
class RunbookStep:
    """A step in a runbook."""
    step_id: str
    description: str
    action_type: ActionType
    
    # Action parameters
    parameters: dict = field(default_factory=dict)
    
    # Conditions
    condition: str = ""  # When to execute this step
    
    # Verification
    verification_command: str = ""
    expected_result: str = ""
    
    # Timing
    estimated_duration: int = 60  # seconds
    timeout: int = 300  # seconds
    
    # Flags
    requires_approval: bool = False
    is_destructive: bool = False


@dataclass
class Runbook:
    """Automated runbook for incident remediation."""
    runbook_id: str
    name: str
    description: str
    
    # Targeting
    target_services: list[str] = field(default_factory=list)
    target_alerts: list[str] = field(default_factory=list)
    
    # Steps
    steps: list[RunbookStep] = field(default_factory=list)
    
    # Metadata
    author: str = ""
    version: str = "1.0"
    last_updated: str = ""
    
    # Execution
    auto_execute: bool = False
    max_auto_severity: Severity = Severity.SEV3
    
    def to_dict(self) -> dict:
        return {
            "runbook_id": self.runbook_id,
            "name": self.name,
            "steps": len(self.steps),
            "auto_execute": self.auto_execute,
        }


@dataclass
class RemediationAction:
    """A remediation action taken during incident response."""
    action_id: str
    action_type: ActionType
    
    # Target
    target_service: str
    target_resource: str = ""
    
    # Details
    description: str = ""
    parameters: dict = field(default_factory=dict)
    
    # Execution
    executed_by: str = ""  # Agent or human
    executed_at: str = ""
    
    # Result
    success: bool = False
    result_message: str = ""
    duration_seconds: int = 0
    
    # Rollback
    rollback_action: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "action_id": self.action_id,
            "type": self.action_type.value,
            "target": self.target_service,
            "success": self.success,
            "executed_at": self.executed_at,
        }


@dataclass
class TimelineEvent:
    """Event in incident timeline."""
    timestamp: str
    event_type: str
    description: str
    actor: str = ""  # Agent or human who triggered
    details: dict = field(default_factory=dict)


@dataclass
class Incident:
    """A tracked incident."""
    incident_id: str
    
    # Classification
    title: str
    description: str
    severity: Severity
    status: IncidentStatus
    
    # Source
    source_alerts: list[Alert] = field(default_factory=list)
    
    # Affected
    affected_services: list[str] = field(default_factory=list)
    affected_regions: list[str] = field(default_factory=list)
    customer_impact: str = ""
    
    # Response
    incident_commander: str = ""
    responders: list[str] = field(default_factory=list)
    
    # Timeline
    detected_at: str = ""
    triaged_at: str = ""
    mitigated_at: str = ""
    resolved_at: str = ""
    
    # Actions taken
    actions: list[RemediationAction] = field(default_factory=list)
    timeline: list[TimelineEvent] = field(default_factory=list)
    
    # Root cause
    root_cause: str = ""
    root_cause_category: str = ""
    
    # Communication
    status_page_updated: bool = False
    stakeholders_notified: bool = False
    
    # Post-mortem
    postmortem_url: str = ""
    
    def add_timeline_event(self, event_type: str, description: str, actor: str = "system"):
        self.timeline.append(TimelineEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            description=description,
            actor=actor,
        ))
    
    def get_duration_minutes(self) -> Optional[int]:
        if self.detected_at and self.resolved_at:
            detected = datetime.fromisoformat(self.detected_at)
            resolved = datetime.fromisoformat(self.resolved_at)
            return int((resolved - detected).total_seconds() / 60)
        return None
    
    def to_dict(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "title": self.title,
            "severity": self.severity.value,
            "status": self.status.value,
            "affected_services": self.affected_services,
            "detected_at": self.detected_at,
            "actions_taken": len(self.actions),
        }
    
    def to_summary(self) -> str:
        summary = f"""# Incident {self.incident_id}

**Severity:** {self.severity.value}
**Status:** {self.status.value}
**Title:** {self.title}

**Description:** {self.description}

**Affected Services:** {', '.join(self.affected_services)}
**Customer Impact:** {self.customer_impact}

**Detected:** {self.detected_at}
**Triaged:** {self.triaged_at or 'N/A'}
**Mitigated:** {self.mitigated_at or 'N/A'}
**Resolved:** {self.resolved_at or 'N/A'}

**Root Cause:** {self.root_cause or 'Under investigation'}

**Actions Taken:** {len(self.actions)}
"""
        return summary


@dataclass
class StatusUpdate:
    """Status page update."""
    update_id: str
    incident_id: str
    
    # Content
    title: str
    message: str
    status: str  # investigating, identified, monitoring, resolved
    
    # Timing
    posted_at: str = ""
    
    # Affected components
    affected_components: list[str] = field(default_factory=list)


@dataclass
class PostMortem:
    """Post-incident report."""
    incident_id: str
    
    # Summary
    title: str
    summary: str
    
    # Timeline
    timeline: list[TimelineEvent] = field(default_factory=list)
    
    # Analysis
    root_cause: str = ""
    contributing_factors: list[str] = field(default_factory=list)
    
    # Impact
    duration_minutes: int = 0
    users_affected: int = 0
    revenue_impact: float = 0.0
    
    # Actions
    action_items: list[dict] = field(default_factory=list)  # {description, owner, due_date}
    
    # Lessons
    what_went_well: list[str] = field(default_factory=list)
    what_went_wrong: list[str] = field(default_factory=list)
    
    # Metadata
    author: str = ""
    created_at: str = ""
    
    def to_markdown(self) -> str:
        md = f"""# Post-Incident Report: {self.title}

**Incident ID:** {self.incident_id}
**Duration:** {self.duration_minutes} minutes
**Users Affected:** {self.users_affected:,}

## Summary
{self.summary}

## Root Cause
{self.root_cause}

## Contributing Factors
"""
        for factor in self.contributing_factors:
            md += f"- {factor}\n"
        
        md += "\n## Timeline\n"
        for event in self.timeline:
            md += f"- **{event.timestamp}** - {event.description}\n"
        
        md += "\n## What Went Well\n"
        for item in self.what_went_well:
            md += f"- {item}\n"
        
        md += "\n## What Went Wrong\n"
        for item in self.what_went_wrong:
            md += f"- {item}\n"
        
        md += "\n## Action Items\n"
        for item in self.action_items:
            md += f"- [ ] {item.get('description', '')} (Owner: {item.get('owner', 'TBD')})\n"
        
        return md


@dataclass 
class SystemState:
    """Current state of the infrastructure."""
    services: dict[str, Service] = field(default_factory=dict)
    active_incidents: list[Incident] = field(default_factory=list)
    recent_alerts: list[Alert] = field(default_factory=list)
    
    # Overall health
    overall_health: str = "healthy"  # healthy, degraded, outage
    
    def get_unhealthy_services(self) -> list[Service]:
        return [s for s in self.services.values() if not s.is_healthy]
    
    def to_dict(self) -> dict:
        return {
            "overall_health": self.overall_health,
            "service_count": len(self.services),
            "healthy_services": len([s for s in self.services.values() if s.is_healthy]),
            "active_incidents": len(self.active_incidents),
            "recent_alerts": len(self.recent_alerts),
        }
