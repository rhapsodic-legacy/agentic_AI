"""
DevOps Incident Response System - Mock Integrations

Mock implementations for:
- Monitoring (Datadog, Prometheus)
- Logging (Elasticsearch, CloudWatch)
- Infrastructure (Kubernetes, AWS)
- Communication (Slack, PagerDuty, Statuspage)
"""

from typing import Optional
from datetime import datetime, timedelta
import random
import uuid

from ..models import (
    Alert, Metric, LogEntry, Service, Runbook, RunbookStep,
    Severity, AlertSource, ServiceType, ActionType, SystemState
)


# =============================================================================
# Monitoring Integration
# =============================================================================

class MockMonitoring:
    """
    Mock monitoring system (Datadog/Prometheus/CloudWatch).
    
    In production, replace with actual API calls.
    """
    
    def __init__(self):
        self.alerts_triggered = []
        self._setup_mock_data()
    
    def _setup_mock_data(self):
        """Setup mock monitoring data."""
        self.services = {
            "api-gateway": {
                "error_rate": 0.5,
                "latency_p99": 150,
                "requests_per_second": 1000,
                "cpu": 45,
                "memory": 60,
            },
            "user-service": {
                "error_rate": 0.1,
                "latency_p99": 80,
                "requests_per_second": 500,
                "cpu": 30,
                "memory": 45,
            },
            "order-service": {
                "error_rate": 0.2,
                "latency_p99": 200,
                "requests_per_second": 300,
                "cpu": 55,
                "memory": 70,
            },
            "payment-service": {
                "error_rate": 0.05,
                "latency_p99": 250,
                "requests_per_second": 100,
                "cpu": 25,
                "memory": 40,
            },
            "postgres-primary": {
                "connections": 80,
                "query_time_avg": 15,
                "replication_lag": 0,
                "cpu": 40,
                "memory": 75,
            },
            "redis-cache": {
                "hit_rate": 95,
                "memory_used": 60,
                "connections": 150,
                "cpu": 15,
                "memory": 65,
            },
        }
    
    def get_metrics(self, service: str) -> list[Metric]:
        """Get current metrics for a service."""
        if service not in self.services:
            return []
        
        data = self.services[service]
        metrics = []
        
        for name, value in data.items():
            # Add some randomness
            jitter = random.uniform(-5, 5)
            actual_value = max(0, value + jitter)
            
            metric = Metric(
                name=name,
                value=actual_value,
                timestamp=datetime.now().isoformat(),
                tags={"service": service},
            )
            
            # Set thresholds
            if name == "error_rate":
                metric.warning_threshold = 1.0
                metric.critical_threshold = 5.0
                metric.unit = "%"
            elif name == "latency_p99":
                metric.warning_threshold = 500
                metric.critical_threshold = 1000
                metric.unit = "ms"
            elif name == "cpu":
                metric.warning_threshold = 70
                metric.critical_threshold = 90
                metric.unit = "%"
            elif name == "memory":
                metric.warning_threshold = 80
                metric.critical_threshold = 95
                metric.unit = "%"
            
            metrics.append(metric)
        
        return metrics
    
    def get_all_metrics(self) -> dict[str, list[Metric]]:
        """Get metrics for all services."""
        return {service: self.get_metrics(service) for service in self.services}
    
    def check_health(self, service: str) -> dict:
        """Health check for a service."""
        metrics = self.get_metrics(service)
        
        issues = []
        status = "healthy"
        
        for metric in metrics:
            if metric.is_critical:
                issues.append(f"{metric.name} is critical: {metric.value:.1f}{metric.unit}")
                status = "critical"
            elif metric.is_warning and status != "critical":
                issues.append(f"{metric.name} is elevated: {metric.value:.1f}{metric.unit}")
                status = "warning"
        
        return {
            "service": service,
            "status": status,
            "issues": issues,
            "metrics": {m.name: m.value for m in metrics},
        }
    
    def simulate_incident(self, service: str, incident_type: str) -> Alert:
        """Simulate an incident for testing."""
        now = datetime.now()
        
        incidents = {
            "high_error_rate": {
                "title": f"High Error Rate on {service}",
                "description": f"Error rate exceeded 5% threshold on {service}",
                "severity": Severity.SEV2,
                "metric_name": "error_rate",
                "metric_value": 8.5,
                "threshold": 5.0,
            },
            "high_latency": {
                "title": f"High Latency on {service}",
                "description": f"P99 latency exceeded 1000ms on {service}",
                "severity": Severity.SEV2,
                "metric_name": "latency_p99",
                "metric_value": 1500,
                "threshold": 1000,
            },
            "service_down": {
                "title": f"Service {service} is DOWN",
                "description": f"Health checks failing for {service}",
                "severity": Severity.SEV1,
                "metric_name": "health_check",
                "metric_value": 0,
                "threshold": 1,
            },
            "high_cpu": {
                "title": f"High CPU Usage on {service}",
                "description": f"CPU usage exceeded 90% on {service}",
                "severity": Severity.SEV3,
                "metric_name": "cpu",
                "metric_value": 95,
                "threshold": 90,
            },
            "database_connections": {
                "title": f"Database Connection Pool Exhausted",
                "description": f"Connection pool at 100% capacity",
                "severity": Severity.SEV1,
                "metric_name": "connections",
                "metric_value": 100,
                "threshold": 90,
            },
        }
        
        incident_data = incidents.get(incident_type, incidents["high_error_rate"])
        
        alert = Alert(
            alert_id=f"alert-{uuid.uuid4().hex[:8]}",
            source=AlertSource.DATADOG,
            title=incident_data["title"],
            description=incident_data["description"],
            severity=incident_data["severity"],
            service=service,
            environment="production",
            metric_name=incident_data["metric_name"],
            metric_value=incident_data["metric_value"],
            threshold=incident_data["threshold"],
            triggered_at=now.isoformat(),
        )
        
        # Update mock data to reflect incident
        if service in self.services:
            self.services[service][incident_data["metric_name"]] = incident_data["metric_value"]
        
        self.alerts_triggered.append(alert)
        
        return alert


# =============================================================================
# Logging Integration
# =============================================================================

class MockLogging:
    """
    Mock logging system (Elasticsearch/Splunk/CloudWatch).
    """
    
    def __init__(self):
        self._setup_mock_logs()
    
    def _setup_mock_logs(self):
        """Setup mock log data."""
        self.error_patterns = [
            ("Connection refused", "database", "connection_error"),
            ("Timeout waiting for response", "upstream", "timeout"),
            ("Out of memory", "memory", "oom"),
            ("Rate limit exceeded", "rate_limit", "throttling"),
            ("Authentication failed", "auth", "auth_error"),
            ("Invalid JSON payload", "parsing", "bad_request"),
            ("Certificate expired", "ssl", "cert_error"),
            ("Disk space low", "storage", "disk_full"),
        ]
    
    def search_logs(
        self,
        service: str,
        level: str = "ERROR",
        time_range_minutes: int = 30,
        limit: int = 50,
    ) -> list[LogEntry]:
        """Search logs for a service."""
        logs = []
        now = datetime.now()
        
        # Generate mock logs
        for i in range(min(limit, random.randint(5, 30))):
            timestamp = now - timedelta(minutes=random.randint(0, time_range_minutes))
            
            if level == "ERROR":
                pattern, category, error_type = random.choice(self.error_patterns)
                message = f"{pattern}: {category} error in {service}"
            else:
                message = f"Processing request in {service}"
            
            logs.append(LogEntry(
                timestamp=timestamp.isoformat(),
                level=level,
                service=service,
                message=message,
                trace_id=f"trace-{uuid.uuid4().hex[:12]}",
                error_type=error_type if level == "ERROR" else "",
                host=f"{service}-pod-{random.randint(1, 5)}",
                pod=f"{service}-{uuid.uuid4().hex[:8]}",
            ))
        
        return sorted(logs, key=lambda x: x.timestamp, reverse=True)
    
    def get_error_summary(self, service: str, time_range_minutes: int = 60) -> dict:
        """Get error summary for a service."""
        errors = self.search_logs(service, "ERROR", time_range_minutes)
        
        # Count by error type
        error_counts = {}
        for log in errors:
            error_type = log.error_type or "unknown"
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            "service": service,
            "time_range_minutes": time_range_minutes,
            "total_errors": len(errors),
            "error_breakdown": error_counts,
            "sample_messages": [log.message for log in errors[:5]],
        }
    
    def find_root_cause_patterns(self, service: str) -> list[dict]:
        """Analyze logs to find potential root cause patterns."""
        errors = self.search_logs(service, "ERROR", 60)
        
        patterns = []
        
        # Count error types
        error_types = {}
        for log in errors:
            et = log.error_type or "unknown"
            error_types[et] = error_types.get(et, 0) + 1
        
        # Find most common
        if error_types:
            sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
            
            for error_type, count in sorted_errors[:3]:
                confidence = min(95, 50 + count * 5)
                patterns.append({
                    "pattern": error_type,
                    "count": count,
                    "confidence": confidence,
                    "description": f"High occurrence of {error_type} errors ({count} in last hour)",
                })
        
        return patterns


# =============================================================================
# Infrastructure Integration
# =============================================================================

class MockInfrastructure:
    """
    Mock infrastructure management (Kubernetes/AWS/GCP).
    """
    
    def __init__(self):
        self._setup_mock_infra()
    
    def _setup_mock_infra(self):
        """Setup mock infrastructure state."""
        self.services = {
            "api-gateway": Service(
                name="api-gateway",
                service_type=ServiceType.GATEWAY,
                namespace="production",
                replicas=5,
                desired_replicas=5,
                healthy_pods=5,
                cpu_usage=45,
                memory_usage=60,
                version="2.3.1",
                endpoint="https://api.example.com",
                dependencies=["user-service", "order-service"],
            ),
            "user-service": Service(
                name="user-service",
                service_type=ServiceType.API,
                namespace="production",
                replicas=3,
                desired_replicas=3,
                healthy_pods=3,
                cpu_usage=30,
                memory_usage=45,
                version="1.5.2",
                dependencies=["postgres-primary", "redis-cache"],
            ),
            "order-service": Service(
                name="order-service",
                service_type=ServiceType.API,
                namespace="production",
                replicas=4,
                desired_replicas=4,
                healthy_pods=4,
                cpu_usage=55,
                memory_usage=70,
                version="3.1.0",
                dependencies=["postgres-primary", "payment-service"],
            ),
            "payment-service": Service(
                name="payment-service",
                service_type=ServiceType.API,
                namespace="production",
                replicas=2,
                desired_replicas=2,
                healthy_pods=2,
                cpu_usage=25,
                memory_usage=40,
                version="2.0.5",
                dependencies=["postgres-primary"],
            ),
            "postgres-primary": Service(
                name="postgres-primary",
                service_type=ServiceType.DATABASE,
                namespace="data",
                replicas=1,
                desired_replicas=1,
                healthy_pods=1,
                cpu_usage=40,
                memory_usage=75,
                version="14.5",
            ),
            "redis-cache": Service(
                name="redis-cache",
                service_type=ServiceType.CACHE,
                namespace="data",
                replicas=3,
                desired_replicas=3,
                healthy_pods=3,
                cpu_usage=15,
                memory_usage=65,
                version="7.0",
            ),
        }
        
        self.deployment_history = {
            "api-gateway": ["2.3.1", "2.3.0", "2.2.5", "2.2.4"],
            "user-service": ["1.5.2", "1.5.1", "1.5.0", "1.4.9"],
            "order-service": ["3.1.0", "3.0.5", "3.0.4", "3.0.3"],
        }
    
    def get_service(self, name: str) -> Optional[Service]:
        """Get service details."""
        return self.services.get(name)
    
    def get_all_services(self) -> list[Service]:
        """Get all services."""
        return list(self.services.values())
    
    def scale_service(self, name: str, replicas: int) -> dict:
        """Scale a service."""
        if name not in self.services:
            return {"success": False, "error": f"Service {name} not found"}
        
        service = self.services[name]
        old_replicas = service.desired_replicas
        service.desired_replicas = replicas
        service.replicas = replicas
        service.healthy_pods = replicas
        
        return {
            "success": True,
            "service": name,
            "previous_replicas": old_replicas,
            "new_replicas": replicas,
            "message": f"Scaled {name} from {old_replicas} to {replicas} replicas",
        }
    
    def restart_service(self, name: str) -> dict:
        """Restart a service (rolling restart)."""
        if name not in self.services:
            return {"success": False, "error": f"Service {name} not found"}
        
        service = self.services[name]
        
        return {
            "success": True,
            "service": name,
            "pods_restarted": service.replicas,
            "message": f"Rolling restart initiated for {name}",
        }
    
    def rollback_service(self, name: str, version: str = None) -> dict:
        """Rollback service to previous version."""
        if name not in self.services:
            return {"success": False, "error": f"Service {name} not found"}
        
        history = self.deployment_history.get(name, [])
        
        if not history or len(history) < 2:
            return {"success": False, "error": "No previous version available"}
        
        service = self.services[name]
        old_version = service.version
        
        # Rollback to specified version or previous
        if version and version in history:
            new_version = version
        else:
            new_version = history[1]  # Previous version
        
        service.version = new_version
        
        return {
            "success": True,
            "service": name,
            "previous_version": old_version,
            "new_version": new_version,
            "message": f"Rolled back {name} from {old_version} to {new_version}",
        }
    
    def simulate_failure(self, name: str, failure_type: str = "pod_crash"):
        """Simulate infrastructure failure."""
        if name not in self.services:
            return
        
        service = self.services[name]
        
        if failure_type == "pod_crash":
            service.unhealthy_pods = min(service.replicas, 2)
            service.healthy_pods = service.replicas - service.unhealthy_pods
        elif failure_type == "high_cpu":
            service.cpu_usage = 95
        elif failure_type == "high_memory":
            service.memory_usage = 95


# =============================================================================
# Communication Integration
# =============================================================================

class MockCommunication:
    """
    Mock communication system (Slack/PagerDuty/Statuspage).
    """
    
    def __init__(self):
        self.messages_sent = []
        self.pages_sent = []
        self.status_updates = []
    
    def send_slack_message(self, channel: str, message: str, priority: str = "normal") -> dict:
        """Send Slack message."""
        msg = {
            "channel": channel,
            "message": message,
            "priority": priority,
            "sent_at": datetime.now().isoformat(),
        }
        self.messages_sent.append(msg)
        
        return {
            "success": True,
            "channel": channel,
            "message_id": f"msg-{uuid.uuid4().hex[:8]}",
        }
    
    def page_oncall(self, team: str, message: str, severity: Severity) -> dict:
        """Page on-call engineer."""
        page = {
            "team": team,
            "message": message,
            "severity": severity.value,
            "paged_at": datetime.now().isoformat(),
        }
        self.pages_sent.append(page)
        
        return {
            "success": True,
            "team": team,
            "oncall_engineer": "John Smith",
            "page_id": f"page-{uuid.uuid4().hex[:8]}",
        }
    
    def update_status_page(
        self,
        status: str,
        message: str,
        affected_components: list[str],
    ) -> dict:
        """Update status page."""
        update = {
            "status": status,
            "message": message,
            "affected_components": affected_components,
            "updated_at": datetime.now().isoformat(),
        }
        self.status_updates.append(update)
        
        return {
            "success": True,
            "status": status,
            "update_id": f"status-{uuid.uuid4().hex[:8]}",
            "url": "https://status.example.com",
        }
    
    def send_email(self, recipients: list[str], subject: str, body: str) -> dict:
        """Send email notification."""
        return {
            "success": True,
            "recipients": recipients,
            "subject": subject,
            "message_id": f"email-{uuid.uuid4().hex[:8]}",
        }
    
    def create_jira_ticket(self, title: str, description: str, priority: str) -> dict:
        """Create Jira ticket for follow-up."""
        return {
            "success": True,
            "ticket_id": f"INC-{random.randint(1000, 9999)}",
            "title": title,
            "url": f"https://jira.example.com/browse/INC-{random.randint(1000, 9999)}",
        }


# =============================================================================
# Runbook Repository
# =============================================================================

class RunbookRepository:
    """Repository of automated runbooks."""
    
    def __init__(self):
        self.runbooks = self._create_runbooks()
    
    def _create_runbooks(self) -> dict[str, Runbook]:
        """Create standard runbooks."""
        return {
            "high_error_rate": Runbook(
                runbook_id="rb-001",
                name="High Error Rate Response",
                description="Automated response to high error rates",
                target_alerts=["high_error_rate"],
                auto_execute=True,
                max_auto_severity=Severity.SEV3,
                steps=[
                    RunbookStep(
                        step_id="1",
                        description="Scale up service by 50%",
                        action_type=ActionType.SCALE_UP,
                        parameters={"scale_factor": 1.5},
                        estimated_duration=60,
                    ),
                    RunbookStep(
                        step_id="2",
                        description="Check if errors decrease",
                        action_type=ActionType.MANUAL,
                        verification_command="check_error_rate",
                        expected_result="error_rate < 5%",
                        estimated_duration=120,
                    ),
                    RunbookStep(
                        step_id="3",
                        description="Rollback if recent deployment",
                        action_type=ActionType.ROLLBACK,
                        condition="recent_deployment < 1h",
                        requires_approval=True,
                        estimated_duration=180,
                    ),
                ],
            ),
            "high_latency": Runbook(
                runbook_id="rb-002",
                name="High Latency Response",
                description="Automated response to high latency",
                target_alerts=["high_latency"],
                auto_execute=True,
                max_auto_severity=Severity.SEV3,
                steps=[
                    RunbookStep(
                        step_id="1",
                        description="Clear application cache",
                        action_type=ActionType.CLEAR_CACHE,
                        estimated_duration=30,
                    ),
                    RunbookStep(
                        step_id="2",
                        description="Scale up service",
                        action_type=ActionType.SCALE_UP,
                        parameters={"scale_factor": 2.0},
                        estimated_duration=60,
                    ),
                ],
            ),
            "service_down": Runbook(
                runbook_id="rb-003",
                name="Service Down Response",
                description="Response when service is completely down",
                target_alerts=["service_down"],
                auto_execute=False,  # SEV1 requires human
                steps=[
                    RunbookStep(
                        step_id="1",
                        description="Attempt service restart",
                        action_type=ActionType.RESTART,
                        estimated_duration=120,
                    ),
                    RunbookStep(
                        step_id="2",
                        description="Rollback to last known good version",
                        action_type=ActionType.ROLLBACK,
                        requires_approval=True,
                        is_destructive=True,
                        estimated_duration=180,
                    ),
                    RunbookStep(
                        step_id="3",
                        description="Failover to backup",
                        action_type=ActionType.FAILOVER,
                        requires_approval=True,
                        is_destructive=True,
                        estimated_duration=300,
                    ),
                ],
            ),
            "database_issues": Runbook(
                runbook_id="rb-004",
                name="Database Issues Response",
                description="Response to database connection/performance issues",
                target_services=["postgres-primary"],
                auto_execute=False,
                steps=[
                    RunbookStep(
                        step_id="1",
                        description="Kill long-running queries",
                        action_type=ActionType.MANUAL,
                        estimated_duration=60,
                    ),
                    RunbookStep(
                        step_id="2",
                        description="Scale connection pooler",
                        action_type=ActionType.SCALE_UP,
                        parameters={"target": "pgbouncer"},
                        estimated_duration=60,
                    ),
                ],
            ),
        }
    
    def get_runbook(self, runbook_id: str) -> Optional[Runbook]:
        """Get runbook by ID."""
        return self.runbooks.get(runbook_id)
    
    def find_runbook_for_alert(self, alert_type: str) -> Optional[Runbook]:
        """Find appropriate runbook for alert type."""
        for runbook in self.runbooks.values():
            if alert_type in runbook.target_alerts:
                return runbook
        return None
    
    def get_all_runbooks(self) -> list[Runbook]:
        """Get all runbooks."""
        return list(self.runbooks.values())


# =============================================================================
# Global Instances
# =============================================================================

monitoring = MockMonitoring()
logging_system = MockLogging()
infrastructure = MockInfrastructure()
communication = MockCommunication()
runbook_repo = RunbookRepository()
