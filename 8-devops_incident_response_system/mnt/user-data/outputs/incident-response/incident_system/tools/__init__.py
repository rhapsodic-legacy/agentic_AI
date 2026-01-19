"""
DevOps Incident Response System - Tools

Tools for:
- Monitoring and alerting
- Log analysis and diagnosis
- Infrastructure remediation
- Communication and reporting
"""

from typing import Optional
from datetime import datetime
import json

from ..models import (
    Alert, Metric, LogEntry, Service, Incident, RemediationAction,
    Severity, IncidentStatus, ActionType, PostMortem
)
from ..integrations import (
    monitoring, logging_system, infrastructure, communication, runbook_repo
)


# =============================================================================
# Monitoring Tools
# =============================================================================

def get_service_metrics(service: str) -> str:
    """
    Get current metrics for a service.
    
    Args:
        service: Service name (e.g., 'api-gateway', 'user-service')
    
    Returns:
        Formatted metrics summary
    """
    metrics = monitoring.get_metrics(service)
    
    if not metrics:
        return f"No metrics found for service: {service}"
    
    result = f"**Metrics for {service}:**\n\n"
    
    for metric in metrics:
        status = "🟢"
        if metric.is_critical:
            status = "🔴"
        elif metric.is_warning:
            status = "🟡"
        
        result += f"{status} **{metric.name}**: {metric.value:.2f}{metric.unit}\n"
        
        if metric.warning_threshold:
            result += f"   Warning: {metric.warning_threshold}{metric.unit} | Critical: {metric.critical_threshold}{metric.unit}\n"
    
    return result


def check_service_health(service: str) -> str:
    """
    Perform health check on a service.
    
    Args:
        service: Service name
    
    Returns:
        Health check results
    """
    health = monitoring.check_health(service)
    
    status_emoji = {"healthy": "🟢", "warning": "🟡", "critical": "🔴"}
    
    result = f"**Health Check: {service}**\n\n"
    result += f"Status: {status_emoji.get(health['status'], '⚪')} {health['status'].upper()}\n\n"
    
    if health['issues']:
        result += "**Issues Found:**\n"
        for issue in health['issues']:
            result += f"  ⚠️ {issue}\n"
    else:
        result += "No issues detected.\n"
    
    return result


def get_all_services_health() -> str:
    """
    Get health status of all services.
    
    Returns:
        Summary of all service health
    """
    services = infrastructure.get_all_services()
    
    result = "**Infrastructure Health Summary:**\n\n"
    
    healthy = 0
    unhealthy = 0
    
    for service in services:
        health = monitoring.check_health(service.name)
        status = health['status']
        
        if status == "healthy":
            healthy += 1
            result += f"🟢 {service.name}: Healthy\n"
        elif status == "warning":
            result += f"🟡 {service.name}: Warning - {', '.join(health['issues'][:2])}\n"
        else:
            unhealthy += 1
            result += f"🔴 {service.name}: Critical - {', '.join(health['issues'][:2])}\n"
    
    result += f"\n**Summary:** {healthy} healthy, {unhealthy} unhealthy\n"
    
    return result


def get_recent_alerts() -> str:
    """
    Get recent alerts from monitoring systems.
    
    Returns:
        List of recent alerts
    """
    alerts = monitoring.alerts_triggered[-10:]  # Last 10 alerts
    
    if not alerts:
        return "No recent alerts."
    
    result = "**Recent Alerts:**\n\n"
    
    for alert in reversed(alerts):
        sev_emoji = {"SEV1": "🔴", "SEV2": "🟠", "SEV3": "🟡", "SEV4": "🔵"}
        emoji = sev_emoji.get(alert.severity.value, "⚪")
        
        result += f"{emoji} **{alert.severity.value}** - {alert.title}\n"
        result += f"   Service: {alert.service} | Time: {alert.triggered_at}\n"
        result += f"   {alert.description}\n\n"
    
    return result


# =============================================================================
# Diagnosis Tools
# =============================================================================

def search_error_logs(service: str, time_range_minutes: int = 30) -> str:
    """
    Search error logs for a service.
    
    Args:
        service: Service name
        time_range_minutes: Time range to search
    
    Returns:
        Error log summary
    """
    logs = logging_system.search_logs(service, "ERROR", time_range_minutes)
    
    if not logs:
        return f"No error logs found for {service} in the last {time_range_minutes} minutes."
    
    result = f"**Error Logs for {service}** (last {time_range_minutes} min):\n\n"
    result += f"Found **{len(logs)}** error entries.\n\n"
    
    # Show sample errors
    result += "**Sample Errors:**\n"
    for log in logs[:5]:
        result += f"  `{log.timestamp}` [{log.level}] {log.message[:100]}\n"
    
    return result


def get_error_summary(service: str) -> str:
    """
    Get error summary with breakdown by type.
    
    Args:
        service: Service name
    
    Returns:
        Error summary with counts
    """
    summary = logging_system.get_error_summary(service)
    
    result = f"**Error Summary for {service}:**\n\n"
    result += f"Total errors (last hour): **{summary['total_errors']}**\n\n"
    
    if summary['error_breakdown']:
        result += "**Error Breakdown:**\n"
        for error_type, count in sorted(summary['error_breakdown'].items(), key=lambda x: x[1], reverse=True):
            result += f"  • {error_type}: {count}\n"
    
    return result


def find_root_cause(service: str) -> str:
    """
    Analyze logs and metrics to identify root cause.
    
    Args:
        service: Service name to analyze
    
    Returns:
        Root cause analysis
    """
    patterns = logging_system.find_root_cause_patterns(service)
    health = monitoring.check_health(service)
    svc = infrastructure.get_service(service)
    
    result = f"**Root Cause Analysis for {service}:**\n\n"
    
    # Check for common issues
    result += "**Potential Root Causes:**\n\n"
    
    # From log patterns
    if patterns:
        result += "Based on log analysis:\n"
        for i, pattern in enumerate(patterns, 1):
            result += f"  {i}. **{pattern['pattern']}** - {pattern['description']}\n"
            result += f"     Confidence: {pattern['confidence']}%\n"
    
    # From metrics
    if health['issues']:
        result += "\nBased on metrics:\n"
        for issue in health['issues']:
            result += f"  • {issue}\n"
    
    # Check dependencies
    if svc and svc.dependencies:
        result += f"\n**Dependencies to check:** {', '.join(svc.dependencies)}\n"
    
    # Recent deployments
    if svc:
        result += f"\n**Current Version:** {svc.version}\n"
        if service in infrastructure.deployment_history:
            prev_version = infrastructure.deployment_history[service][1] if len(infrastructure.deployment_history[service]) > 1 else "N/A"
            result += f"**Previous Version:** {prev_version}\n"
    
    return result


def check_dependencies(service: str) -> str:
    """
    Check health of service dependencies.
    
    Args:
        service: Service name
    
    Returns:
        Dependency health status
    """
    svc = infrastructure.get_service(service)
    
    if not svc:
        return f"Service not found: {service}"
    
    if not svc.dependencies:
        return f"No dependencies configured for {service}"
    
    result = f"**Dependencies for {service}:**\n\n"
    
    for dep in svc.dependencies:
        dep_health = monitoring.check_health(dep)
        status = dep_health['status']
        
        emoji = {"healthy": "🟢", "warning": "🟡", "critical": "🔴"}.get(status, "⚪")
        
        result += f"{emoji} **{dep}**: {status}\n"
        
        if dep_health['issues']:
            for issue in dep_health['issues'][:2]:
                result += f"   ⚠️ {issue}\n"
    
    return result


# =============================================================================
# Infrastructure Remediation Tools
# =============================================================================

def scale_service(service: str, replicas: int) -> str:
    """
    Scale a service to specified replicas.
    
    Args:
        service: Service name
        replicas: Target replica count
    
    Returns:
        Scaling result
    """
    result = infrastructure.scale_service(service, replicas)
    
    if result['success']:
        return f"✅ **Scaled {service}** from {result['previous_replicas']} to {result['new_replicas']} replicas"
    else:
        return f"❌ **Failed to scale {service}**: {result.get('error', 'Unknown error')}"


def restart_service(service: str) -> str:
    """
    Perform rolling restart of a service.
    
    Args:
        service: Service name
    
    Returns:
        Restart result
    """
    result = infrastructure.restart_service(service)
    
    if result['success']:
        return f"✅ **Rolling restart initiated** for {service} ({result['pods_restarted']} pods)"
    else:
        return f"❌ **Failed to restart {service}**: {result.get('error', 'Unknown error')}"


def rollback_service(service: str, version: str = None) -> str:
    """
    Rollback service to previous version.
    
    Args:
        service: Service name
        version: Target version (optional, defaults to previous)
    
    Returns:
        Rollback result
    """
    result = infrastructure.rollback_service(service, version)
    
    if result['success']:
        return f"✅ **Rolled back {service}** from {result['previous_version']} to {result['new_version']}"
    else:
        return f"❌ **Failed to rollback {service}**: {result.get('error', 'Unknown error')}"


def get_service_info(service: str) -> str:
    """
    Get detailed service information.
    
    Args:
        service: Service name
    
    Returns:
        Service details
    """
    svc = infrastructure.get_service(service)
    
    if not svc:
        return f"Service not found: {service}"
    
    health_emoji = "🟢" if svc.is_healthy else "🔴"
    
    result = f"""**Service: {svc.name}**

{health_emoji} **Status:** {'Healthy' if svc.is_healthy else 'Unhealthy'}
**Type:** {svc.service_type.value}
**Namespace:** {svc.namespace}

**Replicas:** {svc.healthy_pods}/{svc.desired_replicas} healthy
**CPU:** {svc.cpu_usage:.1f}%
**Memory:** {svc.memory_usage:.1f}%

**Version:** {svc.version}
**Dependencies:** {', '.join(svc.dependencies) if svc.dependencies else 'None'}
"""
    
    return result


def list_available_runbooks() -> str:
    """
    List available automated runbooks.
    
    Returns:
        List of runbooks
    """
    runbooks = runbook_repo.get_all_runbooks()
    
    result = "**Available Runbooks:**\n\n"
    
    for rb in runbooks:
        auto = "✅ Auto" if rb.auto_execute else "🔒 Manual"
        result += f"• **{rb.name}** (`{rb.runbook_id}`)\n"
        result += f"  {rb.description}\n"
        result += f"  Steps: {len(rb.steps)} | {auto}\n\n"
    
    return result


def get_runbook_for_alert(alert_type: str) -> str:
    """
    Find appropriate runbook for an alert type.
    
    Args:
        alert_type: Type of alert (e.g., 'high_error_rate')
    
    Returns:
        Runbook details
    """
    runbook = runbook_repo.find_runbook_for_alert(alert_type)
    
    if not runbook:
        return f"No runbook found for alert type: {alert_type}"
    
    result = f"**Runbook: {runbook.name}**\n\n"
    result += f"{runbook.description}\n\n"
    result += "**Steps:**\n"
    
    for step in runbook.steps:
        approval = "🔒" if step.requires_approval else ""
        destructive = "⚠️" if step.is_destructive else ""
        
        result += f"  {step.step_id}. {step.description} {approval}{destructive}\n"
        result += f"     Action: {step.action_type.value}\n"
    
    return result


# =============================================================================
# Communication Tools
# =============================================================================

def send_slack_alert(channel: str, message: str, severity: str = "normal") -> str:
    """
    Send Slack notification.
    
    Args:
        channel: Slack channel (e.g., '#incidents')
        message: Message content
        severity: Message priority
    
    Returns:
        Send result
    """
    result = communication.send_slack_message(channel, message, severity)
    
    if result['success']:
        return f"✅ Message sent to {channel}"
    else:
        return f"❌ Failed to send message to {channel}"


def page_oncall_team(team: str, message: str, severity: Severity) -> str:
    """
    Page the on-call engineer.
    
    Args:
        team: Team name (e.g., 'platform', 'backend')
        message: Page message
        severity: Incident severity
    
    Returns:
        Page result
    """
    result = communication.page_oncall(team, message, severity)
    
    if result['success']:
        return f"✅ Paged {result['oncall_engineer']} ({team} team)"
    else:
        return f"❌ Failed to page {team} team"


def update_status_page(status: str, message: str, components: list[str]) -> str:
    """
    Update public status page.
    
    Args:
        status: Status (investigating, identified, monitoring, resolved)
        message: Update message
        components: Affected components
    
    Returns:
        Update result
    """
    result = communication.update_status_page(status, message, components)
    
    if result['success']:
        return f"✅ Status page updated: {status}\nURL: {result['url']}"
    else:
        return f"❌ Failed to update status page"


def create_incident_ticket(title: str, description: str, priority: str = "high") -> str:
    """
    Create incident ticket in Jira.
    
    Args:
        title: Ticket title
        description: Ticket description
        priority: Ticket priority
    
    Returns:
        Ticket creation result
    """
    result = communication.create_jira_ticket(title, description, priority)
    
    if result['success']:
        return f"✅ Created ticket: {result['ticket_id']}\nURL: {result['url']}"
    else:
        return f"❌ Failed to create ticket"


def notify_stakeholders(incident_id: str, message: str, stakeholders: list[str] = None) -> str:
    """
    Send notification to stakeholders.
    
    Args:
        incident_id: Incident ID
        message: Notification message
        stakeholders: List of stakeholder emails
    
    Returns:
        Notification result
    """
    if stakeholders is None:
        stakeholders = ["engineering-leads@example.com", "on-call@example.com"]
    
    subject = f"[Incident {incident_id}] Update"
    
    result = communication.send_email(stakeholders, subject, message)
    
    if result['success']:
        return f"✅ Notified {len(stakeholders)} stakeholders"
    else:
        return f"❌ Failed to notify stakeholders"


# =============================================================================
# Reporting Tools
# =============================================================================

def generate_incident_summary(incident: Incident) -> str:
    """
    Generate incident summary.
    
    Args:
        incident: Incident object
    
    Returns:
        Formatted incident summary
    """
    return incident.to_summary()


def generate_postmortem(incident: Incident) -> str:
    """
    Generate post-incident report.
    
    Args:
        incident: Incident object
    
    Returns:
        Post-mortem report in markdown
    """
    duration = incident.get_duration_minutes() or 0
    
    postmortem = PostMortem(
        incident_id=incident.incident_id,
        title=incident.title,
        summary=incident.description,
        timeline=incident.timeline,
        root_cause=incident.root_cause or "Under investigation",
        contributing_factors=[
            "Root cause analysis in progress",
        ],
        duration_minutes=duration,
        users_affected=0,  # Would be calculated from metrics
        what_went_well=[
            "Alert triggered promptly",
            "Team responded quickly",
            "Communication was clear",
        ],
        what_went_wrong=[
            "Initial detection could be faster",
            "Runbook was outdated",
        ],
        action_items=[
            {"description": "Update runbook with new procedures", "owner": "Platform Team"},
            {"description": "Add additional monitoring", "owner": "SRE Team"},
        ],
        author="Incident Commander",
        created_at=datetime.now().isoformat(),
    )
    
    return postmortem.to_markdown()


# =============================================================================
# Tool Registry
# =============================================================================

TOOLS = {
    # Monitoring
    "get_service_metrics": get_service_metrics,
    "check_service_health": check_service_health,
    "get_all_services_health": get_all_services_health,
    "get_recent_alerts": get_recent_alerts,
    
    # Diagnosis
    "search_error_logs": search_error_logs,
    "get_error_summary": get_error_summary,
    "find_root_cause": find_root_cause,
    "check_dependencies": check_dependencies,
    
    # Infrastructure
    "scale_service": scale_service,
    "restart_service": restart_service,
    "rollback_service": rollback_service,
    "get_service_info": get_service_info,
    "list_available_runbooks": list_available_runbooks,
    "get_runbook_for_alert": get_runbook_for_alert,
    
    # Communication
    "send_slack_alert": send_slack_alert,
    "page_oncall_team": page_oncall_team,
    "update_status_page": update_status_page,
    "create_incident_ticket": create_incident_ticket,
    "notify_stakeholders": notify_stakeholders,
    
    # Reporting
    "generate_incident_summary": generate_incident_summary,
    "generate_postmortem": generate_postmortem,
}


def get_tools_for_role(role: str) -> dict:
    """Get tools for a specific agent role."""
    role_tools = {
        "incident_commander": [
            "get_all_services_health", "get_recent_alerts", "get_service_info",
            "page_oncall_team", "notify_stakeholders", "create_incident_ticket",
            "generate_incident_summary", "generate_postmortem",
        ],
        "monitor_agent": [
            "get_service_metrics", "check_service_health", "get_all_services_health",
            "get_recent_alerts",
        ],
        "diagnose_agent": [
            "search_error_logs", "get_error_summary", "find_root_cause",
            "check_dependencies", "get_service_metrics", "get_service_info",
        ],
        "infra_fixer": [
            "scale_service", "restart_service", "rollback_service",
            "get_service_info", "list_available_runbooks", "get_runbook_for_alert",
        ],
        "app_fixer": [
            "restart_service", "rollback_service", "get_service_info",
            "search_error_logs", "list_available_runbooks",
        ],
        "comms_agent": [
            "send_slack_alert", "page_oncall_team", "update_status_page",
            "create_incident_ticket", "notify_stakeholders",
        ],
    }
    
    tool_names = role_tools.get(role, list(TOOLS.keys()))
    return {name: TOOLS[name] for name in tool_names if name in TOOLS}
