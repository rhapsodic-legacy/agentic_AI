# 🚨 DevOps Incident Response System

An **AutoGen-powered** autonomous incident response system that detects issues, diagnoses root causes, implements fixes, and coordinates communication during outages.

![AutoGen](https://img.shields.io/badge/Framework-AutoGen-blue)
![Architecture](https://img.shields.io/badge/Architecture-Hierarchical-green)
![Complexity](https://img.shields.io/badge/Complexity-⭐⭐⭐⭐⭐-yellow)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📊 **Real-time Monitoring** | Alert ingestion and metric tracking |
| 🔍 **Log Analysis** | Pattern recognition and root cause |
| 📚 **Runbook Execution** | Automated remediation steps |
| ☸️ **Kubernetes Ops** | Scale, rollback, restart |
| 💾 **Database Health** | Connection pool management |
| 📢 **Status Updates** | Slack, PagerDuty, Statuspage |
| 👥 **Stakeholder Comms** | Automated notifications |
| 📝 **Post-Mortems** | Auto-generated incident reports |

## 🏗️ Organization Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    INCIDENT COMMANDER                           │
│         (Coordinates response, makes decisions)                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
      ┌─────────────────────┼─────────────────────────────────────┐
      │                     │                                     │
      ▼                     ▼                                     ▼
┌───────────┐        ┌───────────┐                        ┌───────────┐
│  MONITOR  │        │ DIAGNOSE  │                        │  COMMS    │
│  AGENT    │        │  AGENT    │                        │  AGENT    │
│           │        │           │                        │           │
│ • Alerts  │        │ • Logs    │                        │ • Slack   │
│ • Metrics │        │ • Traces  │                        │ • Status  │
│ • Health  │        │ • Queries │                        │ • Email   │
└─────┬─────┘        └─────┬─────┘                        └───────────┘
      │                    │
      │              ┌─────┴─────┐
      │              │           │
      │              ▼           ▼
      │        ┌───────────┐ ┌───────────┐
      │        │  INFRA    │ │   APP     │
      │        │  FIXER    │ │  FIXER    │
      │        │           │ │           │
      │        │• K8s      │ │• Rollback │
      │        │• Scaling  │ │• Restart  │
      │        │• DNS      │ │• Config   │
      │        └───────────┘ └───────────┘
      │
      └────────────────(Feedback Loop)
```

## 🔄 Incident Response Flow

```
1. DETECTION
   Alert received → Monitor Agent validates severity

2. TRIAGE  
   Incident Commander assigns severity (SEV1-4)
   Notifies on-call and stakeholders

3. DIAGNOSIS
   Diagnose Agent queries logs, metrics, traces
   Identifies potential root cause

4. MITIGATION
   Fixer Agents implement remediation
   - Scale up pods
   - Rollback deployment
   - Clear cache
   - Restart services

5. VERIFICATION
   Monitor Agent confirms metrics normalized

6. COMMUNICATION
   Comms Agent updates status page
   Sends all-clear notification

7. POST-MORTEM
   Auto-generates incident report with timeline
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt

# Set API key
export GOOGLE_API_KEY="your-key"
```

### CLI Usage

```bash
# Simulate and respond to an incident
python main.py simulate api-gateway high_error_rate

# With post-mortem generation
python main.py simulate api-gateway service_down --postmortem

# Check system health
python main.py health

# List runbooks
python main.py runbooks

# Interactive mode
python main.py interactive

# Web server
python main.py serve
```

### Python API

```python
from incident_system import IncidentResponseSystem, simulate_and_respond

# Create system
system = IncidentResponseSystem()

# Simulate an incident
alert = system.simulate_incident("api-gateway", "high_error_rate")

# Respond to the incident
result = system.respond_to_incident(alert)

# View incident summary
print(result.incident.to_summary())

# Generate post-mortem
postmortem = system.generate_postmortem(result.incident.incident_id)
print(postmortem)
```

## 📁 Project Structure

```
incident-response/
├── incident_system/
│   ├── __init__.py           # Package exports
│   ├── response.py           # Main IncidentResponseSystem
│   ├── agents/
│   │   └── __init__.py       # 6 AutoGen agents
│   ├── tools/
│   │   └── __init__.py       # 20+ incident tools
│   ├── models/
│   │   └── __init__.py       # Incident, Alert, Runbook models
│   ├── integrations/
│   │   └── __init__.py       # Mock monitoring/infra/comms
│   └── runbooks/
├── api.py                     # FastAPI backend
├── frontend/
│   └── index.html            # React dashboard
├── main.py                    # CLI
├── requirements.txt
└── README.md
```

## 🤖 Agent Roles

| Agent | Responsibilities |
|-------|-----------------|
| **Incident Commander** | Triage, coordination, decisions, escalation |
| **Monitor Agent** | Metrics, alerts, health checks, recovery verification |
| **Diagnose Agent** | Log analysis, root cause, pattern recognition |
| **Infra Fixer** | K8s scaling, restarts, infrastructure ops |
| **App Fixer** | Rollbacks, application restarts, config updates |
| **Comms Agent** | Slack, PagerDuty, status page, stakeholder comms |

## 🔧 Available Tools

### Monitoring
```python
get_service_metrics(service)      # Current metrics
check_service_health(service)     # Health check
get_all_services_health()         # System-wide health
get_recent_alerts()               # Recent alerts
```

### Diagnosis
```python
search_error_logs(service, time)  # Error logs
get_error_summary(service)        # Error breakdown
find_root_cause(service)          # Root cause analysis
check_dependencies(service)       # Dependency health
```

### Remediation
```python
scale_service(service, replicas)  # Scale up/down
restart_service(service)          # Rolling restart
rollback_service(service, ver)    # Rollback deployment
list_available_runbooks()         # List runbooks
```

### Communication
```python
send_slack_alert(channel, msg)    # Slack notification
page_oncall_team(team, msg, sev)  # Page on-call
update_status_page(status, msg)   # Status page update
create_incident_ticket(title)     # Jira ticket
notify_stakeholders(id, msg)      # Email stakeholders
```

## 📚 Integrations (Mock)

```python
INTEGRATIONS = {
    "monitoring": ["Datadog", "Prometheus", "Grafana", "New Relic"],
    "logging": ["Elasticsearch", "Splunk", "CloudWatch"],
    "infrastructure": ["Kubernetes", "AWS", "GCP", "Terraform"],
    "communication": ["Slack", "PagerDuty", "Statuspage", "Email"],
    "ticketing": ["Jira", "ServiceNow", "Linear"],
}
```

## 🎯 Incident Types

| Type | Severity | Description |
|------|----------|-------------|
| `high_error_rate` | SEV2 | Error rate exceeds threshold |
| `high_latency` | SEV2 | P99 latency too high |
| `service_down` | SEV1 | Service health checks failing |
| `high_cpu` | SEV3 | CPU usage critical |
| `database_connections` | SEV1 | Connection pool exhausted |

## 📊 Severity Levels

| Level | Impact | Response |
|-------|--------|----------|
| **SEV1** | Major outage, all users | Immediate, all hands |
| **SEV2** | Significant impact, subset | Urgent, primary team |
| **SEV3** | Minor impact, workaround | Standard, on-call |
| **SEV4** | Minimal impact | Scheduled |

## 🌐 Web API

```bash
python main.py serve
```

Endpoints:
- `GET /api/status` - System status
- `GET /api/health` - All services health
- `GET /api/health/{service}` - Service health
- `GET /api/services` - List services
- `POST /api/simulate` - Simulate incident
- `GET /api/incident/{id}` - Incident status
- `GET /api/incidents` - Recent incidents
- `POST /api/action` - Execute action
- `GET /api/runbooks` - List runbooks

## 📝 Sample Output

```
============================================================
🚨 INCIDENT DETECTED: INC-20240115-A1B2C3
   Severity: SEV2
   Service: api-gateway
   Alert: High Error Rate on api-gateway
============================================================

📋 Phase 1: TRIAGE
   ✓ Severity: SEV2
   ✓ Impact: Significant subset of users affected

🔍 Phase 2: DIAGNOSIS
   ✓ Root Cause: Elevated error rates due to recent deployment

🔧 Phase 3: MITIGATION
   ✅ Scaled api-gateway from 5 to 10 replicas
   ✅ Rolling restart initiated for api-gateway

✅ Phase 4: VERIFICATION
   Post-mitigation health: 🟢 HEALTHY

📢 Phase 5: COMMUNICATION
   ✅ Message sent to #incidents
   ✅ Status page updated

============================================================
✅ INCIDENT RESOLVED: INC-20240115-A1B2C3
   Duration: 5 minutes
   Actions taken: 3
============================================================
```

## 📝 License

MIT License
