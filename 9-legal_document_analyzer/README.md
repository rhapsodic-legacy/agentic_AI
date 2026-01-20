# ⚖️ Legal Document Analyzer

A **LangGraph-powered** state machine for analyzing legal documents, extracting key terms, identifying risks, and ensuring compliance.

![LangGraph](https://img.shields.io/badge/Framework-LangGraph-blue)
![Architecture](https://img.shields.io/badge/Architecture-State_Machine-green)
![Complexity](https://img.shields.io/badge/Complexity-⭐⭐⭐⭐-yellow)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🏷️ **Document Classification** | Auto-detects NDA, SaaS, Employment, etc. |
| 📌 **Term Extraction** | Parties, dates, obligations, financials |
| ⚠️ **Risk Scoring** | Critical/High/Medium/Low with recommendations |
| ✅ **Compliance Checking** | GDPR, CCPA, SOX, HIPAA |
| ⚖️ **Standard Comparison** | Compare to industry templates |
| 🔍 **Missing Clauses** | Detect absent required clauses |
| 📝 **Plain English Summary** | Human-readable explanations |
| 🤝 **Negotiation Suggestions** | Prioritized action items |

## 🏗️ State Machine Architecture

```
                    ┌─────────────┐
                    │   UPLOAD    │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
              ┌─────│   PARSE     │
              │     └──────┬──────┘
              │            │
              │            ▼
              │     ┌─────────────┐
              │     │  CLASSIFY   │
              │     │ (Contract   │
              │     │  Type)      │
              │     └──────┬──────┘
              │            │
              │  ┌─────────┴─────────┐
              │  │                   │
              │  ▼                   ▼
              │ ┌─────────┐   ┌─────────────┐
              │ │   NDA   │   │ EMPLOYMENT  │ ...
              │ │ ANALYZER│   │  ANALYZER   │
              │ └────┬────┘   └──────┬──────┘
              │      │               │
              │      └───────┬───────┘
              │              │
              │              ▼
              │       ┌─────────────┐
              │       │    RISK     │
              │       │  ASSESSOR   │
              │       └──────┬──────┘
              │              │
              │              ▼
              │       ┌─────────────┐
              │       │ COMPLIANCE  │
              │       │   CHECKER   │
              │       └──────┬──────┘
              │              │
              │              ▼
              │       ┌─────────────┐
              └──────►│   COMPARE   │ (Optional)
                      │ (vs Standard)│
                      └──────┬──────┘
                             │
                             ▼
                      ┌─────────────┐
                      │   REPORT    │
                      │  GENERATOR  │
                      └─────────────┘
```

## 📄 Supported Document Types

| Type | Description |
|------|-------------|
| **NDA** | Non-Disclosure Agreement |
| **SaaS** | Software as a Service Agreement |
| **Employment** | Employment Contract |
| **Lease** | Lease Agreement |
| **Purchase** | Purchase Agreement |
| **TOS** | Terms of Service |
| **Privacy** | Privacy Policy |
| **Partnership** | Partnership Agreement |
| **IP Assignment** | IP Assignment Agreement |
| **Consulting** | Consulting Agreement |

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt

# Set API key (optional, for enhanced analysis)
export GOOGLE_API_KEY="your-key"
```

### CLI Usage

```bash
# Analyze a document
python main.py analyze contract.txt

# Analyze with output file
python main.py analyze contract.txt -o report.md

# Analyze sample document
python main.py sample saas
python main.py sample nda
python main.py sample employment

# Quick classification
python main.py classify document.txt

# Interactive mode
python main.py interactive

# Web server
python main.py serve
```

### Python API

```python
from legal_analyzer import LegalDocumentAnalyzer, analyze_document

# Quick analysis
report = analyze_document(contract_text)
print(report.to_markdown())

# Full control
analyzer = LegalDocumentAnalyzer()
report = analyzer.analyze(
    contract_text,
    compare_to_standard=True,
    frameworks=[ComplianceFramework.GDPR, ComplianceFramework.CCPA],
)

# Access results
doc = report.document
print(f"Type: {doc.document_type.value}")
print(f"Risk Score: {doc.risk_assessment.risk_score}/100")

for risk in doc.risk_assessment.risks:
    print(f"- {risk.level_emoji} {risk.title}: {risk.description}")

for suggestion in doc.negotiation_suggestions:
    print(f"{suggestion.priority}. {suggestion.issue}")
```

## 📁 Project Structure

```
legal-analyzer/
├── legal_analyzer/
│   ├── __init__.py           # Package exports
│   ├── pipeline.py           # Main LangGraph pipeline
│   ├── nodes/
│   │   └── __init__.py       # State machine nodes
│   ├── tools/
│   │   └── __init__.py       # Analysis tools
│   ├── models/
│   │   └── __init__.py       # Data models
│   └── templates/
│       └── __init__.py       # Standard templates
├── api.py                     # FastAPI backend
├── frontend/
│   └── index.html            # React dashboard
├── main.py                    # CLI
├── requirements.txt
└── README.md
```

## 📊 Sample Output

```markdown
# Contract Analysis Report

## Document Overview
- **Type:** SaaS Agreement
- **Parties:** CloudTech Solutions Inc (Provider) ↔ Acme Corporation (Customer)
- **Effective Date:** January 1, 2024
- **Term:** 24 months, auto-renewing

## Key Terms Extracted
| Term | Value | Standard? |
|------|-------|-----------|
| Payment Terms | Net 45 | ⚠️ (Standard: Net 30) |
| SLA Uptime | 99.5% | ⚠️ Below industry (99.9%) |
| Liability Cap | 6 months fees | ⚠️ (Standard: 12 months) |
| Data Retention | 30 days post-termination | ⚠️ Short |

## Risk Assessment
| Risk | Level | Description |
|------|-------|-------------|
| Data breach carveout | 🔴 CRITICAL | Unlimited liability for breaches |
| Auto-renewal with 90-day notice | 🟡 MEDIUM | Consider 30-day notice |
| Unilateral amendment rights | 🟠 HIGH | Provider can change terms |

## Compliance Status
- ✅ GDPR: Data processing provisions present
- ⚠️ CCPA: Missing consumer rights section
- ✅ SOC 2: Security certification mentioned

## Recommended Negotiations
1. **Cap data breach liability** - Negotiate separate cap at 2x annual fees
2. **Increase SLA to 99.9%** - With service credits for downtime
3. **Reduce auto-renewal notice** - Request 30-day notice period
4. **Require mutual amendment consent** - Remove unilateral rights
5. **Extend liability cap** - Request 12 months of fees
```

## ⚠️ Risk Patterns Detected

| Pattern | Risk Level | Description |
|---------|------------|-------------|
| `unlimited_liability` | 🔴 Critical | No liability cap |
| `data_breach_carveout` | 🔴 Critical | Breach excluded from cap |
| `broad_indemnification` | 🟠 High | Wide indemnification scope |
| `unilateral_amendment` | 🟠 High | One-sided change rights |
| `one_sided_termination` | 🟠 High | Unequal termination rights |
| `auto_renewal` | 🟡 Medium | Auto-renewal clause |
| `short_warranty` | 🟡 Medium | Limited warranty period |
| `no_audit_rights` | 🟡 Medium | Cannot verify compliance |

## ✅ Compliance Frameworks

| Framework | Requirements Checked |
|-----------|---------------------|
| **GDPR** | DPA, Data Subject Rights, International Transfers, Security, Breach Notification |
| **CCPA** | Consumer Rights Notice, Sale of PI, Service Provider Terms |
| **SOX** | Internal Controls, Record Retention, Access Controls |
| **HIPAA** | BAA, Security Safeguards, Breach Notification |

## 🌐 Web API

```bash
python main.py serve
```

Endpoints:
- `POST /api/analyze` - Analyze document (async)
- `GET /api/analysis/{id}` - Get analysis result
- `POST /api/classify` - Quick classification
- `GET /api/sample/{type}` - Get sample document
- `GET /api/document-types` - List document types
- `GET /api/frameworks` - List compliance frameworks

## ⚙️ Configuration

```python
from legal_analyzer import LegalDocumentAnalyzer, AnalyzerConfig, ComplianceFramework

config = AnalyzerConfig(
    llm_provider="gemini",        # "gemini", "anthropic", "openai"
    compare_to_standard=True,     # Compare to industry templates
    frameworks=[                  # Compliance frameworks to check
        ComplianceFramework.GDPR,
        ComplianceFramework.CCPA,
        ComplianceFramework.SOX,
    ],
    verbose=True,
)

analyzer = LegalDocumentAnalyzer(config)
```

## 📝 License

MIT License

Disclaimer: This is for educational purposes only and not legal advice.