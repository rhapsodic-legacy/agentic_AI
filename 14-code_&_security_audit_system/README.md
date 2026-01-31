# 🔒 Code Review & Security Audit System

An **automated code review system** that checks code quality, identifies security vulnerabilities, ensures best practices, and provides improvement suggestions using a pipeline architecture.

![AutoGen](https://img.shields.io/badge/Framework-AutoGen-blue)
![Architecture](https://img.shields.io/badge/Architecture-Pipeline-purple)
![OWASP](https://img.shields.io/badge/Security-OWASP_Top_10-red)
![Complexity](https://img.shields.io/badge/Complexity-⭐⭐⭐⭐-yellow)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔒 **Security Scanning** | OWASP Top 10 vulnerability detection |
| 📝 **Syntax Analysis** | Error detection and code validation |
| 🎨 **Style Checking** | PEP8/ESLint-style best practices |
| ⚡ **Performance Analysis** | Anti-pattern detection |
| 📊 **Complexity Metrics** | Cyclomatic complexity, maintainability index |
| 📚 **Documentation Check** | Docstring coverage analysis |
| 🔧 **Auto-Fix Suggestions** | Specific remediation code |
| 📦 **Dependency Scanning** | Known CVE detection |

## 🏗️ Pipeline Architecture

```
┌─────────────┐
│   CODE      │
│   INPUT     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                    PARALLEL ANALYSIS                        │
│                                                             │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐│
│  │  SYNTAX   │  │ SECURITY  │  │   STYLE   │  │   PERF    ││
│  │  CHECKER  │  │  SCANNER  │  │  CHECKER  │  │  ANALYZER ││
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘│
└────────┴──────────────┴──────────────┴──────────────┴───────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     AGGREGATOR      │
                    │  (Combine Findings) │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   PRIORITIZER       │
                    │ (Rank by Severity)  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   FIX SUGGESTER     │
                    │ (Generate Patches)  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   REPORT GENERATOR  │
                    └─────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Command Line

```bash
# Review a file or directory
python main.py review ./src
python main.py review auth_service.py

# Review with options
python main.py review ./src --no-style --save

# Review and fail CI on critical issues
python main.py review ./src --fail-on-critical

# List available security checks
python main.py checks

# Start web server
python main.py serve
```

### Python API

```python
from code_review import CodeReviewEngine, ReviewConfig

# Configure
config = ReviewConfig(
    check_security=True,
    check_style=True,
    check_performance=True,
)

# Run review
engine = CodeReviewEngine(config)
report = engine.review("./src")

# Get results
print(f"Found {report.analysis.issue_count} issues")
print(f"Critical: {report.analysis.critical_count}")

# Save report
engine.save_report(report, format="markdown")
```

### Quick Review

```python
from code_review import quick_review, review_code

# Review a path
report = quick_review("./my_project")

# Review code string
report = review_code("""
password = "hardcoded123"
query = f"SELECT * FROM users WHERE id = '{user_id}'"
""", language="python")
```

## 📁 Project Structure

```
code-review-system/
├── code_review/
│   ├── __init__.py           # Main engine
│   ├── agents/               # AutoGen pipeline agents
│   │   └── __init__.py       # Agent definitions
│   ├── analyzers/
│   │   ├── security.py       # Security scanner (OWASP)
│   │   └── quality.py        # Style, performance, complexity
│   └── models/
│       └── __init__.py       # Data models
├── frontend/
│   └── index.html            # React web interface
├── samples/
│   └── vulnerable_auth.py    # Test vulnerable code
├── main.py                   # Rich CLI
├── api.py                    # FastAPI backend
├── requirements.txt
└── README.md
```

## 🔒 Security Checks

### OWASP Top 10 Coverage

| Category | Checks |
|----------|--------|
| **A03: Injection** | SQL Injection, Command Injection, Code Injection |
| **A02: Crypto Failures** | Weak hashes (MD5, SHA1), Insecure random |
| **A07: XSS** | innerHTML, document.write, dangerouslySetInnerHTML |
| **A09: Logging** | Sensitive data in logs |
| **A05: Security Misconfig** | Debug mode, CORS misconfiguration |
| **A08: Insecure Deserialization** | pickle, unsafe YAML |
| **A06: Vulnerable Components** | Known CVEs in dependencies |

### Hardcoded Secrets Detection

- Passwords and credentials
- API keys
- AWS access keys
- Private keys (RSA, etc.)
- JWT secrets

### CWE IDs

All security findings include CWE (Common Weakness Enumeration) identifiers:

- **CWE-89**: SQL Injection
- **CWE-78**: OS Command Injection
- **CWE-79**: Cross-site Scripting (XSS)
- **CWE-798**: Hardcoded Credentials
- **CWE-328**: Weak Hash
- **CWE-330**: Insufficient Random
- **CWE-502**: Insecure Deserialization

## 📊 Sample Report

```markdown
# Code Review Report: auth_service.py

## Summary
- **Files Analyzed:** 1
- **Total Lines:** 85
- **Total Issues:** 12
- **Critical:** 4 | **High:** 3 | **Medium:** 3 | **Low:** 2

## 🔴 Critical Issues

### 1. SQL Injection Vulnerability
📍 `auth_service.py:25`

**Description:** User input is directly concatenated into SQL query

```python
# ❌ Vulnerable
query = f"SELECT * FROM users WHERE username = '{username}'"

# ✅ Fixed
query = "SELECT * FROM users WHERE username = %s"
cursor.execute(query, (username,))
```

**CWE:** CWE-89
**Impact:** Remote code execution, data breach

### 2. Hardcoded Secret Key
📍 `auth_service.py:8`

```python
# ❌ Vulnerable
SECRET_KEY = "super_secret_key_123"

# ✅ Fixed
SECRET_KEY = os.environ.get("SECRET_KEY")
```

**CWE:** CWE-798
**Impact:** Authentication bypass

## Code Quality
- **Complexity Score:** 7.2/10 (Good)
- **Test Coverage:** N/A
- **Documentation:** 45% (Below standard)

## 💡 Recommendations
1. Address all security vulnerabilities before deployment
2. Add input validation middleware
3. Implement rate limiting
4. Add unit tests for auth functions
5. Update dependencies (2 outdated)
```

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | API status |
| `/api/review` | POST | Review code snippet |
| `/api/review/file` | POST | Review uploaded file |
| `/api/review/{id}` | GET | Get review result |
| `/api/review/{id}/markdown` | GET | Get as markdown |
| `/api/security-checks` | GET | List security checks |
| `/api/pipeline` | GET | Pipeline diagram |

### Example API Request

```bash
curl -X POST http://localhost:8000/api/review \
  -H "Content-Type: application/json" \
  -d '{
    "code": "password = \"secret123\"",
    "language": "python"
  }'
```

## ⚙️ Configuration

```python
from code_review import ReviewConfig

config = ReviewConfig(
    # Analysis toggles
    check_syntax=True,
    check_security=True,
    check_style=True,
    check_performance=True,
    check_complexity=True,
    check_documentation=True,
    check_dead_code=True,
    check_dependencies=True,
    
    # Thresholds
    max_line_length=120,
    max_function_length=50,
    max_complexity=15,
    
    # Severity filtering
    min_severity=Severity.INFO,
    
    # Languages
    languages=["python", "javascript", "typescript"],
    
    # Processing
    parallel=True,
    max_workers=4,
    
    # Output
    output_dir="output",
    verbose=True,
)
```

## 🔧 Supported Languages

| Language | Syntax | Security | Style | Performance |
|----------|--------|----------|-------|-------------|
| Python | ✅ | ✅ | ✅ | ✅ |
| JavaScript | ✅ | ✅ | ✅ | ⚡ |
| TypeScript | ✅ | ✅ | ✅ | ⚡ |
| Go | ⚡ | ⚡ | ⚡ | ⚡ |
| Java | ⚡ | ⚡ | ⚡ | ⚡ |

✅ Full support | ⚡ Partial support

## 🔌 CI/CD Integration

### GitHub Actions

```yaml
- name: Code Review
  run: |
    pip install -r requirements.txt
    python main.py review ./src --fail-on-critical
```

### GitLab CI

```yaml
code_review:
  script:
    - python main.py review ./src --save --format json
  artifacts:
    paths:
      - output/*.json
```

## 📈 Metrics

### Complexity Metrics

- **Cyclomatic Complexity**: Measures code paths
- **Cognitive Complexity**: Measures understandability
- **Maintainability Index**: Overall maintainability score

### Quality Score

The overall quality score (0-10) is calculated from:
- Complexity (lower is better)
- Test coverage (higher is better)
- Documentation coverage (higher is better)

## 🔮 Future Enhancements

- [ ] Git PR integration
- [ ] Automatic PR comments
- [ ] Historical trend tracking
- [ ] Custom rule definitions
- [ ] IDE plugins
- [ ] Slack/Teams notifications

## 📝 License

MIT License

---

*Secure your code before it reaches production!* 🔒🚀
