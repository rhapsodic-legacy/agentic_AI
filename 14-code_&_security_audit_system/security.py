"""
Code Review - Security Analyzer

OWASP-aligned security vulnerability detection including:
- Injection attacks (SQL, Command, XSS)
- Authentication issues
- Sensitive data exposure
- Hardcoded secrets
- Cryptographic issues
- Dependency vulnerabilities
"""

import re
from typing import Optional
from dataclasses import dataclass

from ..models import (
    Issue, SecurityFinding, CodeLocation, CodeSnippet,
    Severity, IssueCategory, SecurityCategory, FixDifficulty, Language
)


# =============================================================================
# Security Patterns
# =============================================================================

@dataclass
class SecurityPattern:
    """A security vulnerability pattern."""
    name: str
    description: str
    pattern: str  # Regex pattern
    severity: Severity
    category: SecurityCategory
    cwe_id: str
    fix_suggestion: str
    fix_template: str = ""
    languages: list[str] = None  # None = all languages
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ["python", "javascript", "typescript", "go", "java"]


# SQL Injection Patterns
SQL_INJECTION_PATTERNS = [
    SecurityPattern(
        name="SQL Injection via String Formatting",
        description="User input is directly concatenated into SQL query using string formatting",
        pattern=r'(f"[^"]*SELECT[^"]*\{[^}]+\}[^"]*"|f\'[^\']*SELECT[^\']*\{[^}]+\}[^\']*\')',
        severity=Severity.CRITICAL,
        category=SecurityCategory.INJECTION,
        cwe_id="CWE-89",
        fix_suggestion="Use parameterized queries instead of string formatting",
        fix_template='cursor.execute("SELECT * FROM table WHERE column = %s", (user_input,))',
        languages=["python"],
    ),
    SecurityPattern(
        name="SQL Injection via Concatenation",
        description="User input is concatenated into SQL query",
        pattern=r'["\']SELECT[^"\']*["\']\s*\+\s*\w+|["\']SELECT[^"\']*%s[^"\']*["\']',
        severity=Severity.CRITICAL,
        category=SecurityCategory.INJECTION,
        cwe_id="CWE-89",
        fix_suggestion="Use parameterized queries or an ORM",
        languages=["python", "javascript", "java"],
    ),
    SecurityPattern(
        name="SQL Injection via Template String",
        description="User input in SQL query via template literal",
        pattern=r'`SELECT[^`]*\$\{[^}]+\}[^`]*`',
        severity=Severity.CRITICAL,
        category=SecurityCategory.INJECTION,
        cwe_id="CWE-89",
        fix_suggestion="Use parameterized queries",
        languages=["javascript", "typescript"],
    ),
]

# Command Injection Patterns
COMMAND_INJECTION_PATTERNS = [
    SecurityPattern(
        name="Command Injection via os.system",
        description="User input passed to os.system() can lead to command injection",
        pattern=r'os\.system\s*\([^)]*[+%f][^)]*\)|os\.system\s*\(.*\{.*\}.*\)',
        severity=Severity.CRITICAL,
        category=SecurityCategory.INJECTION,
        cwe_id="CWE-78",
        fix_suggestion="Use subprocess with shell=False and pass arguments as list",
        fix_template='subprocess.run(["command", arg1, arg2], shell=False)',
        languages=["python"],
    ),
    SecurityPattern(
        name="Command Injection via subprocess shell",
        description="subprocess called with shell=True and user input",
        pattern=r'subprocess\.(run|call|Popen)\s*\([^)]*shell\s*=\s*True',
        severity=Severity.HIGH,
        category=SecurityCategory.INJECTION,
        cwe_id="CWE-78",
        fix_suggestion="Set shell=False and pass command as list",
        languages=["python"],
    ),
    SecurityPattern(
        name="Command Injection via exec",
        description="Using exec with potentially untrusted input",
        pattern=r'exec\s*\([^)]*[+%][^)]*\)|exec\s*\(.*\{.*\}.*\)',
        severity=Severity.CRITICAL,
        category=SecurityCategory.INJECTION,
        cwe_id="CWE-94",
        fix_suggestion="Avoid exec with user input; use safer alternatives",
        languages=["python", "javascript"],
    ),
    SecurityPattern(
        name="eval() Usage",
        description="eval() can execute arbitrary code",
        pattern=r'\beval\s*\(',
        severity=Severity.HIGH,
        category=SecurityCategory.INJECTION,
        cwe_id="CWE-95",
        fix_suggestion="Avoid eval(); use safer parsing methods like JSON.parse() or ast.literal_eval()",
        languages=["python", "javascript", "typescript"],
    ),
]

# XSS Patterns
XSS_PATTERNS = [
    SecurityPattern(
        name="Potential XSS via innerHTML",
        description="Setting innerHTML with potentially untrusted content",
        pattern=r'\.innerHTML\s*=',
        severity=Severity.HIGH,
        category=SecurityCategory.XSS,
        cwe_id="CWE-79",
        fix_suggestion="Use textContent or sanitize HTML before setting innerHTML",
        languages=["javascript", "typescript"],
    ),
    SecurityPattern(
        name="Potential XSS via document.write",
        description="document.write with untrusted content can lead to XSS",
        pattern=r'document\.write\s*\(',
        severity=Severity.MEDIUM,
        category=SecurityCategory.XSS,
        cwe_id="CWE-79",
        fix_suggestion="Use DOM manipulation methods instead of document.write",
        languages=["javascript", "typescript"],
    ),
    SecurityPattern(
        name="Potential XSS via dangerouslySetInnerHTML",
        description="React dangerouslySetInnerHTML without sanitization",
        pattern=r'dangerouslySetInnerHTML\s*=',
        severity=Severity.HIGH,
        category=SecurityCategory.XSS,
        cwe_id="CWE-79",
        fix_suggestion="Sanitize content with DOMPurify before using dangerouslySetInnerHTML",
        languages=["javascript", "typescript"],
    ),
]

# Authentication & Secrets Patterns
AUTH_PATTERNS = [
    SecurityPattern(
        name="Hardcoded Password",
        description="Password appears to be hardcoded in source code",
        pattern=r'(?i)(password|passwd|pwd)\s*=\s*["\'][^"\']{4,}["\']',
        severity=Severity.CRITICAL,
        category=SecurityCategory.HARDCODED_SECRETS,
        cwe_id="CWE-798",
        fix_suggestion="Use environment variables or a secrets manager",
        fix_template='password = os.environ.get("DB_PASSWORD")',
    ),
    SecurityPattern(
        name="Hardcoded API Key",
        description="API key appears to be hardcoded",
        pattern=r'(?i)(api[_-]?key|apikey|api[_-]?secret)\s*=\s*["\'][a-zA-Z0-9_\-]{16,}["\']',
        severity=Severity.CRITICAL,
        category=SecurityCategory.HARDCODED_SECRETS,
        cwe_id="CWE-798",
        fix_suggestion="Store API keys in environment variables or a vault",
    ),
    SecurityPattern(
        name="Hardcoded Secret Key",
        description="Secret key appears to be hardcoded",
        pattern=r'(?i)(secret[_-]?key|private[_-]?key)\s*=\s*["\'][^"\']{8,}["\']',
        severity=Severity.CRITICAL,
        category=SecurityCategory.HARDCODED_SECRETS,
        cwe_id="CWE-798",
        fix_suggestion="Use environment variables for secret keys",
    ),
    SecurityPattern(
        name="AWS Credentials",
        description="Potential AWS credentials in code",
        pattern=r'(?i)(AKIA[0-9A-Z]{16}|aws[_-]?secret[_-]?access[_-]?key)',
        severity=Severity.CRITICAL,
        category=SecurityCategory.HARDCODED_SECRETS,
        cwe_id="CWE-798",
        fix_suggestion="Use AWS credential provider chain or environment variables",
    ),
    SecurityPattern(
        name="Private Key",
        description="Private key embedded in code",
        pattern=r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----',
        severity=Severity.CRITICAL,
        category=SecurityCategory.HARDCODED_SECRETS,
        cwe_id="CWE-321",
        fix_suggestion="Store private keys in secure files with proper permissions",
    ),
    SecurityPattern(
        name="JWT Secret Hardcoded",
        description="JWT secret appears to be hardcoded",
        pattern=r'(?i)(jwt[_-]?secret|token[_-]?secret)\s*=\s*["\'][^"\']{8,}["\']',
        severity=Severity.HIGH,
        category=SecurityCategory.HARDCODED_SECRETS,
        cwe_id="CWE-798",
        fix_suggestion="Use environment variables for JWT secrets",
    ),
]

# Cryptographic Issues
CRYPTO_PATTERNS = [
    SecurityPattern(
        name="Weak Hash Algorithm (MD5)",
        description="MD5 is cryptographically broken and should not be used for security",
        pattern=r'(?i)(md5|hashlib\.md5)',
        severity=Severity.HIGH,
        category=SecurityCategory.CRYPTO,
        cwe_id="CWE-328",
        fix_suggestion="Use SHA-256 or stronger hash algorithms",
    ),
    SecurityPattern(
        name="Weak Hash Algorithm (SHA1)",
        description="SHA1 is considered weak for security purposes",
        pattern=r'(?i)(sha1|hashlib\.sha1)',
        severity=Severity.MEDIUM,
        category=SecurityCategory.CRYPTO,
        cwe_id="CWE-328",
        fix_suggestion="Use SHA-256 or stronger hash algorithms",
    ),
    SecurityPattern(
        name="Insecure Random",
        description="Using non-cryptographic random for security purposes",
        pattern=r'\brandom\.(random|randint|choice|randrange)\s*\(',
        severity=Severity.MEDIUM,
        category=SecurityCategory.CRYPTO,
        cwe_id="CWE-330",
        fix_suggestion="Use secrets module for cryptographic randomness",
        fix_template='import secrets\ntoken = secrets.token_hex(32)',
        languages=["python"],
    ),
    SecurityPattern(
        name="Math.random() for Security",
        description="Math.random() is not cryptographically secure",
        pattern=r'Math\.random\s*\(\)',
        severity=Severity.MEDIUM,
        category=SecurityCategory.CRYPTO,
        cwe_id="CWE-330",
        fix_suggestion="Use crypto.getRandomValues() or crypto.randomBytes()",
        languages=["javascript", "typescript"],
    ),
]

# Data Exposure Patterns
DATA_EXPOSURE_PATTERNS = [
    SecurityPattern(
        name="Sensitive Data in Logs",
        description="Potentially logging sensitive information",
        pattern=r'(?i)(log|print|console\.log|logger)\s*\([^)]*(?:password|secret|token|api[_-]?key|credit[_-]?card)',
        severity=Severity.HIGH,
        category=SecurityCategory.SENSITIVE_DATA,
        cwe_id="CWE-532",
        fix_suggestion="Redact sensitive data before logging",
    ),
    SecurityPattern(
        name="Debug Mode Enabled",
        description="Debug mode may expose sensitive information",
        pattern=r'(?i)(debug\s*=\s*True|DEBUG\s*=\s*True|\.debug\s*\(\s*True\s*\))',
        severity=Severity.MEDIUM,
        category=SecurityCategory.SECURITY_MISCONFIG,
        cwe_id="CWE-489",
        fix_suggestion="Disable debug mode in production",
        languages=["python"],
    ),
    SecurityPattern(
        name="Exception Details Exposed",
        description="Full exception details may expose internal information",
        pattern=r'except.*:\s*\n\s*(return|print|response).*str\s*\(\s*e\s*\)',
        severity=Severity.MEDIUM,
        category=SecurityCategory.SENSITIVE_DATA,
        cwe_id="CWE-209",
        fix_suggestion="Log detailed errors server-side, return generic messages to users",
    ),
]

# Access Control Patterns
ACCESS_CONTROL_PATTERNS = [
    SecurityPattern(
        name="Missing CSRF Protection",
        description="Form endpoint may be missing CSRF protection",
        pattern=r'@app\.(post|put|delete|patch)\s*\([^)]+\)\s*\n(?:.*\n){0,5}(?!.*csrf)',
        severity=Severity.MEDIUM,
        category=SecurityCategory.BROKEN_ACCESS,
        cwe_id="CWE-352",
        fix_suggestion="Add CSRF token validation to state-changing endpoints",
        languages=["python"],
    ),
    SecurityPattern(
        name="CORS Allow All",
        description="CORS configuration allows all origins",
        pattern=r'(?i)(Access-Control-Allow-Origin["\']?\s*:\s*["\']?\*|allow_origins\s*=\s*\[\s*["\']?\*["\']?\s*\]|cors\(\s*\*\s*\))',
        severity=Severity.MEDIUM,
        category=SecurityCategory.SECURITY_MISCONFIG,
        cwe_id="CWE-346",
        fix_suggestion="Restrict CORS to specific trusted origins",
    ),
]

# Deserialization Patterns
DESERIALIZATION_PATTERNS = [
    SecurityPattern(
        name="Unsafe Pickle Usage",
        description="pickle.loads on untrusted data can lead to code execution",
        pattern=r'pickle\.(loads?|Unpickler)',
        severity=Severity.HIGH,
        category=SecurityCategory.INSECURE_DESERIALIZATION,
        cwe_id="CWE-502",
        fix_suggestion="Avoid pickle for untrusted data; use JSON or other safe formats",
        languages=["python"],
    ),
    SecurityPattern(
        name="Unsafe YAML Loading",
        description="yaml.load without safe_load can execute arbitrary code",
        pattern=r'yaml\.load\s*\([^)]*\)(?!\s*,\s*Loader\s*=\s*yaml\.SafeLoader)',
        severity=Severity.HIGH,
        category=SecurityCategory.INSECURE_DESERIALIZATION,
        cwe_id="CWE-502",
        fix_suggestion="Use yaml.safe_load() instead of yaml.load()",
        languages=["python"],
    ),
]


# All patterns grouped
ALL_SECURITY_PATTERNS = (
    SQL_INJECTION_PATTERNS +
    COMMAND_INJECTION_PATTERNS +
    XSS_PATTERNS +
    AUTH_PATTERNS +
    CRYPTO_PATTERNS +
    DATA_EXPOSURE_PATTERNS +
    ACCESS_CONTROL_PATTERNS +
    DESERIALIZATION_PATTERNS
)


# =============================================================================
# Security Analyzer
# =============================================================================

class SecurityAnalyzer:
    """
    Analyzes code for security vulnerabilities.
    
    Implements OWASP Top 10 detection patterns.
    """
    
    def __init__(self):
        self.patterns = ALL_SECURITY_PATTERNS
        self.findings: list[Issue] = []
    
    def analyze(
        self,
        code: str,
        file_path: str,
        language: str = "python",
    ) -> list[Issue]:
        """Analyze code for security vulnerabilities."""
        self.findings = []
        
        lines = code.split('\n')
        
        for pattern in self.patterns:
            # Skip if not applicable to this language
            if language not in pattern.languages:
                continue
            
            try:
                regex = re.compile(pattern.pattern, re.MULTILINE | re.IGNORECASE)
                
                for match in regex.finditer(code):
                    # Find line number
                    line_num = code[:match.start()].count('\n') + 1
                    
                    # Get code snippet
                    snippet_start = max(0, line_num - 3)
                    snippet_end = min(len(lines), line_num + 3)
                    snippet_code = '\n'.join(lines[snippet_start:snippet_end])
                    
                    location = CodeLocation(
                        file_path=file_path,
                        line_start=line_num,
                        line_end=line_num,
                    )
                    
                    snippet = CodeSnippet(
                        code=lines[line_num - 1] if line_num <= len(lines) else "",
                        location=location,
                        context_before=lines[max(0, line_num-3):line_num-1],
                        context_after=lines[line_num:min(len(lines), line_num+2)],
                        highlighted_lines=[line_num],
                    )
                    
                    finding = SecurityFinding(
                        issue_id="",
                        title=pattern.name,
                        description=pattern.description,
                        category=IssueCategory.SECURITY,
                        severity=pattern.severity,
                        location=location,
                        snippet=snippet,
                        security_category=pattern.category,
                        cwe_id=pattern.cwe_id,
                        fix_suggestion=pattern.fix_suggestion,
                        fix_code=pattern.fix_template,
                        fix_difficulty=FixDifficulty.MODERATE,
                        analyzer="security_scanner",
                        references=[
                            f"https://cwe.mitre.org/data/definitions/{pattern.cwe_id.split('-')[1]}.html"
                        ] if pattern.cwe_id else [],
                    )
                    
                    self.findings.append(finding)
            
            except re.error:
                continue
        
        return self.findings
    
    def get_vulnerability_summary(self) -> dict:
        """Get summary of found vulnerabilities."""
        summary = {
            "total": len(self.findings),
            "by_severity": {},
            "by_category": {},
        }
        
        for finding in self.findings:
            # By severity
            sev = finding.severity.value
            summary["by_severity"][sev] = summary["by_severity"].get(sev, 0) + 1
            
            # By category
            if finding.security_category:
                cat = finding.security_category.value
                summary["by_category"][cat] = summary["by_category"].get(cat, 0) + 1
        
        return summary


# =============================================================================
# Dependency Scanner
# =============================================================================

# Known vulnerable packages (simplified database)
KNOWN_VULNERABILITIES = {
    "python": {
        "requests": [
            {"version": "<2.20.0", "cve": "CVE-2018-18074", "severity": "medium", "fixed": "2.20.0"},
        ],
        "django": [
            {"version": "<2.2.28", "cve": "CVE-2022-28346", "severity": "critical", "fixed": "2.2.28"},
            {"version": "<3.2.13", "cve": "CVE-2022-28347", "severity": "critical", "fixed": "3.2.13"},
        ],
        "flask": [
            {"version": "<2.2.5", "cve": "CVE-2023-30861", "severity": "high", "fixed": "2.2.5"},
        ],
        "pyyaml": [
            {"version": "<5.4", "cve": "CVE-2020-14343", "severity": "critical", "fixed": "5.4"},
        ],
        "pillow": [
            {"version": "<9.0.0", "cve": "CVE-2022-22817", "severity": "critical", "fixed": "9.0.0"},
        ],
        "urllib3": [
            {"version": "<1.26.5", "cve": "CVE-2021-33503", "severity": "high", "fixed": "1.26.5"},
        ],
        "cryptography": [
            {"version": "<39.0.1", "cve": "CVE-2023-23931", "severity": "high", "fixed": "39.0.1"},
        ],
    },
    "javascript": {
        "lodash": [
            {"version": "<4.17.21", "cve": "CVE-2021-23337", "severity": "high", "fixed": "4.17.21"},
        ],
        "axios": [
            {"version": "<0.21.1", "cve": "CVE-2020-28168", "severity": "medium", "fixed": "0.21.1"},
        ],
        "express": [
            {"version": "<4.17.3", "cve": "CVE-2022-24999", "severity": "high", "fixed": "4.17.3"},
        ],
        "jsonwebtoken": [
            {"version": "<9.0.0", "cve": "CVE-2022-23529", "severity": "critical", "fixed": "9.0.0"},
        ],
    },
}


class DependencyScanner:
    """Scans dependencies for known vulnerabilities."""
    
    def __init__(self):
        self.vulnerabilities = KNOWN_VULNERABILITIES
    
    def scan_requirements(self, content: str, language: str = "python") -> list[Issue]:
        """Scan a requirements file for vulnerable packages."""
        findings = []
        
        lang_vulns = self.vulnerabilities.get(language, {})
        
        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse package==version or package>=version
            match = re.match(r'^([a-zA-Z0-9_-]+)([=<>!]+)?(.+)?$', line)
            if not match:
                continue
            
            package = match.group(1).lower()
            version = match.group(3) or ""
            
            if package in lang_vulns:
                for vuln in lang_vulns[package]:
                    # Simplified version check
                    finding = Issue(
                        issue_id="",
                        title=f"Vulnerable dependency: {package}",
                        description=f"Package {package} has known vulnerability {vuln['cve']}",
                        category=IssueCategory.DEPENDENCY,
                        severity=Severity[vuln['severity'].upper()],
                        location=CodeLocation(
                            file_path="requirements.txt" if language == "python" else "package.json",
                            line_start=line_num,
                        ),
                        security_category=SecurityCategory.VULNERABLE_COMPONENTS,
                        cwe_id=vuln['cve'],
                        fix_suggestion=f"Upgrade {package} to version {vuln['fixed']} or later",
                        fix_difficulty=FixDifficulty.EASY,
                        auto_fixable=True,
                        analyzer="dependency_scanner",
                    )
                    findings.append(finding)
        
        return findings
