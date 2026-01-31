"""
Code Review & Security Audit System - Data Models

Models for:
- Code analysis findings
- Security vulnerabilities
- Code quality metrics
- Fix suggestions
- Reports
"""

from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json


# =============================================================================
# Enums
# =============================================================================

class Severity(Enum):
    """Issue severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IssueCategory(Enum):
    """Categories of issues."""
    SECURITY = "security"
    SYNTAX = "syntax"
    STYLE = "style"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    DEPENDENCY = "dependency"


class Language(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    JAVA = "java"
    RUST = "rust"
    CPP = "cpp"
    CSHARP = "csharp"
    RUBY = "ruby"
    PHP = "php"
    UNKNOWN = "unknown"


class SecurityCategory(Enum):
    """OWASP-aligned security categories."""
    INJECTION = "injection"
    BROKEN_AUTH = "broken_authentication"
    SENSITIVE_DATA = "sensitive_data_exposure"
    XXE = "xml_external_entities"
    BROKEN_ACCESS = "broken_access_control"
    SECURITY_MISCONFIG = "security_misconfiguration"
    XSS = "cross_site_scripting"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    VULNERABLE_COMPONENTS = "vulnerable_components"
    INSUFFICIENT_LOGGING = "insufficient_logging"
    HARDCODED_SECRETS = "hardcoded_secrets"
    CRYPTO = "cryptographic_issues"


class FixDifficulty(Enum):
    """Difficulty of applying a fix."""
    TRIVIAL = "trivial"      # Auto-fixable
    EASY = "easy"            # Simple change
    MODERATE = "moderate"    # Some refactoring
    HARD = "hard"            # Significant changes
    COMPLEX = "complex"      # Architectural changes


# =============================================================================
# Code Location
# =============================================================================

@dataclass
class CodeLocation:
    """Location of code in a file."""
    file_path: str
    line_start: int
    line_end: int = 0
    column_start: int = 0
    column_end: int = 0
    
    def __post_init__(self):
        if self.line_end == 0:
            self.line_end = self.line_start
    
    def __str__(self) -> str:
        if self.line_start == self.line_end:
            return f"{self.file_path}:{self.line_start}"
        return f"{self.file_path}:{self.line_start}-{self.line_end}"


@dataclass
class CodeSnippet:
    """A snippet of code with context."""
    code: str
    location: CodeLocation
    context_before: list[str] = field(default_factory=list)
    context_after: list[str] = field(default_factory=list)
    highlighted_lines: list[int] = field(default_factory=list)
    
    def to_markdown(self, language: str = "python") -> str:
        """Format as markdown code block."""
        lines = []
        
        # Add context before
        for line in self.context_before[-3:]:
            lines.append(f"  {line}")
        
        # Add main code with marker
        for i, line in enumerate(self.code.split('\n')):
            line_num = self.location.line_start + i
            marker = "→ " if line_num in self.highlighted_lines or not self.highlighted_lines else "  "
            lines.append(f"{marker}{line}")
        
        # Add context after
        for line in self.context_after[:3]:
            lines.append(f"  {line}")
        
        return f"```{language}\n" + '\n'.join(lines) + "\n```"


# =============================================================================
# Issues and Findings
# =============================================================================

@dataclass
class Issue:
    """A code review issue/finding."""
    issue_id: str
    title: str
    description: str
    
    # Classification
    category: IssueCategory
    severity: Severity
    
    # Location
    location: CodeLocation
    snippet: Optional[CodeSnippet] = None
    
    # Security-specific
    security_category: Optional[SecurityCategory] = None
    cwe_id: Optional[str] = None  # e.g., "CWE-89"
    cvss_score: Optional[float] = None
    
    # Impact
    impact: str = ""
    affected_users: str = "all"  # "all", "authenticated", "admin"
    
    # Fix
    fix_suggestion: str = ""
    fix_code: str = ""
    fix_difficulty: FixDifficulty = FixDifficulty.MODERATE
    auto_fixable: bool = False
    
    # References
    references: list[str] = field(default_factory=list)
    related_issues: list[str] = field(default_factory=list)
    
    # Metadata
    analyzer: str = ""  # Which analyzer found this
    confidence: float = 1.0  # 0-1 confidence score
    false_positive: bool = False
    
    # Status
    acknowledged: bool = False
    fixed: bool = False
    
    def __post_init__(self):
        if not self.issue_id:
            # Generate ID from content hash
            content = f"{self.title}{self.location}{self.category.value}"
            self.issue_id = hashlib.md5(content.encode()).hexdigest()[:12]
    
    @property
    def severity_emoji(self) -> str:
        return {
            Severity.CRITICAL: "🔴",
            Severity.HIGH: "🟠",
            Severity.MEDIUM: "🟡",
            Severity.LOW: "🔵",
            Severity.INFO: "⚪",
        }.get(self.severity, "⚪")
    
    @property
    def category_emoji(self) -> str:
        return {
            IssueCategory.SECURITY: "🔒",
            IssueCategory.SYNTAX: "📝",
            IssueCategory.STYLE: "🎨",
            IssueCategory.PERFORMANCE: "⚡",
            IssueCategory.MAINTAINABILITY: "🔧",
            IssueCategory.DOCUMENTATION: "📚",
            IssueCategory.TESTING: "🧪",
            IssueCategory.DEPENDENCY: "📦",
        }.get(self.category, "❓")
    
    def to_dict(self) -> dict:
        return {
            "issue_id": self.issue_id,
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "severity": self.severity.value,
            "location": str(self.location),
            "cwe_id": self.cwe_id,
            "fix_difficulty": self.fix_difficulty.value,
            "auto_fixable": self.auto_fixable,
        }
    
    def to_markdown(self) -> str:
        """Format issue as markdown."""
        md = f"""### {self.severity_emoji} {self.title}
**Severity:** {self.severity.value.upper()} | **Category:** {self.category.value}
**Location:** `{self.location}`
"""
        if self.cwe_id:
            md += f"**CWE:** {self.cwe_id}\n"
        
        md += f"\n{self.description}\n"
        
        if self.snippet:
            md += f"\n**Vulnerable Code:**\n{self.snippet.to_markdown()}\n"
        
        if self.fix_code:
            md += f"\n**Suggested Fix:**\n```\n{self.fix_code}\n```\n"
        
        if self.impact:
            md += f"\n**Impact:** {self.impact}\n"
        
        if self.references:
            md += "\n**References:**\n"
            for ref in self.references:
                md += f"- {ref}\n"
        
        return md


# =============================================================================
# Security Findings
# =============================================================================

@dataclass
class SecurityFinding(Issue):
    """A security-specific finding with additional context."""
    
    # Attack vectors
    attack_vector: str = ""
    exploit_complexity: str = "low"  # low, high
    privileges_required: str = "none"  # none, low, high
    user_interaction: str = "none"  # none, required
    
    # Compliance
    compliance_violations: list[str] = field(default_factory=list)  # GDPR, PCI-DSS, etc.
    
    # Remediation
    remediation_steps: list[str] = field(default_factory=list)
    workaround: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        self.category = IssueCategory.SECURITY


@dataclass
class DependencyVulnerability:
    """A vulnerability in a dependency."""
    package_name: str
    current_version: str
    vulnerable_versions: str
    fixed_version: str = ""
    
    cve_id: str = ""
    severity: Severity = Severity.MEDIUM
    
    description: str = ""
    exploit_available: bool = False
    
    def to_issue(self) -> Issue:
        return Issue(
            issue_id=f"DEP-{self.package_name}-{self.cve_id}",
            title=f"Vulnerable dependency: {self.package_name}",
            description=f"{self.description}\n\nVulnerable versions: {self.vulnerable_versions}",
            category=IssueCategory.DEPENDENCY,
            severity=self.severity,
            location=CodeLocation(file_path="requirements.txt", line_start=0),
            security_category=SecurityCategory.VULNERABLE_COMPONENTS,
            cwe_id=self.cve_id,
            fix_suggestion=f"Upgrade {self.package_name} to {self.fixed_version}" if self.fixed_version else "Check for updates",
            auto_fixable=bool(self.fixed_version),
        )


# =============================================================================
# Code Metrics
# =============================================================================

@dataclass
class ComplexityMetrics:
    """Code complexity metrics."""
    cyclomatic_complexity: float = 0.0
    cognitive_complexity: float = 0.0
    halstead_difficulty: float = 0.0
    maintainability_index: float = 0.0
    lines_of_code: int = 0
    comment_ratio: float = 0.0
    
    @property
    def complexity_rating(self) -> str:
        """Get complexity rating."""
        if self.cyclomatic_complexity <= 5:
            return "Simple"
        elif self.cyclomatic_complexity <= 10:
            return "Moderate"
        elif self.cyclomatic_complexity <= 20:
            return "Complex"
        else:
            return "Very Complex"
    
    @property
    def complexity_score(self) -> float:
        """Normalized complexity score (0-10, higher is better)."""
        # Invert so higher is better
        if self.cyclomatic_complexity <= 5:
            return 10.0
        elif self.cyclomatic_complexity <= 10:
            return 8.0
        elif self.cyclomatic_complexity <= 20:
            return 6.0
        elif self.cyclomatic_complexity <= 50:
            return 4.0
        else:
            return 2.0


@dataclass
class TestMetrics:
    """Test coverage metrics."""
    line_coverage: float = 0.0
    branch_coverage: float = 0.0
    function_coverage: float = 0.0
    
    total_tests: int = 0
    passing_tests: int = 0
    failing_tests: int = 0
    
    uncovered_lines: list[int] = field(default_factory=list)
    uncovered_functions: list[str] = field(default_factory=list)
    
    @property
    def coverage_rating(self) -> str:
        if self.line_coverage >= 80:
            return "Good"
        elif self.line_coverage >= 60:
            return "Acceptable"
        elif self.line_coverage >= 40:
            return "Low"
        else:
            return "Critical"


@dataclass
class DocumentationMetrics:
    """Documentation metrics."""
    docstring_coverage: float = 0.0
    public_api_documented: float = 0.0
    readme_exists: bool = False
    changelog_exists: bool = False
    
    missing_docstrings: list[str] = field(default_factory=list)


@dataclass
class CodeQualityMetrics:
    """Combined code quality metrics."""
    complexity: ComplexityMetrics = field(default_factory=ComplexityMetrics)
    testing: TestMetrics = field(default_factory=TestMetrics)
    documentation: DocumentationMetrics = field(default_factory=DocumentationMetrics)
    
    # Aggregate scores (0-10)
    overall_score: float = 0.0
    
    def calculate_overall(self):
        """Calculate overall quality score."""
        scores = [
            self.complexity.complexity_score,
            min(10, self.testing.line_coverage / 10),
            min(10, self.documentation.docstring_coverage / 10),
        ]
        self.overall_score = sum(scores) / len(scores)


# =============================================================================
# Analysis Results
# =============================================================================

@dataclass
class FileAnalysis:
    """Analysis results for a single file."""
    file_path: str
    language: Language
    
    # Metrics
    lines_of_code: int = 0
    lines_of_comments: int = 0
    blank_lines: int = 0
    
    complexity: ComplexityMetrics = field(default_factory=ComplexityMetrics)
    
    # Issues found
    issues: list[Issue] = field(default_factory=list)
    
    @property
    def issue_count(self) -> int:
        return len(self.issues)
    
    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.CRITICAL)
    
    @property
    def high_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.HIGH)


@dataclass
class AnalysisResult:
    """Complete analysis result for a codebase."""
    result_id: str
    timestamp: str
    
    # Scope
    target_path: str
    files_analyzed: int = 0
    total_lines: int = 0
    
    # Languages
    languages: dict[str, int] = field(default_factory=dict)  # {language: file_count}
    
    # Issues
    issues: list[Issue] = field(default_factory=list)
    
    # Metrics
    quality_metrics: CodeQualityMetrics = field(default_factory=CodeQualityMetrics)
    
    # File-level results
    file_results: list[FileAnalysis] = field(default_factory=list)
    
    # Analysis duration
    duration_seconds: float = 0.0
    
    @property
    def issue_count(self) -> int:
        return len(self.issues)
    
    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.CRITICAL)
    
    @property
    def high_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.HIGH)
    
    @property
    def medium_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.MEDIUM)
    
    @property
    def low_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.LOW)
    
    @property
    def security_issues(self) -> list[Issue]:
        return [i for i in self.issues if i.category == IssueCategory.SECURITY]
    
    def get_issues_by_severity(self, severity: Severity) -> list[Issue]:
        return [i for i in self.issues if i.severity == severity]
    
    def get_issues_by_category(self, category: IssueCategory) -> list[Issue]:
        return [i for i in self.issues if i.category == category]
    
    def to_dict(self) -> dict:
        return {
            "result_id": self.result_id,
            "timestamp": self.timestamp,
            "files_analyzed": self.files_analyzed,
            "total_lines": self.total_lines,
            "issues": {
                "total": self.issue_count,
                "critical": self.critical_count,
                "high": self.high_count,
                "medium": self.medium_count,
                "low": self.low_count,
            },
            "quality_score": self.quality_metrics.overall_score,
        }


# =============================================================================
# Fix Suggestions
# =============================================================================

@dataclass
class FixPatch:
    """A suggested code fix/patch."""
    patch_id: str
    issue_id: str
    
    # Location
    file_path: str
    line_start: int
    line_end: int
    
    # Content
    original_code: str
    fixed_code: str
    
    # Metadata
    description: str = ""
    confidence: float = 1.0
    side_effects: list[str] = field(default_factory=list)
    
    # Diff
    diff: str = ""
    
    def generate_diff(self):
        """Generate unified diff."""
        import difflib
        
        original_lines = self.original_code.splitlines(keepends=True)
        fixed_lines = self.fixed_code.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines,
            fixed_lines,
            fromfile=f"a/{self.file_path}",
            tofile=f"b/{self.file_path}",
            lineterm='',
        )
        
        self.diff = ''.join(diff)
        return self.diff


# =============================================================================
# Report
# =============================================================================

@dataclass
class ReviewReport:
    """Complete code review report."""
    report_id: str
    generated_at: str
    
    # Analysis result
    analysis: AnalysisResult
    
    # Fixes
    suggested_fixes: list[FixPatch] = field(default_factory=list)
    
    # Recommendations
    recommendations: list[str] = field(default_factory=list)
    
    # Summary
    executive_summary: str = ""
    
    def to_markdown(self) -> str:
        """Generate full markdown report."""
        md = f"""# Code Review Report

**Generated:** {self.generated_at}
**Report ID:** {self.report_id}

## Executive Summary

{self.executive_summary}

## Summary Statistics

| Metric | Value |
|--------|-------|
| Files Analyzed | {self.analysis.files_analyzed} |
| Total Lines | {self.analysis.total_lines:,} |
| Total Issues | {self.analysis.issue_count} |
| Critical | {self.analysis.critical_count} |
| High | {self.analysis.high_count} |
| Medium | {self.analysis.medium_count} |
| Low | {self.analysis.low_count} |

## Quality Score

**Overall Score:** {self.analysis.quality_metrics.overall_score:.1f}/10

- **Complexity:** {self.analysis.quality_metrics.complexity.complexity_rating}
- **Test Coverage:** {self.analysis.quality_metrics.testing.line_coverage:.1f}%
- **Documentation:** {self.analysis.quality_metrics.documentation.docstring_coverage:.1f}%

"""
        # Critical Issues
        critical = self.analysis.get_issues_by_severity(Severity.CRITICAL)
        if critical:
            md += "## 🔴 Critical Issues\n\n"
            for issue in critical:
                md += issue.to_markdown() + "\n---\n\n"
        
        # High Issues
        high = self.analysis.get_issues_by_severity(Severity.HIGH)
        if high:
            md += "## 🟠 High Severity Issues\n\n"
            for issue in high[:5]:  # Top 5
                md += issue.to_markdown() + "\n---\n\n"
        
        # Security Issues
        security = self.analysis.security_issues
        if security:
            md += f"## 🔒 Security Issues ({len(security)} total)\n\n"
            md += "| Severity | Issue | Location | CWE |\n"
            md += "|----------|-------|----------|-----|\n"
            for issue in security:
                md += f"| {issue.severity.value} | {issue.title[:40]} | `{issue.location}` | {issue.cwe_id or 'N/A'} |\n"
            md += "\n"
        
        # Recommendations
        if self.recommendations:
            md += "## 💡 Recommendations\n\n"
            for i, rec in enumerate(self.recommendations, 1):
                md += f"{i}. {rec}\n"
        
        return md
    
    def to_json(self) -> str:
        """Export as JSON."""
        return json.dumps({
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "analysis": self.analysis.to_dict(),
            "recommendations": self.recommendations,
        }, indent=2)
