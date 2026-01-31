"""
Code Review - AutoGen Agents

Pipeline architecture with parallel analysis phase:
1. Parallel Analysis (Syntax, Security, Style, Performance)
2. Aggregator (Combine Findings)
3. Prioritizer (Rank by Severity)
4. Fix Suggester (Generate Patches)
5. Report Generator
"""

from typing import Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    
    # Mock classes
    class AssistantAgent:
        def __init__(self, **kwargs):
            self.name = kwargs.get('name', '')
            self.system_message = kwargs.get('system_message', '')
    
    class UserProxyAgent:
        def __init__(self, **kwargs):
            self.name = kwargs.get('name', '')

from ..models import (
    Issue, AnalysisResult, ReviewReport, FixPatch,
    CodeLocation, CodeSnippet, CodeQualityMetrics,
    Severity, IssueCategory, Language
)
from ..analyzers import (
    SecurityAnalyzer, DependencyScanner,
    SyntaxAnalyzer, StyleAnalyzer, PerformanceAnalyzer,
    ComplexityAnalyzer, DocumentationAnalyzer, DeadCodeAnalyzer
)


# =============================================================================
# Agent Configurations
# =============================================================================

SYNTAX_AGENT_CONFIG = {
    "name": "SyntaxChecker",
    "system_message": """You are a syntax analysis expert. Your role is to:
1. Detect syntax errors and invalid code constructs
2. Identify common coding mistakes
3. Check for proper language syntax usage
4. Report issues with clear descriptions and fix suggestions

Focus on correctness and validity of the code structure.""",
}

SECURITY_AGENT_CONFIG = {
    "name": "SecurityScanner",
    "system_message": """You are a security vulnerability expert aligned with OWASP Top 10. Your role is to:
1. Detect injection vulnerabilities (SQL, Command, XSS)
2. Identify authentication and authorization issues
3. Find hardcoded secrets and credentials
4. Check for insecure cryptographic practices
5. Scan for known vulnerability patterns

Always provide CWE IDs and severity ratings. Be thorough but avoid false positives.""",
}

STYLE_AGENT_CONFIG = {
    "name": "StyleChecker",
    "system_message": """You are a code style and best practices expert. Your role is to:
1. Check adherence to language style guides (PEP8, ESLint, etc.)
2. Identify naming convention violations
3. Check code organization and structure
4. Ensure consistent formatting
5. Verify import organization

Focus on maintainability and readability.""",
}

PERFORMANCE_AGENT_CONFIG = {
    "name": "PerformanceAnalyzer",
    "system_message": """You are a performance optimization expert. Your role is to:
1. Identify performance anti-patterns
2. Detect inefficient algorithms (O(n²) when O(n) possible)
3. Find memory leaks and resource issues
4. Check for unnecessary operations in loops
5. Identify opportunities for caching

Focus on algorithmic complexity and resource usage.""",
}

AGGREGATOR_AGENT_CONFIG = {
    "name": "Aggregator",
    "system_message": """You are a findings aggregator. Your role is to:
1. Collect findings from all analyzer agents
2. Deduplicate similar issues
3. Merge related findings
4. Ensure consistent formatting
5. Validate all findings have required fields

Create a comprehensive list of unique issues.""",
}

PRIORITIZER_AGENT_CONFIG = {
    "name": "Prioritizer",
    "system_message": """You are a risk assessment expert. Your role is to:
1. Rank issues by severity and impact
2. Consider exploitability for security issues
3. Factor in blast radius and user impact
4. Group related issues
5. Identify quick wins vs major refactors

Provide a prioritized action plan.""",
}

FIX_SUGGESTER_AGENT_CONFIG = {
    "name": "FixSuggester",
    "system_message": """You are a code repair expert. Your role is to:
1. Generate specific fix suggestions for each issue
2. Provide corrected code snippets
3. Ensure fixes don't introduce new issues
4. Consider backward compatibility
5. Generate unified diffs where possible

Make fixes practical and immediately applicable.""",
}

REPORT_GENERATOR_AGENT_CONFIG = {
    "name": "ReportGenerator",
    "system_message": """You are a technical writing expert. Your role is to:
1. Create clear, actionable review reports
2. Write executive summaries
3. Format findings in markdown
4. Include code examples and fixes
5. Generate metrics visualizations

Make reports useful for developers and managers.""",
}


# =============================================================================
# Pipeline Agents
# =============================================================================

@dataclass
class AnalyzerAgent:
    """Base class for analyzer agents."""
    name: str
    analyzer: object
    
    def analyze(self, code: str, file_path: str, language: str = "python") -> list[Issue]:
        """Run analysis."""
        return self.analyzer.analyze(code, file_path, language)


class SyntaxCheckerAgent(AnalyzerAgent):
    """Agent for syntax checking."""
    
    def __init__(self):
        super().__init__(
            name="SyntaxChecker",
            analyzer=SyntaxAnalyzer(),
        )


class SecurityScannerAgent(AnalyzerAgent):
    """Agent for security scanning."""
    
    def __init__(self):
        super().__init__(
            name="SecurityScanner",
            analyzer=SecurityAnalyzer(),
        )


class StyleCheckerAgent(AnalyzerAgent):
    """Agent for style checking."""
    
    def __init__(self):
        super().__init__(
            name="StyleChecker",
            analyzer=StyleAnalyzer(),
        )


class PerformanceAnalyzerAgent(AnalyzerAgent):
    """Agent for performance analysis."""
    
    def __init__(self):
        super().__init__(
            name="PerformanceAnalyzer",
            analyzer=PerformanceAnalyzer(),
        )


class ComplexityAnalyzerAgent:
    """Agent for complexity analysis."""
    
    def __init__(self):
        self.name = "ComplexityAnalyzer"
        self.analyzer = ComplexityAnalyzer()
    
    def analyze(self, code: str, file_path: str, language: str = "python"):
        """Run complexity analysis."""
        return self.analyzer.analyze(code, file_path, language)


class DocumentationAnalyzerAgent:
    """Agent for documentation analysis."""
    
    def __init__(self):
        self.name = "DocumentationAnalyzer"
        self.analyzer = DocumentationAnalyzer()
    
    def analyze(self, code: str, file_path: str, language: str = "python"):
        """Run documentation analysis."""
        return self.analyzer.analyze(code, file_path, language)


class DeadCodeAnalyzerAgent(AnalyzerAgent):
    """Agent for dead code detection."""
    
    def __init__(self):
        super().__init__(
            name="DeadCodeAnalyzer",
            analyzer=DeadCodeAnalyzer(),
        )


# =============================================================================
# Pipeline Coordinator
# =============================================================================

class AggregatorAgent:
    """Aggregates findings from all analyzers."""
    
    def __init__(self):
        self.name = "Aggregator"
    
    def aggregate(self, all_findings: list[list[Issue]]) -> list[Issue]:
        """Combine and deduplicate findings."""
        combined = []
        seen_hashes = set()
        
        for findings in all_findings:
            for finding in findings:
                # Create hash for deduplication
                finding_hash = f"{finding.title}:{finding.location}"
                
                if finding_hash not in seen_hashes:
                    seen_hashes.add(finding_hash)
                    combined.append(finding)
        
        return combined


class PrioritizerAgent:
    """Prioritizes findings by severity and impact."""
    
    def __init__(self):
        self.name = "Prioritizer"
    
    def prioritize(self, findings: list[Issue]) -> list[Issue]:
        """Sort findings by priority."""
        
        # Priority scoring
        severity_scores = {
            Severity.CRITICAL: 100,
            Severity.HIGH: 75,
            Severity.MEDIUM: 50,
            Severity.LOW: 25,
            Severity.INFO: 10,
        }
        
        category_multipliers = {
            IssueCategory.SECURITY: 2.0,
            IssueCategory.SYNTAX: 1.5,
            IssueCategory.PERFORMANCE: 1.2,
            IssueCategory.MAINTAINABILITY: 1.0,
            IssueCategory.STYLE: 0.8,
            IssueCategory.DOCUMENTATION: 0.6,
        }
        
        def calculate_priority(issue: Issue) -> float:
            base_score = severity_scores.get(issue.severity, 25)
            multiplier = category_multipliers.get(issue.category, 1.0)
            
            # Bonus for auto-fixable
            if issue.auto_fixable:
                base_score *= 1.1
            
            return base_score * multiplier
        
        # Sort by priority (highest first)
        sorted_findings = sorted(
            findings,
            key=calculate_priority,
            reverse=True
        )
        
        return sorted_findings


class FixSuggesterAgent:
    """Generates fix patches for issues."""
    
    def __init__(self):
        self.name = "FixSuggester"
    
    def generate_fixes(self, findings: list[Issue], code: str) -> list[FixPatch]:
        """Generate fix patches for issues."""
        patches = []
        
        for finding in findings:
            if finding.fix_code or finding.fix_suggestion:
                patch = FixPatch(
                    patch_id=f"FIX-{finding.issue_id}",
                    issue_id=finding.issue_id,
                    file_path=finding.location.file_path,
                    line_start=finding.location.line_start,
                    line_end=finding.location.line_end,
                    original_code=finding.snippet.code if finding.snippet else "",
                    fixed_code=finding.fix_code,
                    description=finding.fix_suggestion,
                    confidence=finding.confidence,
                )
                
                if patch.original_code and patch.fixed_code:
                    patch.generate_diff()
                
                patches.append(patch)
        
        return patches


class ReportGeneratorAgent:
    """Generates code review reports."""
    
    def __init__(self):
        self.name = "ReportGenerator"
    
    def generate_report(
        self,
        analysis: AnalysisResult,
        fixes: list[FixPatch] = None,
    ) -> ReviewReport:
        """Generate complete review report."""
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(analysis)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(analysis)
        
        report = ReviewReport(
            report_id=f"REP-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            generated_at=datetime.now().isoformat(),
            analysis=analysis,
            suggested_fixes=fixes or [],
            recommendations=recommendations,
            executive_summary=executive_summary,
        )
        
        return report
    
    def _generate_executive_summary(self, analysis: AnalysisResult) -> str:
        """Generate executive summary."""
        
        # Determine overall status
        if analysis.critical_count > 0:
            status = "🔴 **Critical Issues Found** - Immediate action required"
        elif analysis.high_count > 0:
            status = "🟠 **High Severity Issues** - Action needed"
        elif analysis.medium_count > 0:
            status = "🟡 **Moderate Issues** - Review recommended"
        else:
            status = "🟢 **Code Looks Good** - Minor improvements possible"
        
        security_issues = len(analysis.security_issues)
        security_note = f"**{security_issues} security vulnerabilities** require attention." if security_issues > 0 else ""
        
        summary = f"""{status}

The analysis reviewed **{analysis.files_analyzed} files** with **{analysis.total_lines:,} lines of code**.

Found **{analysis.issue_count} issues**: {analysis.critical_count} critical, {analysis.high_count} high, {analysis.medium_count} medium, {analysis.low_count} low severity.

{security_note}

Quality Score: **{analysis.quality_metrics.overall_score:.1f}/10**
"""
        return summary
    
    def _generate_recommendations(self, analysis: AnalysisResult) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Security
        if analysis.security_issues:
            recommendations.append("Address all security vulnerabilities before deployment")
        
        # By category
        categories = {}
        for issue in analysis.issues:
            cat = issue.category.value
            categories[cat] = categories.get(cat, 0) + 1
        
        if categories.get('security', 0) > 0:
            recommendations.append(f"Fix {categories['security']} security issues - consider security training")
        
        if categories.get('performance', 0) > 3:
            recommendations.append("Performance review recommended - multiple anti-patterns detected")
        
        if categories.get('documentation', 0) > 5:
            recommendations.append("Improve documentation coverage - add docstrings to public APIs")
        
        if categories.get('style', 0) > 10:
            recommendations.append("Set up linter (pylint/flake8) to catch style issues automatically")
        
        # Complexity
        if analysis.quality_metrics.complexity.cyclomatic_complexity > 15:
            recommendations.append("Consider refactoring complex functions into smaller units")
        
        # Test coverage (if available)
        if analysis.quality_metrics.testing.line_coverage < 60:
            recommendations.append("Increase test coverage to at least 60%")
        
        # Auto-fixable
        auto_fixable = sum(1 for i in analysis.issues if i.auto_fixable)
        if auto_fixable > 0:
            recommendations.append(f"{auto_fixable} issues can be auto-fixed - consider running automated fixes")
        
        return recommendations[:10]  # Top 10 recommendations


# =============================================================================
# AutoGen Pipeline (when AutoGen is available)
# =============================================================================

def create_autogen_agents(llm_config: dict = None):
    """Create AutoGen agents for the pipeline."""
    
    if not AUTOGEN_AVAILABLE:
        return None
    
    if llm_config is None:
        llm_config = {
            "config_list": [{"model": "gpt-4", "api_key": "your-key"}],
            "temperature": 0,
        }
    
    agents = {
        "syntax": AssistantAgent(
            name=SYNTAX_AGENT_CONFIG["name"],
            system_message=SYNTAX_AGENT_CONFIG["system_message"],
            llm_config=llm_config,
        ),
        "security": AssistantAgent(
            name=SECURITY_AGENT_CONFIG["name"],
            system_message=SECURITY_AGENT_CONFIG["system_message"],
            llm_config=llm_config,
        ),
        "style": AssistantAgent(
            name=STYLE_AGENT_CONFIG["name"],
            system_message=STYLE_AGENT_CONFIG["system_message"],
            llm_config=llm_config,
        ),
        "performance": AssistantAgent(
            name=PERFORMANCE_AGENT_CONFIG["name"],
            system_message=PERFORMANCE_AGENT_CONFIG["system_message"],
            llm_config=llm_config,
        ),
        "aggregator": AssistantAgent(
            name=AGGREGATOR_AGENT_CONFIG["name"],
            system_message=AGGREGATOR_AGENT_CONFIG["system_message"],
            llm_config=llm_config,
        ),
        "prioritizer": AssistantAgent(
            name=PRIORITIZER_AGENT_CONFIG["name"],
            system_message=PRIORITIZER_AGENT_CONFIG["system_message"],
            llm_config=llm_config,
        ),
        "fix_suggester": AssistantAgent(
            name=FIX_SUGGESTER_AGENT_CONFIG["name"],
            system_message=FIX_SUGGESTER_AGENT_CONFIG["system_message"],
            llm_config=llm_config,
        ),
        "report_generator": AssistantAgent(
            name=REPORT_GENERATOR_AGENT_CONFIG["name"],
            system_message=REPORT_GENERATOR_AGENT_CONFIG["system_message"],
            llm_config=llm_config,
        ),
    }
    
    return agents


def get_agent_pipeline():
    """Get the pipeline architecture diagram."""
    return """
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
│        │              │              │              │       │
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
"""
