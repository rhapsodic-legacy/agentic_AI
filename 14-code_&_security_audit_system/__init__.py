"""
Code Review & Security Audit System - Main Engine

Pipeline-based code review with parallel analysis phase.
"""

from typing import Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import os
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import (
    Issue, AnalysisResult, ReviewReport, FileAnalysis, FixPatch,
    CodeLocation, CodeSnippet, CodeQualityMetrics,
    ComplexityMetrics, TestMetrics, DocumentationMetrics,
    Severity, IssueCategory, Language
)
from .analyzers import (
    SecurityAnalyzer, DependencyScanner,
    SyntaxAnalyzer, StyleAnalyzer, PerformanceAnalyzer,
    ComplexityAnalyzer, DocumentationAnalyzer, DeadCodeAnalyzer
)
from .agents import (
    SyntaxCheckerAgent, SecurityScannerAgent, StyleCheckerAgent,
    PerformanceAnalyzerAgent, ComplexityAnalyzerAgent,
    DocumentationAnalyzerAgent, DeadCodeAnalyzerAgent,
    AggregatorAgent, PrioritizerAgent, FixSuggesterAgent,
    ReportGeneratorAgent, get_agent_pipeline
)


# =============================================================================
# Language Detection
# =============================================================================

LANGUAGE_EXTENSIONS = {
    ".py": Language.PYTHON,
    ".js": Language.JAVASCRIPT,
    ".jsx": Language.JAVASCRIPT,
    ".ts": Language.TYPESCRIPT,
    ".tsx": Language.TYPESCRIPT,
    ".go": Language.GO,
    ".java": Language.JAVA,
    ".rs": Language.RUST,
    ".cpp": Language.CPP,
    ".cc": Language.CPP,
    ".c": Language.CPP,
    ".cs": Language.CSHARP,
    ".rb": Language.RUBY,
    ".php": Language.PHP,
}


def detect_language(file_path: str) -> Language:
    """Detect programming language from file extension."""
    ext = Path(file_path).suffix.lower()
    return LANGUAGE_EXTENSIONS.get(ext, Language.UNKNOWN)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ReviewConfig:
    """Configuration for code review."""
    
    # Analysis toggles
    check_syntax: bool = True
    check_security: bool = True
    check_style: bool = True
    check_performance: bool = True
    check_complexity: bool = True
    check_documentation: bool = True
    check_dead_code: bool = True
    check_dependencies: bool = True
    
    # Thresholds
    max_line_length: int = 120
    max_function_length: int = 50
    max_complexity: int = 15
    min_doc_coverage: float = 50.0
    
    # Severity filtering
    min_severity: Severity = Severity.INFO
    
    # Languages to analyze
    languages: list[str] = field(default_factory=lambda: ["python", "javascript", "typescript"])
    
    # Parallel processing
    parallel: bool = True
    max_workers: int = 4
    
    # Output
    output_dir: str = "output"
    verbose: bool = True


# =============================================================================
# Code Review Engine
# =============================================================================

class CodeReviewEngine:
    """
    Main code review pipeline engine.
    
    Pipeline architecture:
    1. Parallel Analysis (Syntax, Security, Style, Performance)
    2. Aggregator (Combine Findings)
    3. Prioritizer (Rank by Severity)
    4. Fix Suggester (Generate Patches)
    5. Report Generator
    """
    
    def __init__(self, config: ReviewConfig = None):
        self.config = config or ReviewConfig()
        
        # Initialize analyzer agents
        self.syntax_agent = SyntaxCheckerAgent()
        self.security_agent = SecurityScannerAgent()
        self.style_agent = StyleCheckerAgent()
        self.performance_agent = PerformanceAnalyzerAgent()
        self.complexity_agent = ComplexityAnalyzerAgent()
        self.documentation_agent = DocumentationAnalyzerAgent()
        self.dead_code_agent = DeadCodeAnalyzerAgent()
        self.dependency_scanner = DependencyScanner()
        
        # Initialize pipeline agents
        self.aggregator = AggregatorAgent()
        self.prioritizer = PrioritizerAgent()
        self.fix_suggester = FixSuggesterAgent()
        self.report_generator = ReportGeneratorAgent()
        
        # Storage
        self.current_analysis: Optional[AnalysisResult] = None
        self.current_report: Optional[ReviewReport] = None
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message."""
        if self.config.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")
    
    # =========================================================================
    # File Processing
    # =========================================================================
    
    def read_file(self, file_path: str) -> Optional[str]:
        """Read a file's contents."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            self.log(f"Error reading {file_path}: {e}", "ERROR")
            return None
    
    def find_files(self, path: str) -> list[str]:
        """Find all analyzable files in a directory."""
        files = []
        path = Path(path)
        
        if path.is_file():
            return [str(path)]
        
        # Walk directory
        for ext, lang in LANGUAGE_EXTENSIONS.items():
            if lang.value in self.config.languages or Language.UNKNOWN in self.config.languages:
                for file_path in path.rglob(f"*{ext}"):
                    # Skip common directories
                    skip_dirs = ['node_modules', 'venv', '.venv', '__pycache__', '.git', 'dist', 'build']
                    if not any(skip in str(file_path) for skip in skip_dirs):
                        files.append(str(file_path))
        
        return files
    
    # =========================================================================
    # Parallel Analysis Phase
    # =========================================================================
    
    def _run_analyzer(
        self,
        analyzer_name: str,
        code: str,
        file_path: str,
        language: str,
    ) -> list[Issue]:
        """Run a single analyzer."""
        try:
            if analyzer_name == "syntax" and self.config.check_syntax:
                return self.syntax_agent.analyze(code, file_path, language)
            elif analyzer_name == "security" and self.config.check_security:
                return self.security_agent.analyze(code, file_path, language)
            elif analyzer_name == "style" and self.config.check_style:
                return self.style_agent.analyze(code, file_path, language)
            elif analyzer_name == "performance" and self.config.check_performance:
                return self.performance_agent.analyze(code, file_path, language)
            elif analyzer_name == "dead_code" and self.config.check_dead_code:
                return self.dead_code_agent.analyze(code, file_path, language)
        except Exception as e:
            self.log(f"Error in {analyzer_name} analyzer: {e}", "ERROR")
        
        return []
    
    def analyze_file_parallel(self, code: str, file_path: str, language: str) -> list[Issue]:
        """Run all analyzers in parallel for a single file."""
        all_findings = []
        
        analyzers = ["syntax", "security", "style", "performance", "dead_code"]
        
        if self.config.parallel:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(self._run_analyzer, name, code, file_path, language): name
                    for name in analyzers
                }
                
                for future in as_completed(futures):
                    findings = future.result()
                    if findings:
                        all_findings.append(findings)
        else:
            for name in analyzers:
                findings = self._run_analyzer(name, code, file_path, language)
                if findings:
                    all_findings.append(findings)
        
        return all_findings
    
    def analyze_file(self, file_path: str) -> FileAnalysis:
        """Analyze a single file."""
        language = detect_language(file_path)
        code = self.read_file(file_path)
        
        if not code:
            return FileAnalysis(file_path=file_path, language=language)
        
        lines = code.split('\n')
        
        # Run parallel analysis
        all_findings = self.analyze_file_parallel(code, file_path, language.value)
        
        # Run complexity analysis
        complexity = ComplexityMetrics()
        if self.config.check_complexity:
            complexity = self.complexity_agent.analyze(code, file_path, language.value)
        
        # Run documentation analysis
        doc_metrics = DocumentationMetrics()
        doc_findings = []
        if self.config.check_documentation:
            doc_metrics, doc_findings = self.documentation_agent.analyze(code, file_path, language.value)
            all_findings.append(doc_findings)
        
        # Aggregate findings
        aggregated = self.aggregator.aggregate(all_findings)
        
        # Filter by severity
        filtered = [
            f for f in aggregated
            if self._severity_rank(f.severity) >= self._severity_rank(self.config.min_severity)
        ]
        
        return FileAnalysis(
            file_path=file_path,
            language=language,
            lines_of_code=len([l for l in lines if l.strip() and not l.strip().startswith('#')]),
            lines_of_comments=len([l for l in lines if l.strip().startswith('#')]),
            blank_lines=len([l for l in lines if not l.strip()]),
            complexity=complexity,
            issues=filtered,
        )
    
    def _severity_rank(self, severity: Severity) -> int:
        """Get numeric rank for severity comparison."""
        ranks = {
            Severity.CRITICAL: 5,
            Severity.HIGH: 4,
            Severity.MEDIUM: 3,
            Severity.LOW: 2,
            Severity.INFO: 1,
        }
        return ranks.get(severity, 0)
    
    # =========================================================================
    # Main Review Pipeline
    # =========================================================================
    
    def review(self, path: str) -> ReviewReport:
        """
        Run complete code review pipeline.
        
        Pipeline:
        1. Find and analyze files (parallel)
        2. Aggregate all findings
        3. Prioritize by severity
        4. Generate fix suggestions
        5. Create report
        """
        start_time = datetime.now()
        
        self.log("=" * 60)
        self.log("CODE REVIEW PIPELINE STARTING")
        self.log("=" * 60)
        self.log(f"Target: {path}")
        
        # Find files
        files = self.find_files(path)
        self.log(f"Found {len(files)} files to analyze")
        
        # Phase 1: Parallel File Analysis
        self.log("\n[Phase 1] PARALLEL ANALYSIS")
        self.log("-" * 40)
        
        file_results = []
        all_issues = []
        total_lines = 0
        languages = {}
        
        for file_path in files:
            self.log(f"  Analyzing: {file_path}")
            result = self.analyze_file(file_path)
            file_results.append(result)
            all_issues.extend(result.issues)
            total_lines += result.lines_of_code
            
            lang = result.language.value
            languages[lang] = languages.get(lang, 0) + 1
        
        # Check dependencies if requirements file exists
        dep_issues = []
        if self.config.check_dependencies:
            for dep_file in ["requirements.txt", "package.json"]:
                dep_path = Path(path) / dep_file if Path(path).is_dir() else Path(path).parent / dep_file
                if dep_path.exists():
                    self.log(f"  Scanning dependencies: {dep_file}")
                    content = self.read_file(str(dep_path))
                    if content:
                        lang = "python" if dep_file == "requirements.txt" else "javascript"
                        dep_issues = self.dependency_scanner.scan_requirements(content, lang)
                        all_issues.extend(dep_issues)
        
        # Phase 2: Aggregation
        self.log("\n[Phase 2] AGGREGATION")
        self.log("-" * 40)
        
        aggregated = self.aggregator.aggregate([all_issues])
        self.log(f"  Combined {len(all_issues)} findings, {len(aggregated)} unique")
        
        # Phase 3: Prioritization
        self.log("\n[Phase 3] PRIORITIZATION")
        self.log("-" * 40)
        
        prioritized = self.prioritizer.prioritize(aggregated)
        
        critical = sum(1 for i in prioritized if i.severity == Severity.CRITICAL)
        high = sum(1 for i in prioritized if i.severity == Severity.HIGH)
        self.log(f"  Prioritized: {critical} critical, {high} high severity")
        
        # Phase 4: Fix Suggestions
        self.log("\n[Phase 4] FIX SUGGESTION")
        self.log("-" * 40)
        
        fixes = self.fix_suggester.generate_fixes(prioritized, "")
        auto_fixable = sum(1 for f in fixes if f.fixed_code)
        self.log(f"  Generated {len(fixes)} fix suggestions ({auto_fixable} auto-fixable)")
        
        # Calculate quality metrics
        complexity_avg = ComplexityMetrics()
        if file_results:
            complexity_avg.cyclomatic_complexity = sum(
                f.complexity.cyclomatic_complexity for f in file_results
            ) / len(file_results)
            complexity_avg.lines_of_code = total_lines
        
        quality_metrics = CodeQualityMetrics(
            complexity=complexity_avg,
            testing=TestMetrics(line_coverage=0),  # Would need actual test runner
            documentation=DocumentationMetrics(),
        )
        quality_metrics.calculate_overall()
        
        # Create analysis result
        duration = (datetime.now() - start_time).total_seconds()
        
        self.current_analysis = AnalysisResult(
            result_id=f"REVIEW-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            target_path=path,
            files_analyzed=len(files),
            total_lines=total_lines,
            languages=languages,
            issues=prioritized,
            quality_metrics=quality_metrics,
            file_results=file_results,
            duration_seconds=duration,
        )
        
        # Phase 5: Report Generation
        self.log("\n[Phase 5] REPORT GENERATION")
        self.log("-" * 40)
        
        self.current_report = self.report_generator.generate_report(
            self.current_analysis,
            fixes,
        )
        
        self.log(f"  Report generated: {self.current_report.report_id}")
        
        # Summary
        self.log("\n" + "=" * 60)
        self.log("REVIEW COMPLETE")
        self.log("=" * 60)
        self.log(f"Files: {len(files)} | Lines: {total_lines:,} | Issues: {len(prioritized)}")
        self.log(f"Duration: {duration:.2f}s")
        
        return self.current_report
    
    def review_code(self, code: str, file_name: str = "code.py") -> ReviewReport:
        """Review a code string directly."""
        # Create temp file
        temp_path = Path(self.config.output_dir) / file_name
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_text(code)
        
        try:
            return self.review(str(temp_path))
        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()
    
    def save_report(self, report: ReviewReport = None, format: str = "markdown") -> str:
        """Save the review report."""
        report = report or self.current_report
        if not report:
            raise ValueError("No report to save")
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        if format == "markdown":
            filepath = os.path.join(self.config.output_dir, f"{report.report_id}.md")
            with open(filepath, 'w') as f:
                f.write(report.to_markdown())
        elif format == "json":
            filepath = os.path.join(self.config.output_dir, f"{report.report_id}.json")
            with open(filepath, 'w') as f:
                f.write(report.to_json())
        else:
            raise ValueError(f"Unknown format: {format}")
        
        self.log(f"Report saved to {filepath}")
        return filepath
    
    def get_summary(self) -> dict:
        """Get analysis summary."""
        if not self.current_analysis:
            return {}
        
        return self.current_analysis.to_dict()


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_review(path: str, verbose: bool = True) -> ReviewReport:
    """Quick code review with default settings."""
    config = ReviewConfig(verbose=verbose)
    engine = CodeReviewEngine(config)
    return engine.review(path)


def review_code(code: str, language: str = "python") -> ReviewReport:
    """Review a code string."""
    config = ReviewConfig(verbose=False)
    engine = CodeReviewEngine(config)
    ext = {"python": ".py", "javascript": ".js", "typescript": ".ts"}.get(language, ".py")
    return engine.review_code(code, f"code{ext}")


def get_pipeline_diagram() -> str:
    """Get the pipeline architecture diagram."""
    return get_agent_pipeline()
