"""
Code Review - Code Quality Analyzers

Analyzers for:
- Syntax and error detection
- Code style (PEP8, ESLint-like)
- Performance anti-patterns
- Complexity metrics
- Dead code detection
- Documentation coverage
"""

import re
import ast
from typing import Optional
from dataclasses import dataclass

from ..models import (
    Issue, CodeLocation, CodeSnippet,
    Severity, IssueCategory, FixDifficulty, Language,
    ComplexityMetrics, DocumentationMetrics
)


# =============================================================================
# Syntax Analyzer
# =============================================================================

class SyntaxAnalyzer:
    """Analyzes code for syntax errors and issues."""
    
    def __init__(self):
        self.findings: list[Issue] = []
    
    def analyze_python(self, code: str, file_path: str) -> list[Issue]:
        """Analyze Python code for syntax errors."""
        self.findings = []
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            self.findings.append(Issue(
                issue_id="",
                title="Syntax Error",
                description=str(e.msg) if e.msg else "Invalid Python syntax",
                category=IssueCategory.SYNTAX,
                severity=Severity.CRITICAL,
                location=CodeLocation(
                    file_path=file_path,
                    line_start=e.lineno or 1,
                    column_start=e.offset or 0,
                ),
                fix_suggestion="Fix the syntax error",
                analyzer="syntax_checker",
            ))
        
        # Check for common issues
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Trailing whitespace
            if line.rstrip() != line and line.strip():
                self.findings.append(Issue(
                    issue_id="",
                    title="Trailing Whitespace",
                    description="Line has trailing whitespace",
                    category=IssueCategory.STYLE,
                    severity=Severity.INFO,
                    location=CodeLocation(file_path=file_path, line_start=line_num),
                    fix_suggestion="Remove trailing whitespace",
                    auto_fixable=True,
                    fix_difficulty=FixDifficulty.TRIVIAL,
                    analyzer="syntax_checker",
                ))
            
            # Tab/space mixing (check for tabs)
            if '\t' in line and '    ' in line:
                self.findings.append(Issue(
                    issue_id="",
                    title="Mixed Tabs and Spaces",
                    description="Line mixes tabs and spaces for indentation",
                    category=IssueCategory.STYLE,
                    severity=Severity.LOW,
                    location=CodeLocation(file_path=file_path, line_start=line_num),
                    fix_suggestion="Use consistent indentation (spaces preferred)",
                    auto_fixable=True,
                    fix_difficulty=FixDifficulty.TRIVIAL,
                    analyzer="syntax_checker",
                ))
        
        return self.findings
    
    def analyze_javascript(self, code: str, file_path: str) -> list[Issue]:
        """Analyze JavaScript code for common syntax issues."""
        self.findings = []
        lines = code.split('\n')
        
        # Common JS syntax patterns that might be issues
        patterns = [
            (r'==(?!=)', "Loose Equality", "Use === instead of == for strict comparison", Severity.LOW),
            (r'!=(?!=)', "Loose Inequality", "Use !== instead of != for strict comparison", Severity.LOW),
            (r'\bvar\s+', "var Usage", "Consider using const or let instead of var", Severity.INFO),
        ]
        
        for line_num, line in enumerate(lines, 1):
            for pattern, title, suggestion, severity in patterns:
                if re.search(pattern, line):
                    self.findings.append(Issue(
                        issue_id="",
                        title=title,
                        description=f"{title} detected",
                        category=IssueCategory.STYLE,
                        severity=severity,
                        location=CodeLocation(file_path=file_path, line_start=line_num),
                        fix_suggestion=suggestion,
                        fix_difficulty=FixDifficulty.EASY,
                        analyzer="syntax_checker",
                    ))
        
        return self.findings
    
    def analyze(self, code: str, file_path: str, language: str = "python") -> list[Issue]:
        """Analyze code based on language."""
        if language == "python":
            return self.analyze_python(code, file_path)
        elif language in ["javascript", "typescript"]:
            return self.analyze_javascript(code, file_path)
        return []


# =============================================================================
# Style Analyzer
# =============================================================================

class StyleAnalyzer:
    """Analyzes code for style issues and best practices."""
    
    def __init__(self):
        self.findings: list[Issue] = []
        self.max_line_length = 120
        self.max_function_length = 50
    
    def analyze_python(self, code: str, file_path: str) -> list[Issue]:
        """Analyze Python code style."""
        self.findings = []
        lines = code.split('\n')
        
        # Line length
        for line_num, line in enumerate(lines, 1):
            if len(line) > self.max_line_length:
                self.findings.append(Issue(
                    issue_id="",
                    title="Line Too Long",
                    description=f"Line exceeds {self.max_line_length} characters ({len(line)} chars)",
                    category=IssueCategory.STYLE,
                    severity=Severity.INFO,
                    location=CodeLocation(file_path=file_path, line_start=line_num),
                    fix_suggestion="Break line into multiple lines",
                    fix_difficulty=FixDifficulty.EASY,
                    analyzer="style_checker",
                ))
        
        # Try to parse AST for more checks
        try:
            tree = ast.parse(code)
            self._check_python_ast(tree, file_path, lines)
        except SyntaxError:
            pass
        
        # Naming conventions
        for line_num, line in enumerate(lines, 1):
            # Class names should be CamelCase
            class_match = re.search(r'class\s+([a-z][a-z0-9_]*)\s*[:\(]', line)
            if class_match:
                self.findings.append(Issue(
                    issue_id="",
                    title="Class Name Not CamelCase",
                    description=f"Class '{class_match.group(1)}' should use CamelCase naming",
                    category=IssueCategory.STYLE,
                    severity=Severity.LOW,
                    location=CodeLocation(file_path=file_path, line_start=line_num),
                    fix_suggestion="Use CamelCase for class names",
                    fix_difficulty=FixDifficulty.EASY,
                    analyzer="style_checker",
                ))
            
            # Constants should be UPPER_CASE (module level assignments with simple values)
            const_match = re.match(r'^([a-z][a-z0-9_]*)\s*=\s*[\d\'"]+', line)
            if const_match and not line.strip().startswith('#'):
                name = const_match.group(1)
                if not name.startswith('_') and name.isupper() == False and line_num < 50:
                    # Only flag if it looks like a constant at module level
                    pass  # Skip for now - needs context
        
        # Import ordering
        import_lines = []
        for line_num, line in enumerate(lines, 1):
            if line.startswith('import ') or line.startswith('from '):
                import_lines.append((line_num, line))
        
        if import_lines:
            # Check if imports are grouped (stdlib, third-party, local)
            # Simplified check: just ensure imports are at the top
            last_import_line = import_lines[-1][0]
            first_code_line = 0
            for line_num, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped and not stripped.startswith('#') and not stripped.startswith('import') and not stripped.startswith('from') and not stripped.startswith('"""') and not stripped.startswith("'''"):
                    first_code_line = line_num
                    break
            
            if first_code_line and first_code_line < last_import_line:
                self.findings.append(Issue(
                    issue_id="",
                    title="Import Not At Top",
                    description="Imports should be at the top of the file",
                    category=IssueCategory.STYLE,
                    severity=Severity.LOW,
                    location=CodeLocation(file_path=file_path, line_start=last_import_line),
                    fix_suggestion="Move imports to the top of the file",
                    fix_difficulty=FixDifficulty.EASY,
                    analyzer="style_checker",
                ))
        
        return self.findings
    
    def _check_python_ast(self, tree: ast.AST, file_path: str, lines: list[str]):
        """Check Python AST for style issues."""
        
        for node in ast.walk(tree):
            # Function length
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                if func_lines > self.max_function_length:
                    self.findings.append(Issue(
                        issue_id="",
                        title="Function Too Long",
                        description=f"Function '{node.name}' is {func_lines} lines (max {self.max_function_length})",
                        category=IssueCategory.MAINTAINABILITY,
                        severity=Severity.MEDIUM,
                        location=CodeLocation(file_path=file_path, line_start=node.lineno),
                        fix_suggestion="Consider breaking this function into smaller functions",
                        fix_difficulty=FixDifficulty.MODERATE,
                        analyzer="style_checker",
                    ))
                
                # Too many arguments
                arg_count = len(node.args.args)
                if arg_count > 5:
                    self.findings.append(Issue(
                        issue_id="",
                        title="Too Many Arguments",
                        description=f"Function '{node.name}' has {arg_count} arguments (max 5 recommended)",
                        category=IssueCategory.MAINTAINABILITY,
                        severity=Severity.LOW,
                        location=CodeLocation(file_path=file_path, line_start=node.lineno),
                        fix_suggestion="Consider using a configuration object or dataclass",
                        fix_difficulty=FixDifficulty.MODERATE,
                        analyzer="style_checker",
                    ))
            
            # Nested functions too deep
            if isinstance(node, ast.FunctionDef):
                depth = 0
                parent = node
                for child in ast.walk(node):
                    if isinstance(child, ast.FunctionDef) and child != node:
                        depth += 1
                if depth > 2:
                    self.findings.append(Issue(
                        issue_id="",
                        title="Deeply Nested Functions",
                        description=f"Function '{node.name}' has {depth} levels of nested functions",
                        category=IssueCategory.MAINTAINABILITY,
                        severity=Severity.MEDIUM,
                        location=CodeLocation(file_path=file_path, line_start=node.lineno),
                        fix_suggestion="Refactor to reduce nesting",
                        fix_difficulty=FixDifficulty.MODERATE,
                        analyzer="style_checker",
                    ))
            
            # Bare except
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    self.findings.append(Issue(
                        issue_id="",
                        title="Bare Except Clause",
                        description="Bare except catches all exceptions including KeyboardInterrupt",
                        category=IssueCategory.STYLE,
                        severity=Severity.MEDIUM,
                        location=CodeLocation(file_path=file_path, line_start=node.lineno),
                        fix_suggestion="Specify exception type: 'except Exception:'",
                        fix_code="except Exception:",
                        fix_difficulty=FixDifficulty.EASY,
                        analyzer="style_checker",
                    ))
    
    def analyze(self, code: str, file_path: str, language: str = "python") -> list[Issue]:
        """Analyze code style."""
        if language == "python":
            return self.analyze_python(code, file_path)
        return []


# =============================================================================
# Performance Analyzer
# =============================================================================

class PerformanceAnalyzer:
    """Analyzes code for performance anti-patterns."""
    
    def __init__(self):
        self.findings: list[Issue] = []
    
    def analyze_python(self, code: str, file_path: str) -> list[Issue]:
        """Analyze Python code for performance issues."""
        self.findings = []
        lines = code.split('\n')
        
        # Pattern-based checks
        patterns = [
            # String concatenation in loop
            (r'for\s+.*:\s*\n(?:.*\n)*?\s+\w+\s*\+=\s*[\'"]', 
             "String Concatenation in Loop",
             "String concatenation in loops is inefficient; use ''.join() or list comprehension",
             Severity.MEDIUM),
            
            # Repeated list append vs comprehension
            (r'for\s+\w+\s+in\s+.*:\s*\n\s+\w+\.append\s*\(',
             "Append in Loop",
             "Consider using list comprehension instead of append in loop",
             Severity.LOW),
            
            # Reading entire file into memory
            (r'\.read\(\s*\)',
             "Reading Entire File",
             "Reading entire file at once may cause memory issues; consider iterating line by line",
             Severity.LOW),
            
            # Using + for path concatenation
            (r'[\'"][^\'\"]+[\'\"]\s*\+\s*[\'\"]\/',
             "String Path Concatenation",
             "Use os.path.join() or pathlib for path concatenation",
             Severity.LOW),
            
            # Global variable in function
            (r'def\s+\w+\s*\([^)]*\)\s*:\s*\n(?:.*\n)*?\s+global\s+',
             "Global Variable Usage",
             "Avoid using global variables; pass as parameters or use class attributes",
             Severity.MEDIUM),
        ]
        
        for pattern, title, suggestion, severity in patterns:
            for match in re.finditer(pattern, code, re.MULTILINE):
                line_num = code[:match.start()].count('\n') + 1
                
                self.findings.append(Issue(
                    issue_id="",
                    title=title,
                    description=title,
                    category=IssueCategory.PERFORMANCE,
                    severity=severity,
                    location=CodeLocation(file_path=file_path, line_start=line_num),
                    fix_suggestion=suggestion,
                    fix_difficulty=FixDifficulty.MODERATE,
                    analyzer="performance_analyzer",
                ))
        
        # AST-based checks
        try:
            tree = ast.parse(code)
            self._check_python_ast(tree, file_path, lines)
        except SyntaxError:
            pass
        
        return self.findings
    
    def _check_python_ast(self, tree: ast.AST, file_path: str, lines: list[str]):
        """Check Python AST for performance issues."""
        
        for node in ast.walk(tree):
            # Nested loops (O(n²) complexity)
            if isinstance(node, (ast.For, ast.While)):
                for child in ast.walk(node):
                    if isinstance(child, (ast.For, ast.While)) and child != node:
                        # Check if the inner loop iterates over something related to outer
                        self.findings.append(Issue(
                            issue_id="",
                            title="Nested Loop",
                            description="Nested loops may indicate O(n²) complexity",
                            category=IssueCategory.PERFORMANCE,
                            severity=Severity.LOW,
                            location=CodeLocation(file_path=file_path, line_start=node.lineno),
                            fix_suggestion="Consider if this can be optimized with a dictionary or set lookup",
                            fix_difficulty=FixDifficulty.MODERATE,
                            analyzer="performance_analyzer",
                        ))
                        break
            
            # List membership check (should be set)
            if isinstance(node, ast.Compare):
                for op in node.ops:
                    if isinstance(op, (ast.In, ast.NotIn)):
                        for comparator in node.comparators:
                            if isinstance(comparator, ast.List) and len(comparator.elts) > 3:
                                self.findings.append(Issue(
                                    issue_id="",
                                    title="List Membership Check",
                                    description="Use set for membership checks on large collections",
                                    category=IssueCategory.PERFORMANCE,
                                    severity=Severity.LOW,
                                    location=CodeLocation(file_path=file_path, line_start=node.lineno),
                                    fix_suggestion="Convert list to set for O(1) lookup",
                                    fix_difficulty=FixDifficulty.EASY,
                                    analyzer="performance_analyzer",
                                ))
    
    def analyze(self, code: str, file_path: str, language: str = "python") -> list[Issue]:
        """Analyze code for performance issues."""
        if language == "python":
            return self.analyze_python(code, file_path)
        return []


# =============================================================================
# Complexity Analyzer
# =============================================================================

class ComplexityAnalyzer:
    """Calculates code complexity metrics."""
    
    def __init__(self):
        self.metrics = ComplexityMetrics()
    
    def calculate_cyclomatic_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity for Python code."""
        complexity = 1  # Base complexity
        
        # Decision points
        decision_patterns = [
            r'\bif\b',
            r'\belif\b',
            r'\bfor\b',
            r'\bwhile\b',
            r'\band\b',
            r'\bor\b',
            r'\bexcept\b',
            r'\bwith\b',
            r'\bassert\b',
            r'\?\s*:',  # Ternary
        ]
        
        for pattern in decision_patterns:
            complexity += len(re.findall(pattern, code))
        
        return complexity
    
    def analyze_python(self, code: str, file_path: str) -> ComplexityMetrics:
        """Analyze Python code complexity."""
        lines = code.split('\n')
        
        # Basic metrics
        total_lines = len(lines)
        blank_lines = sum(1 for line in lines if not line.strip())
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        code_lines = total_lines - blank_lines - comment_lines
        
        # Cyclomatic complexity
        cyclomatic = self.calculate_cyclomatic_complexity(code)
        
        # Calculate metrics per function
        try:
            tree = ast.parse(code)
            functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            
            if functions:
                # Average complexity per function
                func_complexities = []
                for func in functions:
                    func_code = ast.get_source_segment(code, func) if hasattr(ast, 'get_source_segment') else ""
                    if func_code:
                        func_complexities.append(self.calculate_cyclomatic_complexity(func_code))
                
                if func_complexities:
                    cyclomatic = sum(func_complexities) / len(func_complexities)
        except:
            pass
        
        # Maintainability Index (simplified)
        # MI = 171 - 5.2 * ln(HV) - 0.23 * CC - 16.2 * ln(LOC)
        import math
        halstead_volume = max(1, code_lines * 10)  # Simplified
        mi = max(0, min(100, 
            171 - 5.2 * math.log(halstead_volume) - 
            0.23 * cyclomatic - 
            16.2 * math.log(max(1, code_lines))
        ))
        
        self.metrics = ComplexityMetrics(
            cyclomatic_complexity=cyclomatic,
            cognitive_complexity=cyclomatic * 1.2,  # Simplified approximation
            halstead_difficulty=halstead_volume / 100,
            maintainability_index=mi,
            lines_of_code=code_lines,
            comment_ratio=comment_lines / max(1, code_lines) * 100,
        )
        
        return self.metrics
    
    def analyze(self, code: str, file_path: str, language: str = "python") -> ComplexityMetrics:
        """Analyze code complexity."""
        if language == "python":
            return self.analyze_python(code, file_path)
        return ComplexityMetrics()


# =============================================================================
# Documentation Analyzer
# =============================================================================

class DocumentationAnalyzer:
    """Analyzes code documentation coverage."""
    
    def __init__(self):
        self.metrics = DocumentationMetrics()
        self.findings: list[Issue] = []
    
    def analyze_python(self, code: str, file_path: str) -> tuple[DocumentationMetrics, list[Issue]]:
        """Analyze Python documentation."""
        self.findings = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return DocumentationMetrics(), []
        
        functions = []
        classes = []
        documented_functions = []
        documented_classes = []
        missing_docstrings = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.name)
                
                # Check for docstring
                if (node.body and isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, (ast.Str, ast.Constant))):
                    documented_functions.append(node.name)
                else:
                    # Only flag public functions
                    if not node.name.startswith('_'):
                        missing_docstrings.append(node.name)
                        self.findings.append(Issue(
                            issue_id="",
                            title="Missing Docstring",
                            description=f"Function '{node.name}' is missing a docstring",
                            category=IssueCategory.DOCUMENTATION,
                            severity=Severity.LOW,
                            location=CodeLocation(file_path=file_path, line_start=node.lineno),
                            fix_suggestion="Add a docstring describing the function's purpose, parameters, and return value",
                            fix_difficulty=FixDifficulty.EASY,
                            analyzer="documentation_analyzer",
                        ))
            
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
                
                if (node.body and isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, (ast.Str, ast.Constant))):
                    documented_classes.append(node.name)
                else:
                    missing_docstrings.append(node.name)
                    self.findings.append(Issue(
                        issue_id="",
                        title="Missing Class Docstring",
                        description=f"Class '{node.name}' is missing a docstring",
                        category=IssueCategory.DOCUMENTATION,
                        severity=Severity.LOW,
                        location=CodeLocation(file_path=file_path, line_start=node.lineno),
                        fix_suggestion="Add a docstring describing the class's purpose",
                        fix_difficulty=FixDifficulty.EASY,
                        analyzer="documentation_analyzer",
                    ))
        
        # Calculate coverage
        total_documented = len(documented_functions) + len(documented_classes)
        total_items = len(functions) + len(classes)
        
        self.metrics = DocumentationMetrics(
            docstring_coverage=total_documented / max(1, total_items) * 100,
            public_api_documented=len(documented_functions) / max(1, len([f for f in functions if not f.startswith('_')])) * 100,
            missing_docstrings=missing_docstrings,
        )
        
        return self.metrics, self.findings
    
    def analyze(self, code: str, file_path: str, language: str = "python") -> tuple[DocumentationMetrics, list[Issue]]:
        """Analyze documentation."""
        if language == "python":
            return self.analyze_python(code, file_path)
        return DocumentationMetrics(), []


# =============================================================================
# Dead Code Analyzer
# =============================================================================

class DeadCodeAnalyzer:
    """Detects potentially unused code."""
    
    def __init__(self):
        self.findings: list[Issue] = []
    
    def analyze_python(self, code: str, file_path: str) -> list[Issue]:
        """Analyze Python code for dead/unused code."""
        self.findings = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        
        # Collect all definitions
        defined_names = set()
        used_names = set()
        
        for node in ast.walk(tree):
            # Definitions
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith('_'):
                    defined_names.add((node.name, 'function', node.lineno))
            elif isinstance(node, ast.ClassDef):
                defined_names.add((node.name, 'class', node.lineno))
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                if not node.id.startswith('_'):
                    defined_names.add((node.id, 'variable', node.lineno))
            
            # Uses
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                used_names.add(node.attr)
        
        # Find unused (simplified - doesn't account for all uses)
        for name, kind, line_num in defined_names:
            if name not in used_names and not name.startswith('test_') and name not in ['main', '__init__', '__str__', '__repr__']:
                self.findings.append(Issue(
                    issue_id="",
                    title=f"Potentially Unused {kind.capitalize()}",
                    description=f"'{name}' appears to be defined but never used",
                    category=IssueCategory.MAINTAINABILITY,
                    severity=Severity.LOW,
                    location=CodeLocation(file_path=file_path, line_start=line_num),
                    fix_suggestion=f"Remove unused {kind} or add usage",
                    fix_difficulty=FixDifficulty.EASY,
                    analyzer="dead_code_analyzer",
                ))
        
        # Unreachable code after return
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for i, stmt in enumerate(node.body[:-1]):
                    if isinstance(stmt, ast.Return):
                        next_stmt = node.body[i + 1]
                        self.findings.append(Issue(
                            issue_id="",
                            title="Unreachable Code",
                            description="Code after return statement will never execute",
                            category=IssueCategory.MAINTAINABILITY,
                            severity=Severity.MEDIUM,
                            location=CodeLocation(
                                file_path=file_path, 
                                line_start=next_stmt.lineno if hasattr(next_stmt, 'lineno') else node.lineno
                            ),
                            fix_suggestion="Remove unreachable code",
                            fix_difficulty=FixDifficulty.EASY,
                            analyzer="dead_code_analyzer",
                        ))
        
        return self.findings
    
    def analyze(self, code: str, file_path: str, language: str = "python") -> list[Issue]:
        """Analyze for dead code."""
        if language == "python":
            return self.analyze_python(code, file_path)
        return []
