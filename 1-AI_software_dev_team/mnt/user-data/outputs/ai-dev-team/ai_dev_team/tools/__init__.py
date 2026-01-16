"""
AI Dev Team - Tools

Tools available to agents for code manipulation, validation, and execution.
"""

import os
import ast
import subprocess
import tempfile
import json
import re
from typing import Optional
from pathlib import Path
from crewai.tools import tool


# =============================================================================
# File Operations
# =============================================================================

@tool("Write File")
def file_writer(filepath: str, content: str) -> str:
    """
    Write content to a file. Creates directories if they don't exist.
    
    Args:
        filepath: Path to the file (relative to output directory)
        content: Content to write to the file
    
    Returns:
        Success message with filepath
    """
    try:
        # Get output directory from environment or use default
        output_dir = os.environ.get("AI_DEV_OUTPUT_DIR", "./output")
        full_path = os.path.join(output_dir, filepath)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # Write content
        with open(full_path, 'w') as f:
            f.write(content)
        
        return f"✅ Successfully wrote {len(content)} characters to {filepath}"
    except Exception as e:
        return f"❌ Error writing file: {str(e)}"


@tool("Read File")
def file_reader(filepath: str) -> str:
    """
    Read content from a file.
    
    Args:
        filepath: Path to the file
    
    Returns:
        File content or error message
    """
    try:
        output_dir = os.environ.get("AI_DEV_OUTPUT_DIR", "./output")
        full_path = os.path.join(output_dir, filepath)
        
        with open(full_path, 'r') as f:
            content = f.read()
        
        return content
    except Exception as e:
        return f"❌ Error reading file: {str(e)}"


# =============================================================================
# Code Validation
# =============================================================================

@tool("Validate Python Code")
def code_validator(code: str) -> str:
    """
    Validate Python code syntax and check for common issues.
    
    Args:
        code: Python code to validate
    
    Returns:
        Validation results with any errors or warnings
    """
    results = []
    
    # Check syntax
    try:
        ast.parse(code)
        results.append("✅ Syntax: Valid Python syntax")
    except SyntaxError as e:
        results.append(f"❌ Syntax Error: Line {e.lineno}: {e.msg}")
        return "\n".join(results)
    
    # Check for common issues
    issues = []
    
    # Check for bare except
    if re.search(r'\bexcept\s*:', code):
        issues.append("⚠️ Found bare 'except:' clause - consider catching specific exceptions")
    
    # Check for print statements in production code
    print_count = len(re.findall(r'\bprint\s*\(', code))
    if print_count > 5:
        issues.append(f"⚠️ Found {print_count} print statements - consider using logging")
    
    # Check for hardcoded secrets patterns
    secret_patterns = [
        r'password\s*=\s*["\'][^"\']+["\']',
        r'api_key\s*=\s*["\'][^"\']+["\']',
        r'secret\s*=\s*["\'][^"\']+["\']',
    ]
    for pattern in secret_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            issues.append("🔴 SECURITY: Possible hardcoded secret detected")
    
    # Check for TODO/FIXME comments
    todos = len(re.findall(r'#\s*(TODO|FIXME|XXX|HACK)', code, re.IGNORECASE))
    if todos > 0:
        issues.append(f"📝 Found {todos} TODO/FIXME comments")
    
    if issues:
        results.extend(issues)
    else:
        results.append("✅ No common issues detected")
    
    # Basic metrics
    lines = code.count('\n') + 1
    functions = len(re.findall(r'\bdef\s+\w+', code))
    classes = len(re.findall(r'\bclass\s+\w+', code))
    
    results.append(f"\n📊 Metrics: {lines} lines, {functions} functions, {classes} classes")
    
    return "\n".join(results)


@tool("Lint Python Code")
def linter(code: str) -> str:
    """
    Run linting checks on Python code using available linters.
    
    Args:
        code: Python code to lint
    
    Returns:
        Linting results
    """
    results = []
    
    # Write to temp file for linting
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name
    
    try:
        # Try ruff first (fastest)
        try:
            result = subprocess.run(
                ['ruff', 'check', temp_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                results.append("✅ Ruff: No issues found")
            else:
                results.append(f"Ruff issues:\n{result.stdout}")
        except FileNotFoundError:
            # Try pylint
            try:
                result = subprocess.run(
                    ['pylint', temp_path, '--disable=C0114,C0115,C0116', '--output-format=text'],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                # Extract score
                score_match = re.search(r'Your code has been rated at ([\d.]+)', result.stdout)
                if score_match:
                    score = float(score_match.group(1))
                    results.append(f"📊 Pylint Score: {score}/10")
                results.append(result.stdout[:1000] if len(result.stdout) > 1000 else result.stdout)
            except FileNotFoundError:
                results.append("⚠️ No linter available (install ruff or pylint)")
    finally:
        os.unlink(temp_path)
    
    return "\n".join(results) if results else "✅ Code passes linting"


# =============================================================================
# Code Execution
# =============================================================================

@tool("Execute Python Code")
def code_executor(code: str, timeout: int = 30) -> str:
    """
    Execute Python code in a sandboxed environment.
    
    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds
    
    Returns:
        Execution output or error
    """
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name
    
    try:
        result = subprocess.run(
            ['python', temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
        )
        
        output = []
        if result.stdout:
            output.append(f"📤 Output:\n{result.stdout}")
        if result.stderr:
            output.append(f"⚠️ Stderr:\n{result.stderr}")
        if result.returncode != 0:
            output.append(f"❌ Exit code: {result.returncode}")
        else:
            output.append("✅ Execution successful")
        
        return "\n".join(output)
    
    except subprocess.TimeoutExpired:
        return f"❌ Execution timed out after {timeout} seconds"
    except Exception as e:
        return f"❌ Execution error: {str(e)}"
    finally:
        os.unlink(temp_path)


# =============================================================================
# Testing
# =============================================================================

@tool("Run Tests")
def test_runner(test_code: str, source_code: Optional[str] = None) -> str:
    """
    Run pytest tests on the provided test code.
    
    Args:
        test_code: Test code to run
        source_code: Optional source code that tests depend on
    
    Returns:
        Test results
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write source code if provided
        if source_code:
            source_path = os.path.join(temp_dir, 'source.py')
            with open(source_path, 'w') as f:
                f.write(source_code)
        
        # Write test code
        test_path = os.path.join(temp_dir, 'test_code.py')
        with open(test_path, 'w') as f:
            f.write(test_code)
        
        try:
            result = subprocess.run(
                ['python', '-m', 'pytest', test_path, '-v', '--tb=short'],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=temp_dir,
                env={**os.environ, 'PYTHONPATH': temp_dir}
            )
            
            output = result.stdout + result.stderr
            
            # Parse results
            passed = len(re.findall(r'PASSED', output))
            failed = len(re.findall(r'FAILED', output))
            
            summary = f"\n📊 Results: {passed} passed, {failed} failed"
            
            return output + summary
        
        except subprocess.TimeoutExpired:
            return "❌ Tests timed out after 120 seconds"
        except Exception as e:
            return f"❌ Test error: {str(e)}"


# =============================================================================
# Security Scanning
# =============================================================================

@tool("Security Scan")
def security_scanner(code: str) -> str:
    """
    Scan code for security vulnerabilities.
    
    Args:
        code: Code to scan
    
    Returns:
        Security scan results
    """
    findings = []
    
    # OWASP-based checks
    checks = [
        # SQL Injection
        (r'execute\s*\(\s*["\'].*%s', "SQL Injection: Use parameterized queries instead of string formatting"),
        (r'execute\s*\(\s*f["\']', "SQL Injection: Use parameterized queries instead of f-strings"),
        (r'\.format\s*\(.*\).*execute', "SQL Injection: Possible SQL injection via .format()"),
        
        # Command Injection
        (r'os\.system\s*\(', "Command Injection: Use subprocess with shell=False instead of os.system"),
        (r'subprocess.*shell\s*=\s*True', "Command Injection: Avoid shell=True in subprocess"),
        (r'eval\s*\(', "Code Injection: Avoid eval() - use ast.literal_eval() for data"),
        (r'exec\s*\(', "Code Injection: Avoid exec() - high risk of code injection"),
        
        # XSS
        (r'\.innerHTML\s*=', "XSS: Use textContent instead of innerHTML"),
        (r'document\.write\s*\(', "XSS: Avoid document.write"),
        
        # Hardcoded Secrets
        (r'password\s*=\s*["\'][^"\']{8,}["\']', "Secrets: Possible hardcoded password"),
        (r'api_key\s*=\s*["\'][^"\']{16,}["\']', "Secrets: Possible hardcoded API key"),
        (r'secret_key\s*=\s*["\'][^"\']+["\']', "Secrets: Possible hardcoded secret"),
        (r'AWS_ACCESS_KEY', "Secrets: Possible AWS credentials"),
        
        # Insecure Cryptography
        (r'md5\s*\(', "Crypto: MD5 is cryptographically broken - use SHA-256+"),
        (r'sha1\s*\(', "Crypto: SHA1 is weak - use SHA-256+"),
        (r'DES\b', "Crypto: DES is insecure - use AES"),
        
        # Path Traversal
        (r'open\s*\([^)]*\+', "Path Traversal: Validate file paths before opening"),
        (r'os\.path\.join.*request', "Path Traversal: Validate user input in file paths"),
        
        # Insecure Deserialization
        (r'pickle\.load', "Deserialization: pickle.load is unsafe with untrusted data"),
        (r'yaml\.load\s*\([^)]*\)', "Deserialization: Use yaml.safe_load instead"),
        
        # CORS
        (r'Access-Control-Allow-Origin.*\*', "CORS: Wildcard origin can be risky"),
        
        # Debug/Development
        (r'DEBUG\s*=\s*True', "Config: Debug mode should be False in production"),
        (r'verify\s*=\s*False', "SSL: SSL verification disabled"),
    ]
    
    for pattern, message in checks:
        matches = re.findall(pattern, code, re.IGNORECASE)
        if matches:
            findings.append(f"🔴 {message}")
    
    # Check for missing security headers (if it looks like web code)
    if 'flask' in code.lower() or 'fastapi' in code.lower() or 'django' in code.lower():
        security_headers = [
            'X-Content-Type-Options',
            'X-Frame-Options',
            'Content-Security-Policy',
            'Strict-Transport-Security',
        ]
        for header in security_headers:
            if header not in code:
                findings.append(f"⚠️ Missing security header: {header}")
    
    if findings:
        result = "🔒 Security Scan Results:\n\n" + "\n".join(findings)
        result += f"\n\n📊 Total issues: {len(findings)}"
    else:
        result = "✅ Security Scan: No obvious vulnerabilities detected\n"
        result += "Note: This is a basic scan. Consider using dedicated tools like Bandit, Semgrep, or Snyk for comprehensive analysis."
    
    return result


# =============================================================================
# Project Structure
# =============================================================================

@tool("Create Project Structure")
def create_project_structure(project_name: str, structure: str) -> str:
    """
    Create a project directory structure.
    
    Args:
        project_name: Name of the project
        structure: JSON string describing the structure
    
    Returns:
        Status of created directories and files
    """
    try:
        output_dir = os.environ.get("AI_DEV_OUTPUT_DIR", "./output")
        project_dir = os.path.join(output_dir, project_name)
        
        # Parse structure
        try:
            dirs = json.loads(structure)
        except json.JSONDecodeError:
            # Assume it's a simple list of paths
            dirs = [d.strip() for d in structure.split('\n') if d.strip()]
        
        created = []
        for path in dirs:
            full_path = os.path.join(project_dir, path)
            if path.endswith('/'):
                os.makedirs(full_path, exist_ok=True)
                created.append(f"📁 {path}")
            else:
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                Path(full_path).touch()
                created.append(f"📄 {path}")
        
        return f"✅ Created project structure:\n" + "\n".join(created)
    
    except Exception as e:
        return f"❌ Error creating structure: {str(e)}"
