"""
🔒 Code Review & Security Audit System - FastAPI Backend

REST API for code review operations.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
from pathlib import Path
from datetime import datetime
import tempfile
import os

app = FastAPI(
    title="Code Review API",
    description="🔒 Automated Code Review & Security Audit System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Models
# =============================================================================

class CodeReviewRequest(BaseModel):
    code: str
    language: str = "python"
    file_name: str = "code.py"


class ReviewOptions(BaseModel):
    check_security: bool = True
    check_style: bool = True
    check_performance: bool = True
    check_documentation: bool = True


# =============================================================================
# State
# =============================================================================

class AppState:
    def __init__(self):
        self.reviews: Dict[str, dict] = {}


state = AppState()


# =============================================================================
# Routes
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve frontend."""
    html_path = Path(__file__).parent / "frontend" / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return """
    <html>
        <head><title>Code Review System</title></head>
        <body style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: white; font-family: sans-serif; padding: 40px; text-align: center;">
            <h1>🔒 Code Review & Security Audit System</h1>
            <p>Visit <a href="/docs" style="color: #4ecdc4;">/docs</a> for API documentation</p>
        </body>
    </html>
    """


@app.get("/api/status")
async def get_status():
    """Get API status."""
    return {
        "status": "ready",
        "version": "1.0.0",
        "analyzers": ["syntax", "security", "style", "performance", "documentation"],
    }


@app.post("/api/review")
async def review_code(request: CodeReviewRequest):
    """Review code snippet."""
    from code_review import CodeReviewEngine, ReviewConfig
    
    config = ReviewConfig(verbose=False)
    engine = CodeReviewEngine(config)
    
    # Create temp file
    ext = {"python": ".py", "javascript": ".js", "typescript": ".ts"}.get(request.language, ".py")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=False) as f:
        f.write(request.code)
        temp_path = f.name
    
    try:
        report = engine.review(temp_path)
        
        # Store result
        review_id = report.report_id
        state.reviews[review_id] = {
            "report": report,
            "timestamp": datetime.now().isoformat(),
        }
        
        return {
            "review_id": review_id,
            "summary": {
                "files_analyzed": report.analysis.files_analyzed,
                "total_lines": report.analysis.total_lines,
                "issues": {
                    "total": report.analysis.issue_count,
                    "critical": report.analysis.critical_count,
                    "high": report.analysis.high_count,
                    "medium": report.analysis.medium_count,
                    "low": report.analysis.low_count,
                },
                "quality_score": report.analysis.quality_metrics.overall_score,
            },
            "issues": [
                {
                    "id": i.issue_id,
                    "title": i.title,
                    "description": i.description,
                    "severity": i.severity.value,
                    "category": i.category.value,
                    "location": str(i.location),
                    "cwe_id": i.cwe_id,
                    "fix_suggestion": i.fix_suggestion,
                    "auto_fixable": i.auto_fixable,
                }
                for i in report.analysis.issues[:50]  # Limit to 50
            ],
            "recommendations": report.recommendations,
            "executive_summary": report.executive_summary,
        }
    
    finally:
        os.unlink(temp_path)


@app.post("/api/review/file")
async def review_file(file: UploadFile = File(...)):
    """Review uploaded file."""
    from code_review import CodeReviewEngine, ReviewConfig
    
    content = await file.read()
    code = content.decode('utf-8')
    
    request = CodeReviewRequest(
        code=code,
        file_name=file.filename or "uploaded.py",
        language="python" if file.filename.endswith(".py") else "javascript",
    )
    
    return await review_code(request)


@app.get("/api/review/{review_id}")
async def get_review(review_id: str):
    """Get a previous review result."""
    if review_id not in state.reviews:
        raise HTTPException(status_code=404, detail="Review not found")
    
    review = state.reviews[review_id]
    report = review["report"]
    
    return {
        "review_id": review_id,
        "timestamp": review["timestamp"],
        "markdown": report.to_markdown(),
    }


@app.get("/api/review/{review_id}/markdown")
async def get_review_markdown(review_id: str):
    """Get review as markdown."""
    if review_id not in state.reviews:
        raise HTTPException(status_code=404, detail="Review not found")
    
    report = state.reviews[review_id]["report"]
    return {"markdown": report.to_markdown()}


@app.get("/api/security-checks")
async def get_security_checks():
    """List available security checks."""
    from code_review.analyzers.security import (
        SQL_INJECTION_PATTERNS, COMMAND_INJECTION_PATTERNS,
        XSS_PATTERNS, AUTH_PATTERNS, CRYPTO_PATTERNS
    )
    
    checks = []
    
    for pattern in SQL_INJECTION_PATTERNS:
        checks.append({
            "name": pattern.name,
            "category": "SQL Injection",
            "cwe_id": pattern.cwe_id,
            "severity": pattern.severity.value,
        })
    
    for pattern in COMMAND_INJECTION_PATTERNS:
        checks.append({
            "name": pattern.name,
            "category": "Command Injection",
            "cwe_id": pattern.cwe_id,
            "severity": pattern.severity.value,
        })
    
    for pattern in XSS_PATTERNS:
        checks.append({
            "name": pattern.name,
            "category": "XSS",
            "cwe_id": pattern.cwe_id,
            "severity": pattern.severity.value,
        })
    
    for pattern in AUTH_PATTERNS:
        checks.append({
            "name": pattern.name,
            "category": "Authentication",
            "cwe_id": pattern.cwe_id,
            "severity": pattern.severity.value,
        })
    
    return {"checks": checks, "total": len(checks)}


@app.get("/api/pipeline")
async def get_pipeline():
    """Get pipeline architecture diagram."""
    from code_review import get_pipeline_diagram
    return {"diagram": get_pipeline_diagram()}


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
