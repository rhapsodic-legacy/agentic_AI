"""
Legal Document Analyzer - LangGraph Nodes

State machine nodes for document analysis pipeline:
- Upload → Parse → Classify → Analyze → Risk → Compliance → Compare → Report
"""

from typing import Optional, Literal
from datetime import datetime
import os

from ..models import (
    LegalDocument, PipelineState, AnalysisState, DocumentType,
    ExtractedTerms, RiskAssessment, ComplianceReport, AnalysisReport,
    ComplianceFramework
)
from ..tools import (
    parse_document, classify_document, extract_terms,
    assess_risks, check_compliance, compare_to_standard,
    find_missing_clauses, generate_negotiation_suggestions,
    generate_plain_english_summary
)


def create_llm(provider: str = "gemini"):
    """Create LLM for enhanced analysis."""
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0.3,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
        )
    return None


# =============================================================================
# State Machine Nodes
# =============================================================================

def upload_node(state: PipelineState) -> PipelineState:
    """
    Node: Upload and initialize document.
    
    Transitions: upload → parse
    """
    print("📄 Processing document upload...")
    
    if not state.raw_text:
        state.error = "No document text provided"
        state.current_state = AnalysisState.ERROR
        return state
    
    # Create document object
    state.document = parse_document(state.raw_text, "uploaded_document.txt")
    state.current_state = AnalysisState.PARSE
    
    print(f"   ✓ Document loaded: {state.document.word_count:,} words")
    
    return state


def parse_node(state: PipelineState) -> PipelineState:
    """
    Node: Parse document structure.
    
    Transitions: parse → classify
    """
    print("📝 Parsing document structure...")
    
    if not state.document:
        state.error = "No document to parse"
        state.current_state = AnalysisState.ERROR
        return state
    
    # Basic parsing is done in upload
    state.current_state = AnalysisState.CLASSIFY
    
    print(f"   ✓ Document parsed: {state.document.page_count} pages")
    
    return state


def classify_node(state: PipelineState) -> PipelineState:
    """
    Node: Classify document type.
    
    Transitions: classify → analyze (appropriate analyzer)
    """
    print("🏷️ Classifying document type...")
    
    doc = state.document
    if not doc:
        state.error = "No document to classify"
        state.current_state = AnalysisState.ERROR
        return state
    
    doc_type, confidence = classify_document(doc)
    doc.document_type = doc_type
    doc.classification_confidence = confidence
    
    print(f"   ✓ Type: {doc_type.value} (confidence: {confidence:.0%})")
    
    state.current_state = AnalysisState.ANALYZE
    
    return state


def analyze_node(state: PipelineState) -> PipelineState:
    """
    Node: Type-specific analysis and term extraction.
    
    Routes based on document type, then transitions to risk assessment.
    """
    print("🔍 Extracting key terms...")
    
    doc = state.document
    if not doc:
        state.error = "No document to analyze"
        state.current_state = AnalysisState.ERROR
        return state
    
    # Extract terms
    terms = extract_terms(doc)
    doc.extracted_terms = terms
    state.extracted_terms = terms
    
    # Document-type specific analysis
    if doc.document_type == DocumentType.NDA:
        print("   → NDA-specific analysis")
        _analyze_nda(doc, terms)
    elif doc.document_type == DocumentType.SAAS:
        print("   → SaaS Agreement-specific analysis")
        _analyze_saas(doc, terms)
    elif doc.document_type == DocumentType.EMPLOYMENT:
        print("   → Employment Contract-specific analysis")
        _analyze_employment(doc, terms)
    else:
        print(f"   → General analysis for {doc.document_type.value}")
    
    # Summary
    party_count = len(terms.parties)
    date_count = len(terms.dates)
    financial_count = len(terms.financial_terms)
    print(f"   ✓ Extracted: {party_count} parties, {date_count} dates, {financial_count} financial terms")
    
    state.current_state = AnalysisState.RISK_ASSESS
    
    return state


def _analyze_nda(doc: LegalDocument, terms: ExtractedTerms):
    """NDA-specific analysis."""
    text_lower = doc.raw_text.lower()
    
    # Check for mutual vs one-way
    if "mutual" in text_lower:
        terms.confidentiality_period = terms.confidentiality_period or "Mutual NDA"
    
    # Check for permitted disclosures
    if "employees" in text_lower and "need to know" in text_lower:
        pass  # Standard permitted disclosure


def _analyze_saas(doc: LegalDocument, terms: ExtractedTerms):
    """SaaS agreement-specific analysis."""
    text_lower = doc.raw_text.lower()
    
    # Check for SLA details
    if "soc 2" in text_lower or "soc2" in text_lower:
        terms.ip_ownership = terms.ip_ownership or "SOC 2 certified"


def _analyze_employment(doc: LegalDocument, terms: ExtractedTerms):
    """Employment contract-specific analysis."""
    text_lower = doc.raw_text.lower()
    
    # Check for at-will
    if "at-will" in text_lower or "at will" in text_lower:
        terms.termination_notice = terms.termination_notice or "At-will employment"


def risk_assessment_node(state: PipelineState) -> PipelineState:
    """
    Node: Assess risks in the document.
    
    Transitions: risk_assess → compliance_check
    """
    print("⚠️ Assessing risks...")
    
    doc = state.document
    terms = state.extracted_terms or ExtractedTerms()
    
    if not doc:
        state.error = "No document for risk assessment"
        state.current_state = AnalysisState.ERROR
        return state
    
    # Perform risk assessment
    assessment = assess_risks(doc, terms)
    doc.risk_assessment = assessment
    state.risk_assessment = assessment
    
    # Summary
    print(f"   ✓ Overall risk: {assessment.overall_risk_level.value.upper()} (score: {assessment.risk_score}/100)")
    print(f"   ✓ Found: {assessment.critical_count} critical, {assessment.high_count} high, {assessment.medium_count} medium risks")
    
    state.current_state = AnalysisState.COMPLIANCE_CHECK
    
    return state


def compliance_check_node(state: PipelineState) -> PipelineState:
    """
    Node: Check compliance with regulatory frameworks.
    
    Transitions: compliance_check → compare (if enabled) or report
    """
    print("✅ Checking compliance...")
    
    doc = state.document
    if not doc:
        state.error = "No document for compliance check"
        state.current_state = AnalysisState.ERROR
        return state
    
    # Determine frameworks to check
    frameworks = state.frameworks_to_check
    if not frameworks:
        # Default frameworks based on document type
        frameworks = [ComplianceFramework.GDPR, ComplianceFramework.CCPA]
        if doc.document_type == DocumentType.SAAS:
            frameworks.append(ComplianceFramework.SOC2)
    
    # Check compliance
    report = check_compliance(doc, frameworks)
    doc.compliance_report = report
    state.compliance_report = report
    
    # Summary
    print(f"   ✓ Checked {len(frameworks)} frameworks")
    for fw, status in report.framework_status.items():
        emoji = {"compliant": "✅", "non_compliant": "❌", "partial": "⚠️"}.get(status, "❓")
        print(f"   {emoji} {fw}: {status.replace('_', ' ')}")
    
    # Next state
    if state.compare_to_standard:
        state.current_state = AnalysisState.COMPARE
    else:
        state.current_state = AnalysisState.REPORT
    
    return state


def compare_node(state: PipelineState) -> PipelineState:
    """
    Node: Compare document to standard templates.
    
    Transitions: compare → report
    """
    print("⚖️ Comparing to standard...")
    
    doc = state.document
    terms = state.extracted_terms or ExtractedTerms()
    
    if not doc:
        state.error = "No document for comparison"
        state.current_state = AnalysisState.ERROR
        return state
    
    # Compare to standard
    comparisons = compare_to_standard(doc, terms)
    doc.comparison_results = comparisons
    
    # Find missing clauses
    missing = find_missing_clauses(doc)
    doc.missing_clauses = missing
    
    # Summary
    better = len([c for c in comparisons if c.status == "better"])
    worse = len([c for c in comparisons if c.status == "worse"])
    print(f"   ✓ Comparisons: {better} better, {worse} worse than standard")
    print(f"   ✓ Missing clauses: {len(missing)}")
    
    state.current_state = AnalysisState.REPORT
    
    return state


def report_node(state: PipelineState) -> PipelineState:
    """
    Node: Generate final analysis report.
    
    Transitions: report → complete
    """
    print("📊 Generating report...")
    
    doc = state.document
    if not doc:
        state.error = "No document for report generation"
        state.current_state = AnalysisState.ERROR
        return state
    
    terms = state.extracted_terms or ExtractedTerms()
    risks = state.risk_assessment or RiskAssessment()
    
    # Generate negotiation suggestions
    comparisons = doc.comparison_results or []
    suggestions = generate_negotiation_suggestions(doc, risks, comparisons)
    doc.negotiation_suggestions = suggestions
    
    # Generate plain English summary
    doc.plain_english_summary = generate_plain_english_summary(doc, terms)
    
    # Create report
    report = AnalysisReport(
        document=doc,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    state.report = report
    
    print(f"   ✓ Report generated")
    print(f"   ✓ {len(suggestions)} negotiation suggestions")
    
    state.current_state = AnalysisState.COMPLETE
    
    return state


# =============================================================================
# Router Functions
# =============================================================================

def route_after_classify(state: PipelineState) -> Literal["analyze", "error"]:
    """Route after classification based on document type."""
    if state.error:
        return "error"
    if state.document and state.document.document_type != DocumentType.UNKNOWN:
        return "analyze"
    return "error"


def route_after_compliance(state: PipelineState) -> Literal["compare", "report"]:
    """Route after compliance - to compare or directly to report."""
    if state.compare_to_standard:
        return "compare"
    return "report"


def should_continue(state: PipelineState) -> bool:
    """Check if pipeline should continue."""
    return state.current_state not in [AnalysisState.COMPLETE, AnalysisState.ERROR]


# =============================================================================
# Node Registry
# =============================================================================

NODES = {
    "upload": upload_node,
    "parse": parse_node,
    "classify": classify_node,
    "analyze": analyze_node,
    "risk_assess": risk_assessment_node,
    "compliance_check": compliance_check_node,
    "compare": compare_node,
    "report": report_node,
}


def get_node(name: str):
    """Get a node function by name."""
    return NODES.get(name)
