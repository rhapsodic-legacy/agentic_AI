"""
Legal Document Analyzer - Analysis Tools

Tools for:
- Document parsing and classification
- Key term extraction
- Risk assessment
- Compliance checking
- Contract comparison
"""

from typing import Optional
import re
import uuid
from datetime import datetime

from ..models import (
    LegalDocument, DocumentType, ExtractedTerms, Party, KeyDate, Obligation,
    FinancialTerm, Clause, Risk, RiskLevel, RiskAssessment,
    ComplianceCheck, ComplianceFramework, ComplianceReport,
    ComparisonResult, MissingClause, NegotiationSuggestion
)
from ..templates import (
    STANDARD_TERMS, REQUIRED_CLAUSES, COMPLIANCE_REQUIREMENTS, RISK_PATTERNS,
    get_standard_terms, get_required_clauses, get_compliance_requirements
)


# =============================================================================
# Document Parsing Tools
# =============================================================================

def parse_document(raw_text: str, filename: str = "document.txt") -> LegalDocument:
    """
    Parse a raw document into structured format.
    
    Args:
        raw_text: The document text
        filename: Original filename
    
    Returns:
        LegalDocument object
    """
    doc = LegalDocument(
        document_id=f"doc-{uuid.uuid4().hex[:8]}",
        filename=filename,
        raw_text=raw_text,
        uploaded_at=datetime.now().isoformat(),
        word_count=len(raw_text.split()),
    )
    
    # Estimate page count (roughly 500 words per page)
    doc.page_count = max(1, doc.word_count // 500)
    
    return doc


def classify_document(doc: LegalDocument) -> tuple[DocumentType, float]:
    """
    Classify document type based on content.
    
    Args:
        doc: Document to classify
    
    Returns:
        Tuple of (DocumentType, confidence)
    """
    text_lower = doc.raw_text.lower()
    
    # Classification patterns
    patterns = {
        DocumentType.NDA: [
            "non-disclosure agreement", "nda", "confidential information",
            "receiving party", "disclosing party", "confidentiality agreement"
        ],
        DocumentType.SAAS: [
            "software as a service", "saas", "subscription", "service level",
            "uptime", "cloud service", "api access"
        ],
        DocumentType.EMPLOYMENT: [
            "employment agreement", "employee", "employer", "salary",
            "at-will employment", "compensation", "job duties"
        ],
        DocumentType.LEASE: [
            "lease agreement", "landlord", "tenant", "rent",
            "premises", "security deposit", "leasehold"
        ],
        DocumentType.CONSULTING: [
            "consulting agreement", "consultant", "services agreement",
            "independent contractor", "statement of work"
        ],
        DocumentType.PURCHASE: [
            "purchase agreement", "buyer", "seller", "purchase price",
            "sale of goods", "delivery terms"
        ],
        DocumentType.TOS: [
            "terms of service", "terms and conditions", "acceptable use",
            "user agreement", "service terms"
        ],
        DocumentType.PRIVACY: [
            "privacy policy", "personal data", "data collection",
            "cookies", "gdpr", "data processing"
        ],
        DocumentType.PARTNERSHIP: [
            "partnership agreement", "partners", "profit sharing",
            "joint venture", "partnership interest"
        ],
        DocumentType.IP_ASSIGNMENT: [
            "ip assignment", "intellectual property assignment",
            "patent assignment", "copyright assignment"
        ],
    }
    
    scores = {}
    for doc_type, keywords in patterns.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[doc_type] = score
    
    if not scores:
        return DocumentType.UNKNOWN, 0.0
    
    best_type = max(scores, key=scores.get)
    max_score = scores[best_type]
    confidence = min(0.95, max_score / len(patterns[best_type]) * 1.2)
    
    return best_type, confidence


# =============================================================================
# Term Extraction Tools
# =============================================================================

def extract_terms(doc: LegalDocument) -> ExtractedTerms:
    """
    Extract key terms from a legal document.
    
    Args:
        doc: Document to analyze
    
    Returns:
        ExtractedTerms with all extracted information
    """
    text = doc.raw_text
    text_lower = text.lower()
    terms = ExtractedTerms()
    
    # Extract parties
    terms.parties = extract_parties(text)
    
    # Extract dates
    terms.dates = extract_dates(text)
    
    # Extract financial terms
    terms.financial_terms = extract_financial_terms(text)
    
    # Extract common terms
    terms.term_length = extract_term_length(text_lower)
    terms.renewal_terms = extract_renewal_terms(text_lower)
    terms.termination_notice = extract_termination_notice(text_lower)
    terms.governing_law = extract_governing_law(text_lower)
    terms.liability_cap = extract_liability_cap(text_lower)
    terms.confidentiality_period = extract_confidentiality_period(text_lower)
    
    return terms


def extract_parties(text: str) -> list[Party]:
    """Extract parties from contract."""
    parties = []
    
    # Common patterns
    patterns = [
        r'between[:\s]+([A-Z][A-Za-z\s,\.]+(?:Inc|LLC|Corp|Ltd)\.?)[,\s]+(?:a\s+\w+\s+\w+\s+)?\("(\w+)"\)',
        r'([A-Z][A-Za-z\s]+(?:Inc|LLC|Corp|Ltd)\.?)\s*\("(\w+)"\)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match) >= 2:
                parties.append(Party(
                    name=match[0].strip(),
                    role=match[1].strip(),
                ))
    
    # Deduplicate
    seen = set()
    unique_parties = []
    for p in parties:
        if p.name not in seen:
            seen.add(p.name)
            unique_parties.append(p)
    
    return unique_parties[:5]  # Limit to 5 parties


def extract_dates(text: str) -> list[KeyDate]:
    """Extract important dates from contract."""
    dates = []
    
    # Effective date
    effective_match = re.search(
        r'(?:effective|as of|dated)[:\s]+(\w+\s+\d{1,2},?\s+\d{4})',
        text, re.IGNORECASE
    )
    if effective_match:
        dates.append(KeyDate(
            date_type="effective_date",
            date_value=effective_match.group(1),
            description="Agreement effective date"
        ))
    
    # Term/expiration patterns
    term_match = re.search(
        r'(?:term|period)[^\n]*?(\d+)\s*(?:month|year)s?',
        text, re.IGNORECASE
    )
    if term_match:
        dates.append(KeyDate(
            date_type="term_length",
            date_value=term_match.group(0),
            description="Contract term"
        ))
    
    return dates


def extract_financial_terms(text: str) -> list[FinancialTerm]:
    """Extract financial terms from contract."""
    terms = []
    
    # Currency amounts
    amount_pattern = r'\$[\d,]+(?:\.\d{2})?(?:\s+(?:per\s+)?(?:month|year|annual)(?:ly)?)?'
    matches = re.findall(amount_pattern, text)
    
    for match in matches[:5]:
        frequency = "one-time"
        if "month" in match.lower():
            frequency = "monthly"
        elif "year" in match.lower() or "annual" in match.lower():
            frequency = "annually"
        
        terms.append(FinancialTerm(
            term_type="payment",
            amount=match.split()[0],
            frequency=frequency,
        ))
    
    return terms


def extract_term_length(text: str) -> str:
    """Extract contract term length."""
    patterns = [
        r'(?:initial\s+)?term\s+(?:is|of|shall\s+be)\s+(\d+)\s*(month|year)s?',
        r'(\d+)\s*(?:-|–)?\s*(month|year)\s+term',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return f"{match.group(1)} {match.group(2)}s"
    
    return ""


def extract_renewal_terms(text: str) -> str:
    """Extract renewal terms."""
    if "automatically renew" in text or "auto-renewal" in text:
        notice_match = re.search(r'(\d+)\s*days?\s*(?:written\s*)?notice', text)
        if notice_match:
            return f"Auto-renews with {notice_match.group(1)} days notice"
        return "Auto-renewal"
    return ""


def extract_termination_notice(text: str) -> str:
    """Extract termination notice period."""
    match = re.search(
        r'(?:termination|terminate)[^\n]*?(\d+)\s*days?\s*(?:written\s*)?notice',
        text, re.IGNORECASE
    )
    if match:
        return f"{match.group(1)} days notice"
    return ""


def extract_governing_law(text: str) -> str:
    """Extract governing law."""
    match = re.search(
        r'governed\s+by\s+(?:the\s+)?(?:laws\s+of\s+)?(?:the\s+)?(?:state\s+of\s+)?(\w+(?:\s+\w+)?)\s+law',
        text, re.IGNORECASE
    )
    if match:
        return match.group(1).title()
    return ""


def extract_liability_cap(text: str) -> str:
    """Extract liability cap."""
    patterns = [
        r'liability\s+shall\s+not\s+exceed\s+([^\.\n]+)',
        r'(?:total|aggregate)\s+liability[^\n]*?(?:limited\s+to|not\s+exceed)\s+([^\.\n]+)',
        r'cap\s+(?:on\s+)?liability[^\n]*?(\$[\d,]+|\d+\s+months?\s+(?:of\s+)?fees?)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return ""


def extract_confidentiality_period(text: str) -> str:
    """Extract confidentiality period."""
    match = re.search(
        r'confidentiality\s+(?:obligations?\s+)?(?:shall\s+)?surviv[^\n]*?(\d+)\s*years?',
        text, re.IGNORECASE
    )
    if match:
        return f"{match.group(1)} years"
    return ""


# =============================================================================
# Risk Assessment Tools
# =============================================================================

def assess_risks(doc: LegalDocument, terms: ExtractedTerms) -> RiskAssessment:
    """
    Assess risks in a legal document.
    
    Args:
        doc: The document
        terms: Extracted terms
    
    Returns:
        RiskAssessment with identified risks
    """
    assessment = RiskAssessment()
    text_lower = doc.raw_text.lower()
    
    # Check each risk pattern
    for risk_id, pattern_info in RISK_PATTERNS.items():
        patterns = pattern_info["patterns"]
        
        for pattern in patterns:
            if pattern in text_lower:
                risk = Risk(
                    risk_id=f"risk-{uuid.uuid4().hex[:6]}",
                    title=risk_id.replace("_", " ").title(),
                    description=pattern_info["description"],
                    level=RiskLevel(pattern_info["level"]),
                    recommendation=pattern_info["recommendation"],
                )
                assessment.risks.append(risk)
                break  # Only add once per risk type
    
    # Document-type specific risks
    if doc.document_type == DocumentType.SAAS:
        assessment.risks.extend(assess_saas_risks(doc, terms))
    elif doc.document_type == DocumentType.EMPLOYMENT:
        assessment.risks.extend(assess_employment_risks(doc, terms))
    elif doc.document_type == DocumentType.NDA:
        assessment.risks.extend(assess_nda_risks(doc, terms))
    
    # Calculate summary
    assessment.calculate_summary()
    
    return assessment


def assess_saas_risks(doc: LegalDocument, terms: ExtractedTerms) -> list[Risk]:
    """Assess SaaS-specific risks."""
    risks = []
    text_lower = doc.raw_text.lower()
    
    # Check SLA
    if "99.9" not in text_lower and "99.99" not in text_lower:
        if "99.5" in text_lower or "99%" in text_lower:
            risks.append(Risk(
                risk_id=f"risk-sla-{uuid.uuid4().hex[:6]}",
                title="Below Standard SLA",
                description="SLA uptime is below industry standard of 99.9%",
                level=RiskLevel.MEDIUM,
                recommendation="Negotiate SLA to 99.9% with service credits",
            ))
    
    # Check data retention
    retention_match = re.search(r'(?:retain|retention)[^\n]*?(\d+)\s*days?', text_lower)
    if retention_match:
        days = int(retention_match.group(1))
        if days < 60:
            risks.append(Risk(
                risk_id=f"risk-retention-{uuid.uuid4().hex[:6]}",
                title="Short Data Retention",
                description=f"Data retention of {days} days may be insufficient",
                level=RiskLevel.MEDIUM,
                recommendation="Negotiate minimum 90-day retention post-termination",
            ))
    
    # Check support response time
    if "48 hour" in text_lower or "48-hour" in text_lower:
        risks.append(Risk(
            risk_id=f"risk-support-{uuid.uuid4().hex[:6]}",
            title="Slow Support Response",
            description="48-hour support response is slow for critical issues",
            level=RiskLevel.LOW,
            recommendation="Negotiate 24-hour response for critical issues",
        ))
    
    return risks


def assess_employment_risks(doc: LegalDocument, terms: ExtractedTerms) -> list[Risk]:
    """Assess employment contract specific risks."""
    risks = []
    text_lower = doc.raw_text.lower()
    
    # Broad non-compete
    if "worldwide" in text_lower and "non-compete" in text_lower:
        risks.append(Risk(
            risk_id=f"risk-compete-{uuid.uuid4().hex[:6]}",
            title="Overly Broad Non-Compete",
            description="Worldwide non-compete is likely unenforceable and overly restrictive",
            level=RiskLevel.HIGH,
            recommendation="Limit to specific geographic areas where company operates",
        ))
    
    # Long non-compete duration
    nc_match = re.search(r'non-?compete[^\n]*?(\d+)\s*years?', text_lower)
    if nc_match and int(nc_match.group(1)) > 1:
        risks.append(Risk(
            risk_id=f"risk-nc-duration-{uuid.uuid4().hex[:6]}",
            title="Long Non-Compete Duration",
            description=f"{nc_match.group(1)}-year non-compete exceeds typical enforceability",
            level=RiskLevel.HIGH,
            recommendation="Negotiate to 1 year or less",
        ))
    
    # No severance
    if "no severance" in text_lower or ("severance" not in text_lower and "termination" in text_lower):
        risks.append(Risk(
            risk_id=f"risk-sev-{uuid.uuid4().hex[:6]}",
            title="No Severance Provision",
            description="No severance terms specified",
            level=RiskLevel.MEDIUM,
            recommendation="Negotiate severance package for termination without cause",
        ))
    
    return risks


def assess_nda_risks(doc: LegalDocument, terms: ExtractedTerms) -> list[Risk]:
    """Assess NDA specific risks."""
    risks = []
    text_lower = doc.raw_text.lower()
    
    # Very long confidentiality period
    if terms.confidentiality_period:
        match = re.search(r'(\d+)', terms.confidentiality_period)
        if match and int(match.group(1)) > 5:
            risks.append(Risk(
                risk_id=f"risk-conf-{uuid.uuid4().hex[:6]}",
                title="Extended Confidentiality Period",
                description=f"Confidentiality period of {terms.confidentiality_period} is unusually long",
                level=RiskLevel.MEDIUM,
                recommendation="Standard is 2-3 years; negotiate shorter period",
            ))
    
    return risks


# =============================================================================
# Compliance Checking Tools
# =============================================================================

def check_compliance(doc: LegalDocument, frameworks: list[ComplianceFramework]) -> ComplianceReport:
    """
    Check document compliance with regulatory frameworks.
    
    Args:
        doc: Document to check
        frameworks: List of frameworks to check
    
    Returns:
        ComplianceReport with all checks
    """
    report = ComplianceReport()
    text_lower = doc.raw_text.lower()
    
    for framework in frameworks:
        requirements = get_compliance_requirements(framework)
        
        for req in requirements:
            keywords = req.get("keywords", [])
            matches = sum(1 for kw in keywords if kw in text_lower)
            
            if matches >= len(keywords) * 0.6:
                status = "compliant"
            elif matches > 0:
                status = "partial"
            else:
                status = "non_compliant"
            
            check = ComplianceCheck(
                framework=framework,
                requirement=req["requirement"],
                status=status,
                details=req["description"],
                recommendation=f"Ensure {req['requirement'].lower()} is adequately addressed" if status != "compliant" else "",
            )
            report.checks.append(check)
    
    report.calculate_summary()
    return report


# =============================================================================
# Comparison Tools
# =============================================================================

def compare_to_standard(doc: LegalDocument, terms: ExtractedTerms) -> list[ComparisonResult]:
    """
    Compare document terms to industry standards.
    
    Args:
        doc: Document
        terms: Extracted terms
    
    Returns:
        List of comparison results
    """
    results = []
    standards = get_standard_terms(doc.document_type)
    
    if not standards:
        return results
    
    # Compare key terms
    if doc.document_type == DocumentType.SAAS:
        # SLA comparison
        if "sla_uptime" in standards:
            doc_sla = extract_sla_uptime(doc.raw_text)
            results.append(ComparisonResult(
                term="SLA Uptime",
                document_value=doc_sla or "Not specified",
                standard_value=standards["sla_uptime"],
                status="worse" if doc_sla and float(doc_sla.replace("%", "")) < 99.9 else "match",
            ))
        
        # Payment terms
        if "payment_terms" in standards and terms.financial_terms:
            results.append(ComparisonResult(
                term="Payment Terms",
                document_value="Net 45" if "net 45" in doc.raw_text.lower() else "Net 30",
                standard_value=standards["payment_terms"],
                status="match" if "net 30" in doc.raw_text.lower() else "worse",
            ))
        
        # Liability cap
        if terms.liability_cap:
            results.append(ComparisonResult(
                term="Liability Cap",
                document_value=terms.liability_cap,
                standard_value=standards.get("liability_cap", "12 months fees"),
                status="match",
            ))
    
    return results


def extract_sla_uptime(text: str) -> str:
    """Extract SLA uptime percentage."""
    match = re.search(r'(\d{2,3}\.?\d*)\s*%\s*(?:uptime|availability)', text, re.IGNORECASE)
    if match:
        return f"{match.group(1)}%"
    return ""


def find_missing_clauses(doc: LegalDocument) -> list[MissingClause]:
    """
    Find clauses that should be present but are missing.
    
    Args:
        doc: Document to check
    
    Returns:
        List of missing clauses
    """
    missing = []
    required = get_required_clauses(doc.document_type)
    text_lower = doc.raw_text.lower()
    
    for clause_name, importance in required:
        # Check if clause appears to be present
        keywords = clause_name.lower().split()
        found = any(kw in text_lower for kw in keywords)
        
        if not found:
            missing.append(MissingClause(
                clause_name=clause_name,
                importance=importance,
                reason=f"Standard {doc.document_type.value} should include {clause_name}",
            ))
    
    return missing


# =============================================================================
# Negotiation Suggestions
# =============================================================================

def generate_negotiation_suggestions(
    doc: LegalDocument,
    risks: RiskAssessment,
    comparisons: list[ComparisonResult]
) -> list[NegotiationSuggestion]:
    """
    Generate negotiation suggestions based on analysis.
    
    Args:
        doc: Analyzed document
        risks: Risk assessment
        comparisons: Comparison results
    
    Returns:
        Prioritized list of negotiation suggestions
    """
    suggestions = []
    priority = 1
    
    # High/Critical risks first
    for risk in sorted(risks.risks, key=lambda r: r.level.value == "critical", reverse=True):
        if risk.level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            suggestions.append(NegotiationSuggestion(
                priority=priority,
                issue=risk.title,
                current_language=risk.description,
                suggested_language=risk.recommendation,
                rationale=f"Risk level: {risk.level.value.upper()}",
            ))
            priority += 1
    
    # Below-standard comparisons
    for comp in comparisons:
        if comp.status == "worse":
            suggestions.append(NegotiationSuggestion(
                priority=priority,
                issue=f"{comp.term} Below Standard",
                current_language=f"Current: {comp.document_value}",
                suggested_language=f"Request: {comp.standard_value}",
                rationale="Below industry standard",
            ))
            priority += 1
    
    # Medium risks
    for risk in risks.risks:
        if risk.level == RiskLevel.MEDIUM:
            suggestions.append(NegotiationSuggestion(
                priority=priority,
                issue=risk.title,
                current_language=risk.description,
                suggested_language=risk.recommendation,
                rationale=f"Risk level: {risk.level.value.upper()}",
            ))
            priority += 1
    
    return suggestions[:10]  # Top 10 suggestions


# =============================================================================
# Plain English Summary
# =============================================================================

def generate_plain_english_summary(doc: LegalDocument, terms: ExtractedTerms) -> str:
    """
    Generate a plain English summary of the contract.
    
    Args:
        doc: Document
        terms: Extracted terms
    
    Returns:
        Plain English summary
    """
    summary_parts = []
    
    # Parties
    if terms.parties:
        party_desc = " and ".join([f"{p.name} (the {p.role})" for p in terms.parties[:2]])
        summary_parts.append(f"This is a {doc.document_type.value} between {party_desc}.")
    else:
        summary_parts.append(f"This is a {doc.document_type.value}.")
    
    # Term
    if terms.term_length:
        summary_parts.append(f"The agreement lasts for {terms.term_length}.")
    
    # Renewal
    if terms.renewal_terms:
        summary_parts.append(f"Regarding renewal: {terms.renewal_terms}.")
    
    # Termination
    if terms.termination_notice:
        summary_parts.append(f"Either party can terminate with {terms.termination_notice}.")
    
    # Liability
    if terms.liability_cap:
        summary_parts.append(f"Liability is capped at {terms.liability_cap}.")
    
    # Governing law
    if terms.governing_law:
        summary_parts.append(f"This agreement is governed by {terms.governing_law} law.")
    
    return " ".join(summary_parts)
