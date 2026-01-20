"""
Legal Document Analyzer - Data Models

Models for:
- Document types and classification
- Key terms and clauses
- Risk assessment
- Compliance checking
- Analysis reports
"""

from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class DocumentType(Enum):
    """Supported legal document types."""
    NDA = "NDA"
    EMPLOYMENT = "Employment Contract"
    SAAS = "SaaS Agreement"
    LEASE = "Lease Agreement"
    PURCHASE = "Purchase Agreement"
    TOS = "Terms of Service"
    PRIVACY = "Privacy Policy"
    PARTNERSHIP = "Partnership Agreement"
    IP_ASSIGNMENT = "IP Assignment"
    CONSULTING = "Consulting Agreement"
    UNKNOWN = "Unknown"


class RiskLevel(Enum):
    """Risk severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ComplianceFramework(Enum):
    """Compliance frameworks to check."""
    GDPR = "GDPR"
    CCPA = "CCPA"
    SOX = "SOX"
    HIPAA = "HIPAA"
    PCI_DSS = "PCI-DSS"
    SOC2 = "SOC 2"


class AnalysisState(Enum):
    """State machine states."""
    UPLOAD = "upload"
    PARSE = "parse"
    CLASSIFY = "classify"
    ANALYZE = "analyze"
    RISK_ASSESS = "risk_assess"
    COMPLIANCE_CHECK = "compliance_check"
    COMPARE = "compare"
    REPORT = "report"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class Party:
    """A party to the contract."""
    name: str
    role: str  # e.g., "Provider", "Customer", "Employer", "Employee"
    address: str = ""
    jurisdiction: str = ""


@dataclass
class KeyDate:
    """Important dates in the document."""
    date_type: str  # effective_date, termination_date, renewal_date, etc.
    date_value: str
    description: str = ""


@dataclass
class Obligation:
    """A contractual obligation."""
    party: str  # Who has the obligation
    description: str
    section: str = ""  # Section reference
    deadline: str = ""
    is_recurring: bool = False


@dataclass
class FinancialTerm:
    """Financial terms in the contract."""
    term_type: str  # payment, fee, penalty, cap, etc.
    amount: str
    currency: str = "USD"
    frequency: str = ""  # one-time, monthly, annual
    conditions: str = ""


@dataclass
class Clause:
    """A contract clause."""
    clause_id: str
    title: str
    section: str
    content: str
    
    # Analysis
    is_standard: bool = True
    is_favorable: Optional[bool] = None  # True = favorable to client
    notes: str = ""


@dataclass
class ExtractedTerms:
    """All extracted terms from a document."""
    parties: list[Party] = field(default_factory=list)
    dates: list[KeyDate] = field(default_factory=list)
    obligations: list[Obligation] = field(default_factory=list)
    financial_terms: list[FinancialTerm] = field(default_factory=list)
    clauses: list[Clause] = field(default_factory=list)
    
    # Specific common terms
    term_length: str = ""
    renewal_terms: str = ""
    termination_notice: str = ""
    governing_law: str = ""
    dispute_resolution: str = ""
    confidentiality_period: str = ""
    liability_cap: str = ""
    indemnification: str = ""
    ip_ownership: str = ""
    
    def to_dict(self) -> dict:
        return {
            "parties": [{"name": p.name, "role": p.role} for p in self.parties],
            "term_length": self.term_length,
            "governing_law": self.governing_law,
            "liability_cap": self.liability_cap,
            "key_dates": [{"type": d.date_type, "value": d.date_value} for d in self.dates],
        }


@dataclass
class Risk:
    """An identified risk in the document."""
    risk_id: str
    title: str
    description: str
    level: RiskLevel
    
    # Location
    section: str = ""
    clause_reference: str = ""
    
    # Impact
    impact: str = ""
    likelihood: str = ""
    
    # Recommendation
    recommendation: str = ""
    suggested_language: str = ""
    
    def to_dict(self) -> dict:
        return {
            "risk_id": self.risk_id,
            "title": self.title,
            "level": self.level.value,
            "description": self.description,
            "recommendation": self.recommendation,
        }
    
    @property
    def level_emoji(self) -> str:
        emojis = {
            RiskLevel.CRITICAL: "🔴",
            RiskLevel.HIGH: "🟠",
            RiskLevel.MEDIUM: "🟡",
            RiskLevel.LOW: "🟢",
            RiskLevel.INFO: "🔵",
        }
        return emojis.get(self.level, "⚪")


@dataclass
class RiskAssessment:
    """Complete risk assessment."""
    risks: list[Risk] = field(default_factory=list)
    
    # Summary
    overall_risk_level: RiskLevel = RiskLevel.LOW
    risk_score: int = 0  # 0-100
    
    # Counts by level
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    
    def calculate_summary(self):
        self.critical_count = len([r for r in self.risks if r.level == RiskLevel.CRITICAL])
        self.high_count = len([r for r in self.risks if r.level == RiskLevel.HIGH])
        self.medium_count = len([r for r in self.risks if r.level == RiskLevel.MEDIUM])
        self.low_count = len([r for r in self.risks if r.level == RiskLevel.LOW])
        
        # Calculate score (higher = more risky)
        self.risk_score = min(100, 
            self.critical_count * 25 + 
            self.high_count * 15 + 
            self.medium_count * 5 + 
            self.low_count * 1
        )
        
        # Determine overall level
        if self.critical_count > 0:
            self.overall_risk_level = RiskLevel.CRITICAL
        elif self.high_count > 0:
            self.overall_risk_level = RiskLevel.HIGH
        elif self.medium_count > 0:
            self.overall_risk_level = RiskLevel.MEDIUM
        else:
            self.overall_risk_level = RiskLevel.LOW
    
    def to_dict(self) -> dict:
        return {
            "overall_level": self.overall_risk_level.value,
            "risk_score": self.risk_score,
            "critical": self.critical_count,
            "high": self.high_count,
            "medium": self.medium_count,
            "low": self.low_count,
            "risks": [r.to_dict() for r in self.risks],
        }


@dataclass
class ComplianceCheck:
    """A compliance check result."""
    framework: ComplianceFramework
    requirement: str
    status: str  # "compliant", "non_compliant", "partial", "not_applicable"
    details: str = ""
    section_reference: str = ""
    recommendation: str = ""
    
    @property
    def status_emoji(self) -> str:
        emojis = {
            "compliant": "✅",
            "non_compliant": "❌",
            "partial": "⚠️",
            "not_applicable": "➖",
        }
        return emojis.get(self.status, "❓")


@dataclass
class ComplianceReport:
    """Complete compliance report."""
    checks: list[ComplianceCheck] = field(default_factory=list)
    
    # Summary by framework
    framework_status: dict = field(default_factory=dict)  # {framework: status}
    
    # Counts
    compliant_count: int = 0
    non_compliant_count: int = 0
    partial_count: int = 0
    
    def calculate_summary(self):
        self.compliant_count = len([c for c in self.checks if c.status == "compliant"])
        self.non_compliant_count = len([c for c in self.checks if c.status == "non_compliant"])
        self.partial_count = len([c for c in self.checks if c.status == "partial"])
        
        # Group by framework
        for check in self.checks:
            fw = check.framework.value
            if fw not in self.framework_status:
                self.framework_status[fw] = "compliant"
            
            if check.status == "non_compliant":
                self.framework_status[fw] = "non_compliant"
            elif check.status == "partial" and self.framework_status[fw] != "non_compliant":
                self.framework_status[fw] = "partial"


@dataclass
class ComparisonResult:
    """Result of comparing document to standard."""
    term: str
    document_value: str
    standard_value: str
    status: str  # "match", "better", "worse", "missing"
    notes: str = ""
    
    @property
    def status_emoji(self) -> str:
        emojis = {
            "match": "✓",
            "better": "✓+",
            "worse": "⚠️",
            "missing": "❌",
        }
        return emojis.get(self.status, "?")


@dataclass
class MissingClause:
    """A clause that should be present but is missing."""
    clause_name: str
    importance: str  # critical, recommended, optional
    standard_language: str = ""
    reason: str = ""


@dataclass
class NegotiationSuggestion:
    """A suggested negotiation point."""
    priority: int  # 1 = highest
    issue: str
    current_language: str
    suggested_language: str
    rationale: str
    section_reference: str = ""


@dataclass
class LegalDocument:
    """A legal document being analyzed."""
    document_id: str
    filename: str
    
    # Content
    raw_text: str = ""
    
    # Classification
    document_type: DocumentType = DocumentType.UNKNOWN
    classification_confidence: float = 0.0
    
    # Metadata
    uploaded_at: str = ""
    page_count: int = 0
    word_count: int = 0
    
    # Analysis results
    extracted_terms: Optional[ExtractedTerms] = None
    risk_assessment: Optional[RiskAssessment] = None
    compliance_report: Optional[ComplianceReport] = None
    
    # Comparison
    comparison_results: list[ComparisonResult] = field(default_factory=list)
    missing_clauses: list[MissingClause] = field(default_factory=list)
    
    # Suggestions
    negotiation_suggestions: list[NegotiationSuggestion] = field(default_factory=list)
    
    # Summary
    plain_english_summary: str = ""
    executive_summary: str = ""
    
    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "document_type": self.document_type.value,
            "classification_confidence": self.classification_confidence,
            "word_count": self.word_count,
            "risk_score": self.risk_assessment.risk_score if self.risk_assessment else 0,
        }


@dataclass
class AnalysisReport:
    """Complete analysis report."""
    document: LegalDocument
    
    # Generated at
    generated_at: str = ""
    
    # Sections
    overview: str = ""
    key_terms_table: str = ""
    risk_section: str = ""
    compliance_section: str = ""
    comparison_section: str = ""
    recommendations: str = ""
    
    def to_markdown(self) -> str:
        """Generate full markdown report."""
        doc = self.document
        
        md = f"""# Contract Analysis Report

**Generated:** {self.generated_at}
**Document:** {doc.filename}

---

## Document Overview

- **Type:** {doc.document_type.value}
- **Classification Confidence:** {doc.classification_confidence:.0%}
- **Word Count:** {doc.word_count:,}
"""
        
        if doc.extracted_terms:
            terms = doc.extracted_terms
            
            # Parties
            if terms.parties:
                parties_str = " ↔ ".join([f"{p.name} ({p.role})" for p in terms.parties])
                md += f"- **Parties:** {parties_str}\n"
            
            if terms.term_length:
                md += f"- **Term:** {terms.term_length}\n"
            
            if terms.governing_law:
                md += f"- **Governing Law:** {terms.governing_law}\n"
        
        # Key Terms
        md += "\n## Key Terms Extracted\n\n"
        md += "| Term | Value | Standard? |\n"
        md += "|------|-------|----------|\n"
        
        if doc.extracted_terms:
            terms = doc.extracted_terms
            if terms.liability_cap:
                md += f"| Liability Cap | {terms.liability_cap} | ✓ |\n"
            if terms.termination_notice:
                md += f"| Termination Notice | {terms.termination_notice} | ✓ |\n"
            if terms.confidentiality_period:
                md += f"| Confidentiality | {terms.confidentiality_period} | ✓ |\n"
            for ft in terms.financial_terms[:5]:
                md += f"| {ft.term_type} | {ft.amount} | ✓ |\n"
        
        # Risk Assessment
        if doc.risk_assessment:
            ra = doc.risk_assessment
            md += f"\n## Risk Assessment\n\n"
            md += f"**Overall Risk Level:** {ra.overall_risk_level.value.upper()} (Score: {ra.risk_score}/100)\n\n"
            
            if ra.risks:
                md += "| Risk | Level | Description |\n"
                md += "|------|-------|-------------|\n"
                for risk in ra.risks[:10]:
                    md += f"| {risk.title} | {risk.level_emoji} {risk.level.value.upper()} | {risk.description[:50]}... |\n"
        
        # Compliance
        if doc.compliance_report:
            cr = doc.compliance_report
            md += "\n## Compliance Status\n\n"
            
            for fw, status in cr.framework_status.items():
                emoji = {"compliant": "✅", "non_compliant": "❌", "partial": "⚠️"}.get(status, "❓")
                md += f"- {emoji} **{fw}:** {status.replace('_', ' ').title()}\n"
        
        # Missing Clauses
        if doc.missing_clauses:
            md += "\n## Missing Clauses\n\n"
            for mc in doc.missing_clauses:
                md += f"- **{mc.clause_name}** ({mc.importance}): {mc.reason}\n"
        
        # Negotiation Suggestions
        if doc.negotiation_suggestions:
            md += "\n## Recommended Negotiations\n\n"
            for i, ns in enumerate(doc.negotiation_suggestions[:5], 1):
                md += f"{i}. **{ns.issue}**\n"
                md += f"   - Current: {ns.current_language[:100]}...\n"
                md += f"   - Suggested: {ns.suggested_language[:100]}...\n"
                md += f"   - Rationale: {ns.rationale}\n\n"
        
        # Plain English Summary
        if doc.plain_english_summary:
            md += f"\n## Plain English Summary\n\n{doc.plain_english_summary}\n"
        
        return md


@dataclass
class PipelineState:
    """State for the LangGraph pipeline."""
    # Input
    document: Optional[LegalDocument] = None
    raw_text: str = ""
    
    # Current state
    current_state: AnalysisState = AnalysisState.UPLOAD
    
    # Processing flags
    compare_to_standard: bool = False
    frameworks_to_check: list[ComplianceFramework] = field(default_factory=list)
    
    # Results
    extracted_terms: Optional[ExtractedTerms] = None
    risk_assessment: Optional[RiskAssessment] = None
    compliance_report: Optional[ComplianceReport] = None
    
    # Final output
    report: Optional[AnalysisReport] = None
    
    # Messages for LangGraph
    messages: list = field(default_factory=list)
    
    # Error tracking
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "current_state": self.current_state.value,
            "document_type": self.document.document_type.value if self.document else None,
            "has_terms": self.extracted_terms is not None,
            "has_risks": self.risk_assessment is not None,
            "has_compliance": self.compliance_report is not None,
            "error": self.error,
        }
