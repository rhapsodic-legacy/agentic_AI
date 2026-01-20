"""
Legal Document Analyzer - LangGraph Pipeline

State Machine architecture for legal document analysis.

State Flow:
    ┌─────────────┐
    │   UPLOAD    │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   PARSE     │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │  CLASSIFY   │
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
   NDA          SAAS    ... (type-specific)
    │             │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │    RISK     │
    │  ASSESSOR   │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ COMPLIANCE  │
    │   CHECKER   │
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             │
┌─────────┐       │
│ COMPARE │◄──────┘ (Optional)
└────┬────┘
     │
     ▼
┌─────────────┐
│   REPORT    │
│  GENERATOR  │
└─────────────┘
"""

from typing import Optional, TypedDict, Annotated
from dataclasses import dataclass, field
from datetime import datetime
import operator

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    raise ImportError("Install langgraph: pip install langgraph")

from .models import (
    LegalDocument, PipelineState, AnalysisState, AnalysisReport,
    DocumentType, ComplianceFramework, ExtractedTerms, RiskAssessment,
    ComplianceReport
)
from .nodes import (
    upload_node, parse_node, classify_node, analyze_node,
    risk_assessment_node, compliance_check_node, compare_node, report_node,
    route_after_compliance
)
from .tools import parse_document
from .templates import get_sample_document


@dataclass
class AnalyzerConfig:
    """Configuration for the legal document analyzer."""
    llm_provider: str = "gemini"
    compare_to_standard: bool = True
    frameworks: list[ComplianceFramework] = field(default_factory=lambda: [
        ComplianceFramework.GDPR,
        ComplianceFramework.CCPA,
    ])
    verbose: bool = True


class LegalDocumentAnalyzer:
    """
    Legal Document Analyzer using LangGraph State Machine.
    
    Analyzes legal documents through a multi-stage pipeline:
    1. Upload and parse document
    2. Classify document type
    3. Extract key terms
    4. Assess risks
    5. Check compliance
    6. Compare to standards (optional)
    7. Generate report
    
    Usage:
        analyzer = LegalDocumentAnalyzer()
        
        # Analyze a document
        report = analyzer.analyze(document_text)
        
        # Get markdown report
        print(report.to_markdown())
    """
    
    def __init__(self, config: Optional[AnalyzerConfig] = None):
        self.config = config or AnalyzerConfig()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        
        # Define state schema
        class GraphState(TypedDict):
            document: Optional[LegalDocument]
            raw_text: str
            current_state: AnalysisState
            compare_to_standard: bool
            frameworks_to_check: list[ComplianceFramework]
            extracted_terms: Optional[ExtractedTerms]
            risk_assessment: Optional[RiskAssessment]
            compliance_report: Optional[ComplianceReport]
            report: Optional[AnalysisReport]
            error: Optional[str]
        
        # Create graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("upload", self._wrap_node(upload_node))
        workflow.add_node("parse", self._wrap_node(parse_node))
        workflow.add_node("classify", self._wrap_node(classify_node))
        workflow.add_node("analyze", self._wrap_node(analyze_node))
        workflow.add_node("risk_assess", self._wrap_node(risk_assessment_node))
        workflow.add_node("compliance_check", self._wrap_node(compliance_check_node))
        workflow.add_node("compare", self._wrap_node(compare_node))
        workflow.add_node("report", self._wrap_node(report_node))
        
        # Add edges (linear flow with conditional compare)
        workflow.set_entry_point("upload")
        workflow.add_edge("upload", "parse")
        workflow.add_edge("parse", "classify")
        workflow.add_edge("classify", "analyze")
        workflow.add_edge("analyze", "risk_assess")
        workflow.add_edge("risk_assess", "compliance_check")
        
        # Conditional edge after compliance
        workflow.add_conditional_edges(
            "compliance_check",
            self._route_after_compliance,
            {
                "compare": "compare",
                "report": "report",
            }
        )
        
        workflow.add_edge("compare", "report")
        workflow.add_edge("report", END)
        
        return workflow.compile()
    
    def _wrap_node(self, node_func):
        """Wrap a node function to work with dict state."""
        def wrapper(state: dict) -> dict:
            # Convert dict to PipelineState
            pipeline_state = PipelineState(
                document=state.get("document"),
                raw_text=state.get("raw_text", ""),
                current_state=state.get("current_state", AnalysisState.UPLOAD),
                compare_to_standard=state.get("compare_to_standard", self.config.compare_to_standard),
                frameworks_to_check=state.get("frameworks_to_check", self.config.frameworks),
                extracted_terms=state.get("extracted_terms"),
                risk_assessment=state.get("risk_assessment"),
                compliance_report=state.get("compliance_report"),
                report=state.get("report"),
                error=state.get("error"),
            )
            
            # Call node
            result = node_func(pipeline_state)
            
            # Convert back to dict
            return {
                "document": result.document,
                "raw_text": result.raw_text,
                "current_state": result.current_state,
                "compare_to_standard": result.compare_to_standard,
                "frameworks_to_check": result.frameworks_to_check,
                "extracted_terms": result.extracted_terms,
                "risk_assessment": result.risk_assessment,
                "compliance_report": result.compliance_report,
                "report": result.report,
                "error": result.error,
            }
        return wrapper
    
    def _route_after_compliance(self, state: dict) -> str:
        """Route after compliance check."""
        if state.get("compare_to_standard", True):
            return "compare"
        return "report"
    
    def analyze(
        self,
        document_text: str,
        compare_to_standard: bool = None,
        frameworks: list[ComplianceFramework] = None,
    ) -> AnalysisReport:
        """
        Analyze a legal document.
        
        Args:
            document_text: The document text to analyze
            compare_to_standard: Whether to compare to standard templates
            frameworks: Compliance frameworks to check
        
        Returns:
            AnalysisReport with full analysis
        """
        if self.config.verbose:
            print("\n" + "="*60)
            print("⚖️ LEGAL DOCUMENT ANALYZER")
            print("="*60 + "\n")
        
        # Initialize state
        initial_state = {
            "raw_text": document_text,
            "document": None,
            "current_state": AnalysisState.UPLOAD,
            "compare_to_standard": compare_to_standard if compare_to_standard is not None else self.config.compare_to_standard,
            "frameworks_to_check": frameworks or self.config.frameworks,
            "extracted_terms": None,
            "risk_assessment": None,
            "compliance_report": None,
            "report": None,
            "error": None,
        }
        
        # Run the graph
        try:
            result = self.graph.invoke(initial_state)
            
            if result.get("error"):
                print(f"\n❌ Error: {result['error']}")
                return None
            
            report = result.get("report")
            
            if self.config.verbose and report:
                print("\n" + "="*60)
                print("✅ ANALYSIS COMPLETE")
                print("="*60)
                
                doc = report.document
                print(f"\nDocument Type: {doc.document_type.value}")
                print(f"Risk Score: {doc.risk_assessment.risk_score if doc.risk_assessment else 0}/100")
                
                if doc.compliance_report:
                    print(f"Compliance: {doc.compliance_report.compliant_count} compliant, "
                          f"{doc.compliance_report.non_compliant_count} non-compliant")
                
                print(f"Negotiation Points: {len(doc.negotiation_suggestions)}")
            
            return report
            
        except Exception as e:
            print(f"\n❌ Pipeline error: {e}")
            raise
    
    def analyze_file(self, filepath: str) -> AnalysisReport:
        """Analyze a document from a file."""
        with open(filepath, 'r') as f:
            text = f.read()
        return self.analyze(text)
    
    def quick_classify(self, document_text: str) -> tuple[DocumentType, float]:
        """Quick classification without full analysis."""
        from .tools import classify_document
        doc = parse_document(document_text)
        return classify_document(doc)
    
    def get_sample_analysis(self, doc_type: DocumentType = DocumentType.SAAS) -> AnalysisReport:
        """Run analysis on a sample document."""
        sample_text = get_sample_document(doc_type)
        return self.analyze(sample_text)


# =============================================================================
# Convenience Functions
# =============================================================================

def analyze_document(text: str, **kwargs) -> AnalysisReport:
    """
    Quick function to analyze a document.
    
    Args:
        text: Document text
        **kwargs: Additional config options
    
    Returns:
        AnalysisReport
    """
    config = AnalyzerConfig(**{k: v for k, v in kwargs.items() if hasattr(AnalyzerConfig, k)})
    analyzer = LegalDocumentAnalyzer(config)
    return analyzer.analyze(text)


def classify_document_type(text: str) -> tuple[DocumentType, float]:
    """Quick document classification."""
    analyzer = LegalDocumentAnalyzer(AnalyzerConfig(verbose=False))
    return analyzer.quick_classify(text)


def analyze_sample(doc_type: str = "saas") -> AnalysisReport:
    """Analyze a sample document."""
    type_map = {
        "saas": DocumentType.SAAS,
        "nda": DocumentType.NDA,
        "employment": DocumentType.EMPLOYMENT,
    }
    analyzer = LegalDocumentAnalyzer()
    return analyzer.get_sample_analysis(type_map.get(doc_type.lower(), DocumentType.SAAS))
