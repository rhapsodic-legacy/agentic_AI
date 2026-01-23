"""
Scientific Experiment Designer - Data Models

Models for:
- Research questions and hypotheses
- Experimental designs
- Variables and controls
- Protocols and procedures
- Statistical analysis plans
- Safety and ethics
"""

from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid


class ExperimentType(Enum):
    """Types of experimental designs."""
    RCT = "Randomized Controlled Trial"
    QUASI_EXPERIMENTAL = "Quasi-Experimental"
    OBSERVATIONAL = "Observational Study"
    COHORT = "Cohort Study"
    CASE_CONTROL = "Case-Control Study"
    CROSS_SECTIONAL = "Cross-Sectional Study"
    LONGITUDINAL = "Longitudinal Study"
    FACTORIAL = "Factorial Design"
    CROSSOVER = "Crossover Design"
    SINGLE_BLIND = "Single-Blind Trial"
    DOUBLE_BLIND = "Double-Blind Trial"


class VariableType(Enum):
    """Types of variables in experiments."""
    INDEPENDENT = "Independent"
    DEPENDENT = "Dependent"
    CONTROLLED = "Controlled"
    CONFOUNDING = "Confounding"
    MODERATING = "Moderating"
    MEDIATING = "Mediating"


class MeasurementLevel(Enum):
    """Levels of measurement."""
    NOMINAL = "Nominal"
    ORDINAL = "Ordinal"
    INTERVAL = "Interval"
    RATIO = "Ratio"


class RiskLevel(Enum):
    """Risk levels for safety assessment."""
    MINIMAL = "Minimal"
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical"


class ResearchField(Enum):
    """Scientific research fields."""
    BIOLOGY = "Biology"
    CHEMISTRY = "Chemistry"
    PHYSICS = "Physics"
    MEDICINE = "Medicine"
    PSYCHOLOGY = "Psychology"
    NEUROSCIENCE = "Neuroscience"
    ECOLOGY = "Ecology"
    GENETICS = "Genetics"
    PHARMACOLOGY = "Pharmacology"
    COMPUTER_SCIENCE = "Computer Science"
    SOCIAL_SCIENCE = "Social Science"
    MATERIALS_SCIENCE = "Materials Science"


@dataclass
class Reference:
    """A literature reference."""
    ref_id: str
    title: str
    authors: list[str]
    year: int
    journal: str = ""
    doi: str = ""
    abstract: str = ""
    relevance_score: float = 0.0
    key_findings: list[str] = field(default_factory=list)


@dataclass
class ResearchGap:
    """An identified gap in existing research."""
    gap_id: str
    description: str
    supporting_references: list[str] = field(default_factory=list)
    significance: str = ""
    addressable: bool = True


@dataclass
class Hypothesis:
    """A scientific hypothesis."""
    hypothesis_id: str
    
    # Type
    is_null: bool  # H₀ or H₁
    
    # Statement
    statement: str
    
    # Predictions
    predictions: list[str] = field(default_factory=list)
    
    # Testability
    testable: bool = True
    falsifiable: bool = True
    
    # Basis
    rationale: str = ""
    supporting_evidence: list[str] = field(default_factory=list)
    
    # Statistical
    alpha_level: float = 0.05
    effect_direction: str = ""  # "increase", "decrease", "difference"
    
    def to_dict(self) -> dict:
        return {
            "id": self.hypothesis_id,
            "type": "H₀" if self.is_null else "H₁",
            "statement": self.statement,
            "alpha": self.alpha_level,
        }


@dataclass
class Variable:
    """A variable in the experiment."""
    variable_id: str
    name: str
    var_type: VariableType
    
    # Measurement
    measurement_method: str = ""
    measurement_level: MeasurementLevel = MeasurementLevel.RATIO
    unit: str = ""
    
    # Range
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    expected_values: list[str] = field(default_factory=list)
    
    # Operationalization
    operational_definition: str = ""
    
    # For controlled variables
    control_method: str = ""
    
    # For confounding variables
    potential_impact: str = ""
    mitigation_strategy: str = ""
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.var_type.value,
            "measurement": self.measurement_method,
            "unit": self.unit,
        }


@dataclass
class Group:
    """An experimental group."""
    group_id: str
    name: str  # "Treatment", "Control", "Placebo", etc.
    
    # Size
    target_n: int
    actual_n: int = 0
    
    # Treatment
    treatment_description: str = ""
    treatment_dose: str = ""
    treatment_frequency: str = ""
    
    # Assignment
    assignment_method: str = "random"
    
    # Blinding
    blinded: bool = False


@dataclass
class ProtocolStep:
    """A step in the experimental protocol."""
    step_number: int
    title: str
    description: str
    
    # Timing
    day_start: int = 0
    day_end: int = 0
    duration_minutes: int = 0
    
    # Requirements
    materials: list[str] = field(default_factory=list)
    equipment: list[str] = field(default_factory=list)
    personnel: list[str] = field(default_factory=list)
    
    # Data collection
    measurements: list[str] = field(default_factory=list)
    
    # Safety
    safety_precautions: list[str] = field(default_factory=list)
    
    # Notes
    critical_notes: str = ""
    alternatives: str = ""


@dataclass
class InclusionCriteria:
    """Criteria for participant inclusion/exclusion."""
    inclusion: list[str] = field(default_factory=list)
    exclusion: list[str] = field(default_factory=list)


@dataclass
class PowerAnalysis:
    """Statistical power analysis."""
    test_type: str  # "t-test", "ANOVA", "chi-square", etc.
    effect_size: float  # Cohen's d, eta-squared, etc.
    effect_size_type: str = "Cohen's d"
    alpha: float = 0.05
    power: float = 0.80
    
    # Calculated
    required_n_per_group: int = 0
    total_n: int = 0
    
    # Assumptions
    assumptions: list[str] = field(default_factory=list)


@dataclass
class StatisticalTest:
    """A planned statistical test."""
    test_id: str
    name: str  # "Two-sample t-test", "ANOVA", etc.
    
    # Purpose
    purpose: str  # "primary", "secondary", "exploratory"
    
    # Variables
    dependent_variable: str = ""
    independent_variable: str = ""
    covariates: list[str] = field(default_factory=list)
    
    # Parameters
    alpha: float = 0.05
    tails: int = 2  # 1 or 2
    
    # Assumptions
    assumptions: list[str] = field(default_factory=list)
    assumption_tests: list[str] = field(default_factory=list)
    
    # Alternative
    non_parametric_alternative: str = ""


@dataclass
class AnalysisPlan:
    """Complete statistical analysis plan."""
    # Tests
    primary_analysis: StatisticalTest = None
    secondary_analyses: list[StatisticalTest] = field(default_factory=list)
    
    # Effect size
    effect_size_measure: str = "Cohen's d"
    
    # Missing data
    missing_data_strategy: str = "intention-to-treat"
    
    # Multiple comparisons
    correction_method: str = "Bonferroni"
    
    # Software
    software: str = "R / Python (scipy, statsmodels)"
    
    # Interpretation guidelines
    interpretation_guidelines: list[str] = field(default_factory=list)


@dataclass
class SafetyConsideration:
    """A safety consideration."""
    consideration_id: str
    category: str  # "physical", "psychological", "ethical", "environmental"
    description: str
    risk_level: RiskLevel
    
    # Mitigation
    mitigation: str = ""
    monitoring: str = ""
    
    # Reporting
    reporting_procedure: str = ""


@dataclass
class StoppingRule:
    """A rule for stopping the experiment."""
    rule_id: str
    condition: str
    action: str
    threshold: str = ""


@dataclass
class SafetyPlan:
    """Complete safety and ethics plan."""
    # Considerations
    considerations: list[SafetyConsideration] = field(default_factory=list)
    
    # Stopping rules
    stopping_rules: list[StoppingRule] = field(default_factory=list)
    
    # Adverse events
    adverse_event_definition: str = ""
    adverse_event_reporting: str = ""
    
    # Emergency
    emergency_contacts: list[str] = field(default_factory=list)
    emergency_procedures: str = ""
    
    # Ethics
    irb_required: bool = True
    informed_consent_required: bool = True
    data_privacy_measures: list[str] = field(default_factory=list)
    
    # Overall risk
    overall_risk_level: RiskLevel = RiskLevel.LOW


@dataclass
class ReproducibilityItem:
    """An item on the reproducibility checklist."""
    item: str
    category: str  # "materials", "methods", "data", "analysis"
    completed: bool = False
    notes: str = ""


@dataclass
class ReproducibilityChecklist:
    """Checklist for ensuring reproducibility."""
    items: list[ReproducibilityItem] = field(default_factory=list)
    
    # Data
    data_availability: str = ""
    data_format: str = ""
    
    # Code
    code_availability: str = ""
    code_repository: str = ""
    
    # Materials
    materials_list: list[str] = field(default_factory=list)
    reagent_sources: list[str] = field(default_factory=list)


@dataclass
class ExperimentDesign:
    """Complete experimental design."""
    design_id: str
    title: str
    
    # Research context
    field: ResearchField = ResearchField.BIOLOGY
    objective: str = ""
    
    # Design type
    experiment_type: ExperimentType = ExperimentType.RCT
    
    # Hypotheses
    null_hypothesis: Hypothesis = None
    alternative_hypothesis: Hypothesis = None
    
    # Variables
    variables: list[Variable] = field(default_factory=list)
    
    # Groups
    groups: list[Group] = field(default_factory=list)
    
    # Participants/Subjects
    population: str = ""
    sample_size: int = 0
    inclusion_criteria: InclusionCriteria = None
    
    # Timeline
    duration_days: int = 0
    
    # Power analysis
    power_analysis: PowerAnalysis = None
    
    # Protocol
    protocol_steps: list[ProtocolStep] = field(default_factory=list)
    
    # Analysis
    analysis_plan: AnalysisPlan = None
    
    # Safety
    safety_plan: SafetyPlan = None
    
    # Reproducibility
    reproducibility: ReproducibilityChecklist = None
    
    # Metadata
    created_at: str = ""
    created_by: str = ""
    version: str = "1.0"
    
    def get_variables_by_type(self, var_type: VariableType) -> list[Variable]:
        return [v for v in self.variables if v.var_type == var_type]
    
    def to_dict(self) -> dict:
        return {
            "design_id": self.design_id,
            "title": self.title,
            "field": self.field.value,
            "type": self.experiment_type.value,
            "sample_size": self.sample_size,
            "duration_days": self.duration_days,
            "groups": len(self.groups),
            "variables": len(self.variables),
        }


@dataclass
class LiteratureReview:
    """Literature review results."""
    query: str
    field: ResearchField
    
    # References found
    references: list[Reference] = field(default_factory=list)
    
    # Key findings
    key_themes: list[str] = field(default_factory=list)
    consensus_findings: list[str] = field(default_factory=list)
    contradictory_findings: list[str] = field(default_factory=list)
    
    # Gaps
    research_gaps: list[ResearchGap] = field(default_factory=list)
    
    # Methods used
    common_methods: list[str] = field(default_factory=list)
    sample_sizes: list[int] = field(default_factory=list)
    
    # Summary
    summary: str = ""


@dataclass
class ResearchProposal:
    """Complete research proposal."""
    proposal_id: str
    title: str
    
    # Background
    field: ResearchField
    background: str = ""
    significance: str = ""
    
    # Review
    literature_review: LiteratureReview = None
    
    # Design
    experiment_design: ExperimentDesign = None
    
    # Resources
    budget_estimate: float = 0.0
    timeline_weeks: int = 0
    personnel_required: list[str] = field(default_factory=list)
    equipment_required: list[str] = field(default_factory=list)
    
    # Deliverables
    expected_outcomes: list[str] = field(default_factory=list)
    potential_publications: list[str] = field(default_factory=list)
    
    # Metadata
    created_at: str = ""
    principal_investigator: str = ""
    institution: str = ""
    
    def to_protocol_markdown(self) -> str:
        """Generate markdown protocol document."""
        design = self.experiment_design
        if not design:
            return "# Experiment Protocol\n\nNo design available."
        
        md = f"""# Experiment Protocol: {design.title}

**Proposal ID:** {self.proposal_id}
**Field:** {self.field.value}
**Generated:** {datetime.now().strftime("%Y-%m-%d")}

---

## 1. Objective

{design.objective}

## 2. Background

{self.background}

## 3. Hypotheses

"""
        if design.null_hypothesis:
            md += f"**H₀ (Null):** {design.null_hypothesis.statement}\n\n"
        if design.alternative_hypothesis:
            md += f"**H₁ (Alternative):** {design.alternative_hypothesis.statement}\n\n"
            if design.alternative_hypothesis.alpha_level:
                md += f"**Significance Level:** α = {design.alternative_hypothesis.alpha_level}\n\n"
        
        md += f"""## 4. Experimental Design

- **Type:** {design.experiment_type.value}
- **Population:** {design.population}
- **Sample Size:** n = {design.sample_size}
- **Duration:** {design.duration_days} days

### Groups
"""
        for group in design.groups:
            md += f"- **{group.name}** (n={group.target_n}): {group.treatment_description}\n"
        
        md += "\n## 5. Variables\n\n"
        md += "| Type | Variable | Measurement | Unit |\n"
        md += "|------|----------|-------------|------|\n"
        
        for var in design.variables:
            md += f"| {var.var_type.value} | {var.name} | {var.measurement_method} | {var.unit} |\n"
        
        # Confounding variables
        confounding = design.get_variables_by_type(VariableType.CONFOUNDING)
        if confounding:
            md += "\n### Confounding Variables\n"
            for var in confounding:
                md += f"- **{var.name}:** {var.potential_impact}\n"
                md += f"  - Mitigation: {var.mitigation_strategy}\n"
        
        md += "\n## 6. Protocol Steps\n\n"
        for step in design.protocol_steps:
            md += f"### Step {step.step_number}: {step.title}\n\n"
            md += f"{step.description}\n\n"
            if step.day_start or step.day_end:
                md += f"- **Timing:** Day {step.day_start}"
                if step.day_end != step.day_start:
                    md += f" - Day {step.day_end}"
                md += "\n"
            if step.materials:
                md += f"- **Materials:** {', '.join(step.materials)}\n"
            if step.measurements:
                md += f"- **Measurements:** {', '.join(step.measurements)}\n"
            if step.safety_precautions:
                md += f"- **Safety:** {', '.join(step.safety_precautions)}\n"
            md += "\n"
        
        # Statistical Analysis
        if design.analysis_plan:
            ap = design.analysis_plan
            md += "## 7. Statistical Analysis Plan\n\n"
            
            if ap.primary_analysis:
                pa = ap.primary_analysis
                md += f"### Primary Analysis\n"
                md += f"- **Test:** {pa.name}\n"
                md += f"- **α:** {pa.alpha}\n"
                md += f"- **Tails:** {pa.tails}\n"
                if pa.assumptions:
                    md += f"- **Assumptions:** {', '.join(pa.assumptions)}\n"
                md += "\n"
            
            if ap.secondary_analyses:
                md += "### Secondary Analyses\n"
                for sa in ap.secondary_analyses:
                    md += f"- {sa.name}: {sa.purpose}\n"
                md += "\n"
            
            md += f"- **Effect Size:** {ap.effect_size_measure}\n"
            md += f"- **Missing Data:** {ap.missing_data_strategy}\n"
            md += f"- **Multiple Comparisons:** {ap.correction_method}\n"
        
        # Power Analysis
        if design.power_analysis:
            pa = design.power_analysis
            md += f"\n### Power Analysis\n"
            md += f"- **Target Power:** {pa.power:.0%}\n"
            md += f"- **Expected Effect Size:** {pa.effect_size} ({pa.effect_size_type})\n"
            md += f"- **Required N per Group:** {pa.required_n_per_group}\n"
        
        # Safety
        if design.safety_plan:
            sp = design.safety_plan
            md += "\n## 8. Safety Considerations\n\n"
            md += f"**Overall Risk Level:** {sp.overall_risk_level.value}\n\n"
            
            if sp.considerations:
                for sc in sp.considerations:
                    emoji = {"MINIMAL": "🟢", "LOW": "🟢", "MODERATE": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}.get(sc.risk_level.name, "⚪")
                    md += f"- {emoji} **{sc.category.title()}:** {sc.description}\n"
                    if sc.mitigation:
                        md += f"  - Mitigation: {sc.mitigation}\n"
            
            if sp.stopping_rules:
                md += "\n### Stopping Rules\n"
                for rule in sp.stopping_rules:
                    md += f"- **If** {rule.condition} **then** {rule.action}\n"
            
            if sp.emergency_procedures:
                md += f"\n### Emergency Procedures\n{sp.emergency_procedures}\n"
        
        # Reproducibility
        if design.reproducibility:
            rc = design.reproducibility
            md += "\n## 9. Reproducibility Checklist\n\n"
            
            for item in rc.items[:10]:
                checkbox = "☑" if item.completed else "☐"
                md += f"- {checkbox} {item.item}\n"
            
            if rc.data_availability:
                md += f"\n**Data Availability:** {rc.data_availability}\n"
            if rc.code_repository:
                md += f"**Code Repository:** {rc.code_repository}\n"
        
        return md
