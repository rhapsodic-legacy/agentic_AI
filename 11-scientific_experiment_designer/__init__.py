"""
Scientific Experiment Designer

An AI system that designs scientific experiments using AutoGen with
nested team architecture (Research Teams within Teams).

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                    PRINCIPAL INVESTIGATOR                       │
│               (Oversees entire research project)                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  HYPOTHESIS     │ │   EXPERIMENT    │ │    ANALYSIS     │
│  TEAM           │ │   DESIGN TEAM   │ │    TEAM         │
│                 │ │                 │ │                 │
│ • Literature    │ │ • Protocol      │ │ • Statistics    │
│   Reviewer      │ │   Designer      │ │   Expert        │
│ • Gap           │ │ • Controls      │ │ • Data          │
│   Identifier    │ │   Specialist    │ │   Visualizer    │
│ • Hypothesis    │ │ • Safety        │ │ • Interpreter   │
│   Generator     │ │   Reviewer      │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘

Usage:
    from experiment_designer import ExperimentDesigner
    
    designer = ExperimentDesigner()
    result = designer.design(
        research_question="Does cognitive training improve memory?",
        field="Psychology",
    )
    print(result.to_protocol_markdown())
"""

__version__ = "1.0.0"

from .models import (
    # Enums
    ExperimentType,
    VariableType,
    MeasurementLevel,
    RiskLevel,
    ResearchField,
    
    # Core models
    Reference,
    ResearchGap,
    Hypothesis,
    Variable,
    Group,
    ProtocolStep,
    InclusionCriteria,
    PowerAnalysis,
    StatisticalTest,
    AnalysisPlan,
    SafetyConsideration,
    StoppingRule,
    SafetyPlan,
    ReproducibilityItem,
    ReproducibilityChecklist,
    ExperimentDesign,
    LiteratureReview,
    ResearchProposal,
)

from .teams import (
    # Configuration
    AgentConfig,
    
    # Teams
    HypothesisTeam,
    ExperimentDesignTeam,
    AnalysisTeam,
    
    # Agents
    PrincipalInvestigator,
    LiteratureReviewer,
    GapIdentifier,
    HypothesisGenerator,
    ProtocolDesigner,
    ControlsSpecialist,
    SafetyReviewer,
    StatisticsExpert,
    DataVisualizer,
    ResultsInterpreter,
)

from .tools import (
    # Literature
    search_literature,
    identify_research_gaps,
    generate_literature_review,
    
    # Hypothesis
    generate_hypotheses,
    
    # Variables
    identify_variables,
    
    # Statistics
    calculate_power_analysis,
    create_statistical_test,
    create_analysis_plan,
    
    # Safety
    create_safety_plan,
    
    # Protocol
    generate_protocol_steps,
    create_reproducibility_checklist,
    
    # Full design
    design_experiment,
)

from .templates import (
    STATISTICAL_TESTS,
    COMMON_CONFOUNDERS,
    SAFETY_TEMPLATES,
    REPRODUCIBILITY_ITEMS,
    PROTOCOL_TEMPLATES,
    EXAMPLE_RESEARCH_QUESTIONS,
    get_statistical_test,
    get_common_confounders,
)


# =============================================================================
# Main Designer Class
# =============================================================================

from dataclasses import dataclass
from typing import Optional
from datetime import datetime
import uuid


@dataclass
class DesignerConfig:
    """Configuration for the experiment designer."""
    llm_provider: str = "gemini"
    default_alpha: float = 0.05
    default_power: float = 0.80
    default_effect_size: float = 0.5
    default_duration_days: int = 56
    verbose: bool = True


class ExperimentDesigner:
    """
    Scientific Experiment Designer
    
    An AI system that designs scientific experiments using nested
    AutoGen teams (Research Teams within Teams).
    
    Example:
        designer = ExperimentDesigner()
        
        result = designer.design(
            research_question="Does intervention X improve outcome Y?",
            field="Medicine",
            experiment_type="RCT",
        )
        
        # Generate protocol document
        print(result["proposal"].to_protocol_markdown())
    """
    
    def __init__(self, config: Optional[DesignerConfig] = None):
        self.config = config or DesignerConfig()
        
        # Initialize Principal Investigator
        agent_config = AgentConfig(
            llm_provider=self.config.llm_provider,
            verbose=self.config.verbose,
        )
        self.pi = PrincipalInvestigator(agent_config)
    
    def design(
        self,
        research_question: str,
        field: str = "Biology",
        experiment_type: str = "RCT",
        effect_size: float = None,
        alpha: float = None,
        power: float = None,
        duration_days: int = None,
    ) -> dict:
        """
        Design a complete scientific experiment.
        
        Args:
            research_question: The research question to address
            field: Scientific field (Biology, Psychology, Medicine, etc.)
            experiment_type: Type of experiment (RCT, Factorial, Cohort, etc.)
            effect_size: Expected effect size (default: 0.5)
            alpha: Significance level (default: 0.05)
            power: Statistical power (default: 0.80)
            duration_days: Experiment duration (default: 56)
        
        Returns:
            Dictionary containing:
            - experiment_design: Complete ExperimentDesign object
            - proposal: ResearchProposal with protocol
            - literature_review: LiteratureReview results
        """
        # Parse field
        field_enum = ResearchField.BIOLOGY
        for f in ResearchField:
            if f.value.lower() == field.lower() or f.name.lower() == field.lower():
                field_enum = f
                break
        
        # Parse experiment type
        exp_type_enum = ExperimentType.RCT
        for et in ExperimentType:
            if et.value.lower() == experiment_type.lower() or et.name.lower() == experiment_type.lower():
                exp_type_enum = et
                break
        
        # Use defaults if not specified
        effect_size = effect_size or self.config.default_effect_size
        alpha = alpha or self.config.default_alpha
        power = power or self.config.default_power
        duration_days = duration_days or self.config.default_duration_days
        
        if self.config.verbose:
            print("\n" + "="*60)
            print("🔬 SCIENTIFIC EXPERIMENT DESIGNER")
            print("="*60)
            print(f"\nResearch Question: {research_question}")
            print(f"Field: {field_enum.value}")
            print(f"Design Type: {exp_type_enum.value}")
            print(f"Effect Size: {effect_size}")
            print(f"Alpha: {alpha}, Power: {power}")
            print()
        
        # Run design process
        result = self.pi.design_experiment(
            research_question=research_question,
            field=field_enum,
            experiment_type=exp_type_enum,
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            duration_days=duration_days,
        )
        
        # Create research proposal
        proposal = ResearchProposal(
            proposal_id=f"prop-{uuid.uuid4().hex[:8]}",
            title=result["experiment_design"].title,
            field=field_enum,
            background=f"This study addresses the question: {research_question}",
            significance="This research will contribute to our understanding of the topic.",
            literature_review=result["literature_review"],
            experiment_design=result["experiment_design"],
            timeline_weeks=duration_days // 7,
            expected_outcomes=[
                "Primary outcome measurement",
                "Statistical comparison between groups",
                "Effect size estimation",
            ],
            created_at=datetime.now().isoformat(),
            principal_investigator="AI-Assisted Design",
        )
        
        return {
            "experiment_design": result["experiment_design"],
            "proposal": proposal,
            "literature_review": result["literature_review"],
            "research_gaps": result["research_gaps"],
            "visualizations": result["visualizations"],
            "interpretation_guidelines": result["interpretation_guidelines"],
        }
    
    def quick_design(self, research_question: str, field: str = "Biology") -> ExperimentDesign:
        """Quick design without full verbosity."""
        old_verbose = self.config.verbose
        self.config.verbose = False
        
        result = self.design(research_question, field)
        
        self.config.verbose = old_verbose
        return result["experiment_design"]
    
    def get_example_questions(self, field: str = None) -> list[str]:
        """Get example research questions."""
        if field:
            for f in ResearchField:
                if f.value.lower() == field.lower() or f.name.lower() == field.lower():
                    return EXAMPLE_RESEARCH_QUESTIONS.get(f, [])
        
        all_questions = []
        for questions in EXAMPLE_RESEARCH_QUESTIONS.values():
            all_questions.extend(questions)
        return all_questions


# Convenience function
def design_experiment(research_question: str, field: str = "Biology", **kwargs) -> dict:
    """
    Quick function to design an experiment.
    
    Example:
        result = design_experiment("Does X affect Y?", field="Psychology")
        print(result["proposal"].to_protocol_markdown())
    """
    designer = ExperimentDesigner()
    return designer.design(research_question, field, **kwargs)


__all__ = [
    # Main class
    "ExperimentDesigner",
    "DesignerConfig",
    "design_experiment",
    
    # Enums
    "ExperimentType",
    "VariableType",
    "MeasurementLevel",
    "RiskLevel",
    "ResearchField",
    
    # Models
    "Reference",
    "ResearchGap",
    "Hypothesis",
    "Variable",
    "Group",
    "ProtocolStep",
    "InclusionCriteria",
    "PowerAnalysis",
    "StatisticalTest",
    "AnalysisPlan",
    "SafetyConsideration",
    "StoppingRule",
    "SafetyPlan",
    "ReproducibilityItem",
    "ReproducibilityChecklist",
    "ExperimentDesign",
    "LiteratureReview",
    "ResearchProposal",
    
    # Teams
    "AgentConfig",
    "HypothesisTeam",
    "ExperimentDesignTeam",
    "AnalysisTeam",
    "PrincipalInvestigator",
    
    # Templates
    "STATISTICAL_TESTS",
    "EXAMPLE_RESEARCH_QUESTIONS",
]
