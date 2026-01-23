"""
Scientific Experiment Designer - Nested AutoGen Teams

Nested Team Structure:
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
"""

from typing import Optional
from dataclasses import dataclass
import os


@dataclass
class AgentConfig:
    """Configuration for AutoGen agents."""
    llm_provider: str = "gemini"
    verbose: bool = True


def get_llm_config(provider: str = "gemini") -> dict:
    """Get LLM configuration for AutoGen."""
    if provider == "gemini":
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        return {
            "config_list": [{
                "model": "gemini-1.5-flash",
                "api_key": api_key,
                "api_type": "google",
            }],
            "temperature": 0.3,
        }
    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        return {
            "config_list": [{
                "model": "claude-sonnet-4-20250514",
                "api_key": api_key,
                "api_type": "anthropic",
            }],
            "temperature": 0.3,
        }
    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        return {
            "config_list": [{
                "model": "gpt-4o-mini",
                "api_key": api_key,
            }],
            "temperature": 0.3,
        }
    return {}


# =============================================================================
# Base Agent Classes
# =============================================================================

class BaseAgent:
    """Base class for experiment designer agents."""
    
    def __init__(self, name: str, role: str, config: AgentConfig = None):
        self.name = name
        self.role = role
        self.config = config or AgentConfig()
    
    def log(self, message: str):
        """Log agent activity."""
        if self.config.verbose:
            print(f"    [{self.name}] {message}")


class BaseTeam:
    """Base class for nested teams."""
    
    def __init__(self, name: str, config: AgentConfig = None):
        self.name = name
        self.config = config or AgentConfig()
        self.agents = []
    
    def log(self, message: str):
        """Log team activity."""
        if self.config.verbose:
            print(f"  [{self.name}] {message}")
    
    def add_agent(self, agent: BaseAgent):
        """Add an agent to the team."""
        self.agents.append(agent)


# =============================================================================
# Hypothesis Team Agents
# =============================================================================

class LiteratureReviewer(BaseAgent):
    """
    Literature Reviewer Agent
    
    Responsibilities:
    - Search scientific databases
    - Summarize relevant research
    - Identify key findings
    - Track methodologies used
    """
    
    def __init__(self, config: AgentConfig = None):
        super().__init__(
            name="Literature Reviewer",
            role="Search and synthesize existing research",
            config=config,
        )
    
    def review(self, query: str, field) -> dict:
        """Conduct literature review."""
        from ..tools import search_literature, generate_literature_review
        
        self.log(f"Searching literature for: {query}")
        
        review = generate_literature_review(query, field)
        
        self.log(f"✓ Found {len(review.references)} relevant references")
        self.log(f"✓ Identified {len(review.key_themes)} key themes")
        
        return {
            "review": review,
            "summary": review.summary,
            "reference_count": len(review.references),
        }


class GapIdentifier(BaseAgent):
    """
    Gap Identifier Agent
    
    Responsibilities:
    - Analyze literature for gaps
    - Identify unexplored questions
    - Assess gap significance
    - Prioritize research opportunities
    """
    
    def __init__(self, config: AgentConfig = None):
        super().__init__(
            name="Gap Identifier",
            role="Identify research gaps and opportunities",
            config=config,
        )
    
    def identify_gaps(self, literature_review) -> dict:
        """Identify research gaps from literature."""
        self.log("Analyzing literature for research gaps...")
        
        gaps = literature_review.research_gaps
        
        self.log(f"✓ Identified {len(gaps)} research gaps")
        
        return {
            "gaps": gaps,
            "top_gap": gaps[0] if gaps else None,
            "addressable_count": len([g for g in gaps if g.addressable]),
        }


class HypothesisGenerator(BaseAgent):
    """
    Hypothesis Generator Agent
    
    Responsibilities:
    - Formulate testable hypotheses
    - Ensure falsifiability
    - Generate null and alternative
    - Specify predictions
    """
    
    def __init__(self, config: AgentConfig = None):
        super().__init__(
            name="Hypothesis Generator",
            role="Generate testable scientific hypotheses",
            config=config,
        )
    
    def generate(self, research_question: str, field, gaps: list) -> dict:
        """Generate hypotheses."""
        from ..tools import generate_hypotheses
        
        self.log("Generating hypotheses based on research gaps...")
        
        h0, h1 = generate_hypotheses(research_question, field)
        
        self.log(f"✓ Generated null hypothesis: {h0.statement[:50]}...")
        self.log(f"✓ Generated alternative hypothesis: {h1.statement[:50]}...")
        
        return {
            "null_hypothesis": h0,
            "alternative_hypothesis": h1,
            "testable": h1.testable,
            "falsifiable": h1.falsifiable,
        }


class HypothesisTeam(BaseTeam):
    """
    Hypothesis Team - Nested team for hypothesis development.
    
    Contains:
    - Literature Reviewer
    - Gap Identifier
    - Hypothesis Generator
    """
    
    def __init__(self, config: AgentConfig = None):
        super().__init__("Hypothesis Team", config)
        
        self.literature_reviewer = LiteratureReviewer(config)
        self.gap_identifier = GapIdentifier(config)
        self.hypothesis_generator = HypothesisGenerator(config)
        
        self.add_agent(self.literature_reviewer)
        self.add_agent(self.gap_identifier)
        self.add_agent(self.hypothesis_generator)
    
    def develop_hypotheses(self, research_question: str, field) -> dict:
        """Run the hypothesis development pipeline."""
        self.log("Starting hypothesis development...")
        
        # Step 1: Literature Review
        lit_result = self.literature_reviewer.review(research_question, field)
        
        # Step 2: Gap Identification
        gap_result = self.gap_identifier.identify_gaps(lit_result["review"])
        
        # Step 3: Hypothesis Generation
        hyp_result = self.hypothesis_generator.generate(
            research_question, field, gap_result["gaps"]
        )
        
        self.log("✓ Hypothesis development complete")
        
        return {
            "literature_review": lit_result["review"],
            "research_gaps": gap_result["gaps"],
            "null_hypothesis": hyp_result["null_hypothesis"],
            "alternative_hypothesis": hyp_result["alternative_hypothesis"],
        }


# =============================================================================
# Experiment Design Team Agents
# =============================================================================

class ProtocolDesigner(BaseAgent):
    """
    Protocol Designer Agent
    
    Responsibilities:
    - Design experimental procedures
    - Create step-by-step protocols
    - Specify timing and resources
    - Ensure reproducibility
    """
    
    def __init__(self, config: AgentConfig = None):
        super().__init__(
            name="Protocol Designer",
            role="Design detailed experimental protocols",
            config=config,
        )
    
    def design_protocol(self, experiment_type, duration_days, groups) -> dict:
        """Design experimental protocol."""
        from ..tools import generate_protocol_steps
        
        self.log("Designing experimental protocol...")
        
        steps = generate_protocol_steps(experiment_type, duration_days, groups)
        
        self.log(f"✓ Created {len(steps)} protocol steps")
        
        return {
            "protocol_steps": steps,
            "duration_days": duration_days,
            "step_count": len(steps),
        }


class ControlsSpecialist(BaseAgent):
    """
    Controls Specialist Agent
    
    Responsibilities:
    - Identify necessary controls
    - Design control conditions
    - Detect confounding variables
    - Propose mitigation strategies
    """
    
    def __init__(self, config: AgentConfig = None):
        super().__init__(
            name="Controls Specialist",
            role="Design controls and identify confounds",
            config=config,
        )
    
    def design_controls(self, research_question, field, experiment_type) -> dict:
        """Design experimental controls and identify confounds."""
        from ..tools import identify_variables
        from ..models import VariableType
        
        self.log("Identifying variables and designing controls...")
        
        variables = identify_variables(research_question, field, experiment_type)
        
        controlled = [v for v in variables if v.var_type == VariableType.CONTROLLED]
        confounding = [v for v in variables if v.var_type == VariableType.CONFOUNDING]
        
        self.log(f"✓ Identified {len(controlled)} controlled variables")
        self.log(f"✓ Identified {len(confounding)} potential confounds")
        
        return {
            "variables": variables,
            "controlled_variables": controlled,
            "confounding_variables": confounding,
        }


class SafetyReviewer(BaseAgent):
    """
    Safety Reviewer Agent
    
    Responsibilities:
    - Assess research risks
    - Develop safety protocols
    - Define stopping rules
    - Ensure ethical compliance
    """
    
    def __init__(self, config: AgentConfig = None):
        super().__init__(
            name="Safety Reviewer",
            role="Review safety and ethical considerations",
            config=config,
        )
    
    def review_safety(self, field, involves_humans=True) -> dict:
        """Review safety considerations."""
        from ..tools import create_safety_plan
        
        self.log("Reviewing safety and ethical considerations...")
        
        safety_plan = create_safety_plan(field, involves_humans)
        
        self.log(f"✓ Overall risk level: {safety_plan.overall_risk_level.value}")
        self.log(f"✓ Created {len(safety_plan.stopping_rules)} stopping rules")
        
        return {
            "safety_plan": safety_plan,
            "risk_level": safety_plan.overall_risk_level,
            "irb_required": safety_plan.irb_required,
        }


class ExperimentDesignTeam(BaseTeam):
    """
    Experiment Design Team - Nested team for experiment design.
    
    Contains:
    - Protocol Designer
    - Controls Specialist
    - Safety Reviewer
    """
    
    def __init__(self, config: AgentConfig = None):
        super().__init__("Experiment Design Team", config)
        
        self.protocol_designer = ProtocolDesigner(config)
        self.controls_specialist = ControlsSpecialist(config)
        self.safety_reviewer = SafetyReviewer(config)
        
        self.add_agent(self.protocol_designer)
        self.add_agent(self.controls_specialist)
        self.add_agent(self.safety_reviewer)
    
    def design_experiment(
        self,
        research_question: str,
        field,
        experiment_type,
        duration_days: int,
        groups: list,
    ) -> dict:
        """Run the experiment design pipeline."""
        self.log("Starting experiment design...")
        
        # Step 1: Design controls
        controls_result = self.controls_specialist.design_controls(
            research_question, field, experiment_type
        )
        
        # Step 2: Design protocol
        protocol_result = self.protocol_designer.design_protocol(
            experiment_type, duration_days, groups
        )
        
        # Step 3: Safety review
        safety_result = self.safety_reviewer.review_safety(field)
        
        self.log("✓ Experiment design complete")
        
        return {
            "variables": controls_result["variables"],
            "protocol_steps": protocol_result["protocol_steps"],
            "safety_plan": safety_result["safety_plan"],
        }


# =============================================================================
# Analysis Team Agents
# =============================================================================

class StatisticsExpert(BaseAgent):
    """
    Statistics Expert Agent
    
    Responsibilities:
    - Select appropriate tests
    - Calculate power analysis
    - Plan statistical analyses
    - Define significance criteria
    """
    
    def __init__(self, config: AgentConfig = None):
        super().__init__(
            name="Statistics Expert",
            role="Plan statistical analyses",
            config=config,
        )
    
    def plan_analysis(self, experiment_type, effect_size, alpha, power) -> dict:
        """Plan statistical analysis."""
        from ..tools import calculate_power_analysis, create_analysis_plan
        from ..models import ExperimentType
        
        self.log("Planning statistical analysis...")
        
        # Determine appropriate test
        test_type = "t_test_independent"
        if experiment_type == ExperimentType.FACTORIAL:
            test_type = "anova_two_way"
        elif experiment_type == ExperimentType.CROSSOVER:
            test_type = "t_test_paired"
        
        # Power analysis
        power_analysis = calculate_power_analysis(test_type, effect_size, alpha, power)
        
        # Analysis plan
        analysis_plan = create_analysis_plan(test_type, ["ancova", "correlation"])
        
        self.log(f"✓ Primary test: {analysis_plan.primary_analysis.name}")
        self.log(f"✓ Required n per group: {power_analysis.required_n_per_group}")
        
        return {
            "power_analysis": power_analysis,
            "analysis_plan": analysis_plan,
            "test_type": test_type,
        }


class DataVisualizer(BaseAgent):
    """
    Data Visualizer Agent
    
    Responsibilities:
    - Plan data visualizations
    - Suggest appropriate graphs
    - Design figure layouts
    - Ensure clarity
    """
    
    def __init__(self, config: AgentConfig = None):
        super().__init__(
            name="Data Visualizer",
            role="Plan data visualizations",
            config=config,
        )
    
    def plan_visualizations(self, variables, groups) -> dict:
        """Plan data visualizations."""
        self.log("Planning data visualizations...")
        
        visualizations = [
            {
                "name": "Group Comparison",
                "type": "bar chart with error bars",
                "variables": ["Treatment condition", "Primary outcome"],
                "purpose": "Compare means between groups",
            },
            {
                "name": "Distribution Plot",
                "type": "violin plot or histogram",
                "variables": ["Primary outcome by group"],
                "purpose": "Show data distribution",
            },
            {
                "name": "Correlation Matrix",
                "type": "heatmap",
                "variables": ["All continuous variables"],
                "purpose": "Identify relationships",
            },
            {
                "name": "Time Series",
                "type": "line plot",
                "variables": ["Outcome over time by group"],
                "purpose": "Show temporal patterns",
            },
        ]
        
        self.log(f"✓ Planned {len(visualizations)} visualizations")
        
        return {
            "visualizations": visualizations,
            "count": len(visualizations),
        }


class ResultsInterpreter(BaseAgent):
    """
    Results Interpreter Agent
    
    Responsibilities:
    - Provide interpretation guidelines
    - Distinguish statistical vs practical significance
    - Plan result reporting
    - Anticipate limitations
    """
    
    def __init__(self, config: AgentConfig = None):
        super().__init__(
            name="Results Interpreter",
            role="Guide results interpretation",
            config=config,
        )
    
    def create_interpretation_guide(self, analysis_plan) -> dict:
        """Create interpretation guidelines."""
        self.log("Creating interpretation guidelines...")
        
        guidelines = [
            "Report exact p-values, not just significance thresholds",
            "Always report effect sizes with confidence intervals",
            "Distinguish between statistical and practical significance",
            "Consider the clinical/practical meaning of effect sizes",
            "Report all pre-specified analyses, regardless of outcome",
            "Acknowledge limitations of the study design",
            "Discuss alternative interpretations of results",
            "Avoid overstatement of findings",
        ]
        
        limitations = [
            "Generalizability to other populations",
            "Potential unmeasured confounders",
            "Duration of follow-up",
            "Sample size constraints",
        ]
        
        self.log(f"✓ Created {len(guidelines)} interpretation guidelines")
        
        return {
            "interpretation_guidelines": guidelines,
            "anticipated_limitations": limitations,
        }


class AnalysisTeam(BaseTeam):
    """
    Analysis Team - Nested team for analysis planning.
    
    Contains:
    - Statistics Expert
    - Data Visualizer
    - Results Interpreter
    """
    
    def __init__(self, config: AgentConfig = None):
        super().__init__("Analysis Team", config)
        
        self.statistics_expert = StatisticsExpert(config)
        self.data_visualizer = DataVisualizer(config)
        self.results_interpreter = ResultsInterpreter(config)
        
        self.add_agent(self.statistics_expert)
        self.add_agent(self.data_visualizer)
        self.add_agent(self.results_interpreter)
    
    def plan_analysis(
        self,
        experiment_type,
        variables: list,
        groups: list,
        effect_size: float,
        alpha: float,
        power: float,
    ) -> dict:
        """Run the analysis planning pipeline."""
        self.log("Starting analysis planning...")
        
        # Step 1: Statistical planning
        stats_result = self.statistics_expert.plan_analysis(
            experiment_type, effect_size, alpha, power
        )
        
        # Step 2: Visualization planning
        viz_result = self.data_visualizer.plan_visualizations(variables, groups)
        
        # Step 3: Interpretation guidelines
        interp_result = self.results_interpreter.create_interpretation_guide(
            stats_result["analysis_plan"]
        )
        
        self.log("✓ Analysis planning complete")
        
        return {
            "power_analysis": stats_result["power_analysis"],
            "analysis_plan": stats_result["analysis_plan"],
            "visualizations": viz_result["visualizations"],
            "interpretation_guidelines": interp_result["interpretation_guidelines"],
        }


# =============================================================================
# Principal Investigator (Top-Level Agent)
# =============================================================================

class PrincipalInvestigator(BaseAgent):
    """
    Principal Investigator - Oversees entire research project.
    
    Coordinates three nested teams:
    - Hypothesis Team
    - Experiment Design Team
    - Analysis Team
    """
    
    def __init__(self, config: AgentConfig = None):
        super().__init__(
            name="Principal Investigator",
            role="Oversee and coordinate research project",
            config=config,
        )
        
        # Initialize nested teams
        self.hypothesis_team = HypothesisTeam(config)
        self.design_team = ExperimentDesignTeam(config)
        self.analysis_team = AnalysisTeam(config)
    
    def design_experiment(
        self,
        research_question: str,
        field,
        experiment_type,
        effect_size: float = 0.5,
        alpha: float = 0.05,
        power: float = 0.80,
        duration_days: int = 56,
    ) -> dict:
        """
        Coordinate full experiment design process.
        
        Returns complete experimental design with all components.
        """
        from ..tools import create_reproducibility_checklist
        from ..models import Group, ExperimentDesign, InclusionCriteria
        
        self.log("="*50)
        self.log("STARTING EXPERIMENT DESIGN")
        self.log(f"Research Question: {research_question[:60]}...")
        self.log("="*50)
        
        # Phase 1: Hypothesis Development
        self.log("\n📚 PHASE 1: HYPOTHESIS DEVELOPMENT")
        hypothesis_result = self.hypothesis_team.develop_hypotheses(research_question, field)
        
        # Preliminary groups for design
        preliminary_groups = [
            Group(group_id="grp-1", name="Treatment", target_n=30, treatment_description="Active intervention"),
            Group(group_id="grp-2", name="Control", target_n=30, treatment_description="Control condition"),
        ]
        
        # Phase 2: Experiment Design
        self.log("\n🔬 PHASE 2: EXPERIMENT DESIGN")
        design_result = self.design_team.design_experiment(
            research_question, field, experiment_type, duration_days, preliminary_groups
        )
        
        # Phase 3: Analysis Planning
        self.log("\n📊 PHASE 3: ANALYSIS PLANNING")
        analysis_result = self.analysis_team.plan_analysis(
            experiment_type,
            design_result["variables"],
            preliminary_groups,
            effect_size, alpha, power,
        )
        
        # Update groups with power analysis
        n_per_group = analysis_result["power_analysis"].required_n_per_group
        final_groups = [
            Group(group_id="grp-1", name="Treatment", target_n=n_per_group, 
                  treatment_description="Active intervention", assignment_method="random"),
            Group(group_id="grp-2", name="Control", target_n=n_per_group,
                  treatment_description="Control condition", assignment_method="random"),
        ]
        
        # Create reproducibility checklist
        reproducibility = create_reproducibility_checklist()
        
        # Inclusion criteria
        inclusion = InclusionCriteria(
            inclusion=["Age 18-65 years", "Meets study criteria", "Able to consent"],
            exclusion=["Current study participation", "Contraindications", "Unable to complete"],
        )
        
        # Assemble final design
        import uuid
        from datetime import datetime
        
        experiment_design = ExperimentDesign(
            design_id=f"exp-{uuid.uuid4().hex[:8]}",
            title=f"Study: {research_question[:50]}",
            field=field,
            objective=research_question,
            experiment_type=experiment_type,
            null_hypothesis=hypothesis_result["null_hypothesis"],
            alternative_hypothesis=hypothesis_result["alternative_hypothesis"],
            variables=design_result["variables"],
            groups=final_groups,
            population="Target population meeting inclusion criteria",
            sample_size=analysis_result["power_analysis"].total_n,
            inclusion_criteria=inclusion,
            duration_days=duration_days,
            power_analysis=analysis_result["power_analysis"],
            protocol_steps=design_result["protocol_steps"],
            analysis_plan=analysis_result["analysis_plan"],
            safety_plan=design_result["safety_plan"],
            reproducibility=reproducibility,
            created_at=datetime.now().isoformat(),
            created_by="Principal Investigator AI",
        )
        
        self.log("\n" + "="*50)
        self.log("✅ EXPERIMENT DESIGN COMPLETE")
        self.log(f"Sample size: n={experiment_design.sample_size}")
        self.log(f"Duration: {duration_days} days")
        self.log("="*50)
        
        return {
            "experiment_design": experiment_design,
            "literature_review": hypothesis_result["literature_review"],
            "research_gaps": hypothesis_result["research_gaps"],
            "visualizations": analysis_result["visualizations"],
            "interpretation_guidelines": analysis_result["interpretation_guidelines"],
        }
