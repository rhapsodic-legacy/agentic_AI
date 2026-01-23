"""
Scientific Experiment Designer - Research Tools

Tools for:
- Literature review and gap identification
- Hypothesis generation
- Experimental design
- Statistical planning
- Safety assessment
- Protocol generation
"""

from typing import Optional
from datetime import datetime
import random
import uuid
import math

from ..models import (
    ResearchField, ExperimentType, VariableType, MeasurementLevel, RiskLevel,
    Reference, ResearchGap, Hypothesis, Variable, Group, ProtocolStep,
    PowerAnalysis, StatisticalTest, AnalysisPlan, SafetyConsideration,
    StoppingRule, SafetyPlan, ReproducibilityItem, ReproducibilityChecklist,
    InclusionCriteria, ExperimentDesign, LiteratureReview
)
from ..templates import (
    STATISTICAL_TESTS, COMMON_CONFOUNDERS, SAFETY_TEMPLATES,
    REPRODUCIBILITY_ITEMS, PROTOCOL_TEMPLATES,
    get_statistical_test, get_common_confounders
)


# =============================================================================
# Literature Review Tools
# =============================================================================

def search_literature(query: str, field: ResearchField, max_results: int = 10) -> list[Reference]:
    """
    Search scientific literature (simulated).
    
    In production, this would interface with PubMed, Google Scholar, etc.
    """
    # Simulate relevant references
    references = []
    
    base_titles = {
        ResearchField.BIOLOGY: [
            "Molecular mechanisms of {} in cellular systems",
            "A systematic review of {} effects on gene expression",
            "Novel approaches to studying {} in model organisms",
            "The role of {} in developmental biology",
            "Quantitative analysis of {} pathways",
        ],
        ResearchField.PSYCHOLOGY: [
            "Cognitive effects of {} on decision-making",
            "A meta-analysis of {} interventions",
            "Neural correlates of {} processing",
            "Developmental aspects of {} behavior",
            "Individual differences in {} responses",
        ],
        ResearchField.MEDICINE: [
            "Clinical outcomes of {} treatment",
            "Randomized controlled trial of {} intervention",
            "Long-term effects of {} therapy",
            "Comparative effectiveness of {} approaches",
            "Safety and efficacy of {} in patients",
        ],
        ResearchField.PHARMACOLOGY: [
            "Pharmacokinetics of {} in human subjects",
            "Dose-response relationship of {} compound",
            "Drug-drug interactions involving {}",
            "Mechanism of action of {} receptor",
            "Bioavailability studies of {} formulation",
        ],
    }
    
    titles = base_titles.get(field, base_titles[ResearchField.BIOLOGY])
    
    for i in range(min(max_results, len(titles))):
        title = titles[i].format(query.split()[0] if query.split() else "study subject")
        
        references.append(Reference(
            ref_id=f"ref-{uuid.uuid4().hex[:8]}",
            title=title,
            authors=[f"Author{j+1}, A." for j in range(random.randint(2, 5))],
            year=random.randint(2018, 2024),
            journal=random.choice([
                "Nature", "Science", "PNAS", "Cell", "Journal of Experimental Biology",
                "Psychological Science", "JAMA", "Lancet", "Drug Discovery Today"
            ]),
            doi=f"10.1000/example.{uuid.uuid4().hex[:6]}",
            relevance_score=random.uniform(0.7, 0.95),
            key_findings=[f"Finding {j+1} related to {query}" for j in range(3)],
        ))
    
    return sorted(references, key=lambda r: r.relevance_score, reverse=True)


def identify_research_gaps(literature: list[Reference], field: ResearchField) -> list[ResearchGap]:
    """
    Identify gaps in existing research.
    """
    gaps = []
    
    gap_templates = [
        "Limited research on long-term effects in diverse populations",
        "Lack of mechanistic understanding of underlying processes",
        "Inconsistent findings across different methodologies",
        "Insufficient sample sizes in previous studies",
        "Need for replication in independent cohorts",
        "Unexplored moderating variables",
        "Limited ecological validity of laboratory findings",
    ]
    
    for i, template in enumerate(gap_templates[:5]):
        gaps.append(ResearchGap(
            gap_id=f"gap-{uuid.uuid4().hex[:6]}",
            description=template,
            supporting_references=[ref.ref_id for ref in literature[:2]],
            significance="This gap limits our understanding and clinical/practical applications.",
            addressable=True,
        ))
    
    return gaps


def generate_literature_review(query: str, field: ResearchField) -> LiteratureReview:
    """
    Generate a comprehensive literature review.
    """
    references = search_literature(query, field)
    gaps = identify_research_gaps(references, field)
    
    review = LiteratureReview(
        query=query,
        field=field,
        references=references,
        key_themes=[
            f"Theme 1: {query} mechanisms",
            f"Theme 2: {query} applications",
            f"Theme 3: Methodological approaches",
        ],
        consensus_findings=[
            f"Established finding about {query}",
            "Replicated effect across multiple studies",
        ],
        contradictory_findings=[
            "Conflicting results regarding magnitude of effect",
        ],
        research_gaps=gaps,
        common_methods=["RCT", "Observational study", "Meta-analysis"],
        sample_sizes=[random.randint(20, 200) for _ in range(5)],
        summary=f"The literature on {query} reveals several key themes and identifies important gaps for future research.",
    )
    
    return review


# =============================================================================
# Hypothesis Generation Tools
# =============================================================================

def generate_hypotheses(
    research_question: str,
    field: ResearchField,
    effect_direction: str = "increase",
    alpha: float = 0.05,
) -> tuple[Hypothesis, Hypothesis]:
    """
    Generate null and alternative hypotheses.
    """
    # Parse the research question
    words = research_question.lower().split()
    
    # Generate null hypothesis
    null_hypothesis = Hypothesis(
        hypothesis_id=f"h0-{uuid.uuid4().hex[:6]}",
        is_null=True,
        statement=f"There is no significant difference in outcomes between the treatment and control conditions.",
        predictions=["No measurable effect will be observed"],
        testable=True,
        falsifiable=True,
        alpha_level=alpha,
    )
    
    # Generate alternative hypothesis
    alt_hypothesis = Hypothesis(
        hypothesis_id=f"h1-{uuid.uuid4().hex[:6]}",
        is_null=False,
        statement=f"There is a significant {effect_direction} in outcomes for the treatment condition compared to control.",
        predictions=[
            f"Treatment group will show {effect_direction}d outcomes",
            "Effect will be statistically significant",
        ],
        testable=True,
        falsifiable=True,
        rationale=f"Based on the research question: {research_question}",
        alpha_level=alpha,
        effect_direction=effect_direction,
    )
    
    return null_hypothesis, alt_hypothesis


# =============================================================================
# Variable Identification Tools
# =============================================================================

def identify_variables(
    research_question: str,
    field: ResearchField,
    experiment_type: ExperimentType,
) -> list[Variable]:
    """
    Identify relevant variables for the experiment.
    """
    variables = []
    
    # Independent variable
    variables.append(Variable(
        variable_id=f"var-{uuid.uuid4().hex[:6]}",
        name="Treatment Condition",
        var_type=VariableType.INDEPENDENT,
        measurement_method="Group assignment",
        measurement_level=MeasurementLevel.NOMINAL,
        operational_definition="Presence or absence of experimental intervention",
    ))
    
    # Dependent variable
    variables.append(Variable(
        variable_id=f"var-{uuid.uuid4().hex[:6]}",
        name="Primary Outcome",
        var_type=VariableType.DEPENDENT,
        measurement_method="Standardized assessment",
        measurement_level=MeasurementLevel.RATIO,
        unit="scale units",
        operational_definition="Measured response to intervention",
    ))
    
    # Controlled variables
    controlled_vars = ["Age", "Baseline status", "Time of assessment"]
    for cv in controlled_vars:
        variables.append(Variable(
            variable_id=f"var-{uuid.uuid4().hex[:6]}",
            name=cv,
            var_type=VariableType.CONTROLLED,
            control_method="Matching or stratification",
        ))
    
    # Confounding variables from templates
    confounders = get_common_confounders(field)
    for conf in confounders[:3]:
        variables.append(Variable(
            variable_id=f"var-{uuid.uuid4().hex[:6]}",
            name=conf["name"],
            var_type=VariableType.CONFOUNDING,
            potential_impact="May affect outcome independently of treatment",
            mitigation_strategy=conf["mitigation"],
        ))
    
    return variables


# =============================================================================
# Statistical Planning Tools
# =============================================================================

def calculate_power_analysis(
    test_type: str,
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    groups: int = 2,
) -> PowerAnalysis:
    """
    Calculate required sample size for statistical power.
    """
    # Simplified power calculation (in production, use proper statistical formulas)
    z_alpha = 1.96 if alpha == 0.05 else 2.576
    z_beta = 0.84 if power == 0.80 else 1.28
    
    if test_type in ["t_test_independent", "t_test"]:
        n_per_group = int(2 * ((z_alpha + z_beta) ** 2) / (effect_size ** 2))
    elif test_type in ["anova", "anova_one_way"]:
        n_per_group = int(groups * ((z_alpha + z_beta) ** 2) / (effect_size ** 2))
    elif test_type == "correlation":
        # Fisher's z transformation
        n_per_group = int(((z_alpha + z_beta) / (0.5 * math.log((1 + effect_size) / (1 - effect_size + 0.001)))) ** 2 + 3)
    else:
        n_per_group = int(2 * ((z_alpha + z_beta) ** 2) / (effect_size ** 2))
    
    n_per_group = max(10, min(n_per_group, 500))  # Reasonable bounds
    
    return PowerAnalysis(
        test_type=test_type,
        effect_size=effect_size,
        effect_size_type="Cohen's d" if "t_test" in test_type else "f" if "anova" in test_type else "r",
        alpha=alpha,
        power=power,
        required_n_per_group=n_per_group,
        total_n=n_per_group * groups,
        assumptions=[
            "Normal distribution of data",
            "Equal variance across groups",
            "Independence of observations",
        ],
    )


def create_statistical_test(
    test_type: str,
    purpose: str = "primary",
) -> StatisticalTest:
    """
    Create a statistical test specification.
    """
    test_info = get_statistical_test(test_type)
    
    return StatisticalTest(
        test_id=f"test-{uuid.uuid4().hex[:6]}",
        name=test_info.get("name", test_type),
        purpose=purpose,
        alpha=0.05,
        tails=2,
        assumptions=test_info.get("assumptions", []),
        assumption_tests=test_info.get("assumption_tests", []),
        non_parametric_alternative=test_info.get("non_parametric", ""),
    )


def create_analysis_plan(
    primary_test_type: str,
    secondary_tests: list[str] = None,
) -> AnalysisPlan:
    """
    Create a complete statistical analysis plan.
    """
    primary = create_statistical_test(primary_test_type, "primary")
    
    secondaries = []
    if secondary_tests:
        for st in secondary_tests:
            secondaries.append(create_statistical_test(st, "secondary"))
    
    return AnalysisPlan(
        primary_analysis=primary,
        secondary_analyses=secondaries,
        effect_size_measure=STATISTICAL_TESTS.get(primary_test_type, {}).get("effect_size", "Cohen's d"),
        missing_data_strategy="intention-to-treat",
        correction_method="Bonferroni" if len(secondaries) > 1 else "None",
        software="R / Python (scipy, statsmodels)",
        interpretation_guidelines=[
            "Report exact p-values",
            "Include effect sizes with confidence intervals",
            "Distinguish between statistical and practical significance",
            "Report all pre-specified analyses",
        ],
    )


# =============================================================================
# Safety Planning Tools
# =============================================================================

def create_safety_plan(
    field: ResearchField,
    involves_humans: bool = True,
    involves_animals: bool = False,
    hazard_types: list[str] = None,
) -> SafetyPlan:
    """
    Create a comprehensive safety plan.
    """
    considerations = []
    
    if involves_humans:
        considerations.extend(SAFETY_TEMPLATES.get("human_subjects", []))
    
    if involves_animals:
        considerations.extend(SAFETY_TEMPLATES.get("animal_research", []))
    
    if hazard_types:
        if "chemical" in hazard_types:
            considerations.extend(SAFETY_TEMPLATES.get("chemical_hazards", []))
        if "biological" in hazard_types:
            considerations.extend(SAFETY_TEMPLATES.get("biological_hazards", []))
    
    # Stopping rules
    stopping_rules = [
        StoppingRule(
            rule_id="stop-1",
            condition="Serious adverse event occurs",
            action="Pause study, report to IRB/ethics board, assess causality",
        ),
        StoppingRule(
            rule_id="stop-2",
            condition="Interim analysis shows clear benefit or harm",
            action="Consider early termination per pre-specified criteria",
        ),
        StoppingRule(
            rule_id="stop-3",
            condition="Participant withdraws consent",
            action="Discontinue participation, retain collected data if consented",
        ),
    ]
    
    # Calculate overall risk
    risk_scores = [c.risk_level.value for c in considerations]
    risk_order = ["MINIMAL", "LOW", "MODERATE", "HIGH", "CRITICAL"]
    max_risk = max(risk_scores, key=lambda x: risk_order.index(x.upper()) if x.upper() in risk_order else 0) if risk_scores else "LOW"
    
    overall_risk = RiskLevel.LOW
    for level in RiskLevel:
        if level.value.upper() == max_risk.upper():
            overall_risk = level
            break
    
    return SafetyPlan(
        considerations=considerations,
        stopping_rules=stopping_rules,
        adverse_event_definition="Any untoward medical occurrence in a participant",
        adverse_event_reporting="Report within 24 hours to PI, within 7 days to IRB for serious events",
        emergency_contacts=["Principal Investigator", "IRB Office", "Emergency Services"],
        emergency_procedures="Follow institutional emergency protocols. Contact PI immediately.",
        irb_required=involves_humans,
        informed_consent_required=involves_humans,
        data_privacy_measures=[
            "Data anonymization",
            "Secure storage",
            "Access controls",
            "Compliance with HIPAA/GDPR as applicable",
        ],
        overall_risk_level=overall_risk,
    )


# =============================================================================
# Protocol Generation Tools
# =============================================================================

def generate_protocol_steps(
    experiment_type: ExperimentType,
    duration_days: int,
    groups: list[Group],
) -> list[ProtocolStep]:
    """
    Generate detailed protocol steps.
    """
    steps = []
    
    # Get template based on experiment type
    if experiment_type in [ExperimentType.RCT, ExperimentType.DOUBLE_BLIND, ExperimentType.SINGLE_BLIND]:
        template = PROTOCOL_TEMPLATES.get("rct_clinical", [])
    else:
        template = PROTOCOL_TEMPLATES.get("laboratory", [])
    
    # Customize template
    for step in template:
        customized = ProtocolStep(
            step_number=step.step_number,
            title=step.title,
            description=step.description,
            day_start=0 if step.step_number == 1 else (step.step_number - 1) * (duration_days // len(template)),
            day_end=(step.step_number) * (duration_days // len(template)),
            materials=step.materials,
            equipment=step.equipment,
            personnel=step.personnel,
            measurements=step.measurements,
            safety_precautions=step.safety_precautions,
            critical_notes=step.critical_notes,
        )
        steps.append(customized)
    
    return steps


def create_reproducibility_checklist() -> ReproducibilityChecklist:
    """
    Create a reproducibility checklist.
    """
    items = [ReproducibilityItem(
        item=item.item,
        category=item.category,
        completed=False,
    ) for item in REPRODUCIBILITY_ITEMS]
    
    return ReproducibilityChecklist(
        items=items,
        data_availability="Data will be made available upon publication",
        data_format="CSV/Excel with codebook",
        code_availability="Analysis code will be provided",
        code_repository="GitHub repository to be created",
        materials_list=[],
        reagent_sources=[],
    )


# =============================================================================
# Complete Design Generation
# =============================================================================

def design_experiment(
    research_question: str,
    field: ResearchField,
    experiment_type: ExperimentType = ExperimentType.RCT,
    effect_size: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.80,
    duration_days: int = 56,
    involves_humans: bool = True,
) -> ExperimentDesign:
    """
    Design a complete experiment based on research question.
    """
    # Generate hypotheses
    h0, h1 = generate_hypotheses(research_question, field, "increase", alpha)
    
    # Identify variables
    variables = identify_variables(research_question, field, experiment_type)
    
    # Power analysis
    test_type = "t_test_independent"
    if experiment_type == ExperimentType.FACTORIAL:
        test_type = "anova_two_way"
    elif experiment_type in [ExperimentType.COHORT, ExperimentType.LONGITUDINAL]:
        test_type = "regression_linear"
    
    power_analysis = calculate_power_analysis(test_type, effect_size, alpha, power)
    
    # Create groups
    groups = [
        Group(
            group_id="grp-treatment",
            name="Treatment",
            target_n=power_analysis.required_n_per_group,
            treatment_description="Active intervention",
            assignment_method="random",
            blinded=experiment_type in [ExperimentType.DOUBLE_BLIND, ExperimentType.SINGLE_BLIND],
        ),
        Group(
            group_id="grp-control",
            name="Control",
            target_n=power_analysis.required_n_per_group,
            treatment_description="Placebo or standard care",
            assignment_method="random",
            blinded=experiment_type in [ExperimentType.DOUBLE_BLIND, ExperimentType.SINGLE_BLIND],
        ),
    ]
    
    # Generate protocol
    protocol_steps = generate_protocol_steps(experiment_type, duration_days, groups)
    
    # Analysis plan
    analysis_plan = create_analysis_plan(test_type, ["ancova"])
    
    # Safety plan
    safety_plan = create_safety_plan(field, involves_humans)
    
    # Reproducibility
    reproducibility = create_reproducibility_checklist()
    
    # Inclusion criteria
    inclusion = InclusionCriteria(
        inclusion=[
            "Age 18-65 years",
            "Meets diagnostic criteria",
            "Able to provide informed consent",
        ],
        exclusion=[
            "Current participation in another study",
            "Contraindications to intervention",
            "Unable to complete study procedures",
        ],
    )
    
    return ExperimentDesign(
        design_id=f"exp-{uuid.uuid4().hex[:8]}",
        title=f"Study: {research_question[:50]}...",
        field=field,
        objective=research_question,
        experiment_type=experiment_type,
        null_hypothesis=h0,
        alternative_hypothesis=h1,
        variables=variables,
        groups=groups,
        population="Adults meeting inclusion criteria",
        sample_size=power_analysis.total_n,
        inclusion_criteria=inclusion,
        duration_days=duration_days,
        power_analysis=power_analysis,
        protocol_steps=protocol_steps,
        analysis_plan=analysis_plan,
        safety_plan=safety_plan,
        reproducibility=reproducibility,
        created_at=datetime.now().isoformat(),
        created_by="Experiment Designer AI",
    )
