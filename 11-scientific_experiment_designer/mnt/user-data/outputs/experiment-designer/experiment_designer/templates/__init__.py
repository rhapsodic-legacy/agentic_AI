"""
Scientific Experiment Designer - Templates

Standard templates and examples for:
- Experimental designs
- Statistical tests
- Protocols
- Safety considerations
"""

from ..models import (
    ExperimentType, VariableType, MeasurementLevel, RiskLevel, ResearchField,
    Variable, StatisticalTest, SafetyConsideration, ReproducibilityItem, ProtocolStep
)


# =============================================================================
# Standard Statistical Tests
# =============================================================================

STATISTICAL_TESTS = {
    "t_test_independent": {
        "name": "Independent Samples t-test",
        "use_case": "Comparing means of two independent groups",
        "assumptions": [
            "Independence of observations",
            "Normal distribution in each group",
            "Homogeneity of variance (Levene's test)",
        ],
        "assumption_tests": ["Shapiro-Wilk", "Levene's test"],
        "non_parametric": "Mann-Whitney U test",
        "effect_size": "Cohen's d",
    },
    "t_test_paired": {
        "name": "Paired Samples t-test",
        "use_case": "Comparing means from the same subjects at two time points",
        "assumptions": [
            "Paired observations",
            "Normal distribution of differences",
        ],
        "assumption_tests": ["Shapiro-Wilk on differences"],
        "non_parametric": "Wilcoxon signed-rank test",
        "effect_size": "Cohen's d",
    },
    "anova_one_way": {
        "name": "One-way ANOVA",
        "use_case": "Comparing means across 3+ independent groups",
        "assumptions": [
            "Independence of observations",
            "Normal distribution in each group",
            "Homogeneity of variance",
        ],
        "assumption_tests": ["Shapiro-Wilk", "Levene's test"],
        "non_parametric": "Kruskal-Wallis H test",
        "effect_size": "Eta-squared (η²)",
        "post_hoc": ["Tukey HSD", "Bonferroni", "Scheffé"],
    },
    "anova_two_way": {
        "name": "Two-way ANOVA",
        "use_case": "Testing effects of two factors and their interaction",
        "assumptions": [
            "Independence of observations",
            "Normal distribution",
            "Homogeneity of variance",
        ],
        "effect_size": "Partial eta-squared (η²p)",
    },
    "ancova": {
        "name": "ANCOVA",
        "use_case": "Comparing groups while controlling for covariates",
        "assumptions": [
            "Linearity between covariate and dependent variable",
            "Homogeneity of regression slopes",
            "Independence of covariate and treatment",
        ],
        "effect_size": "Partial eta-squared (η²p)",
    },
    "chi_square": {
        "name": "Chi-square test",
        "use_case": "Testing association between categorical variables",
        "assumptions": [
            "Expected frequencies ≥ 5 in each cell",
            "Independence of observations",
        ],
        "non_parametric": "Fisher's exact test (small samples)",
        "effect_size": "Cramér's V",
    },
    "correlation": {
        "name": "Pearson correlation",
        "use_case": "Testing linear relationship between two continuous variables",
        "assumptions": [
            "Linearity",
            "Bivariate normality",
            "No outliers",
        ],
        "non_parametric": "Spearman's rank correlation",
        "effect_size": "r (correlation coefficient)",
    },
    "regression_linear": {
        "name": "Linear Regression",
        "use_case": "Predicting continuous outcome from predictors",
        "assumptions": [
            "Linearity",
            "Independence of residuals",
            "Homoscedasticity",
            "Normality of residuals",
            "No multicollinearity",
        ],
        "effect_size": "R², adjusted R²",
    },
    "logistic_regression": {
        "name": "Logistic Regression",
        "use_case": "Predicting binary outcome from predictors",
        "assumptions": [
            "Binary dependent variable",
            "Independence of observations",
            "Little multicollinearity",
            "Large sample size",
        ],
        "effect_size": "Odds ratio, Nagelkerke R²",
    },
}


# =============================================================================
# Sample Size Calculations
# =============================================================================

SAMPLE_SIZE_FORMULAS = {
    "t_test": {
        "description": "Two-sample t-test",
        "formula": "n = 2 * ((z_α + z_β)² * σ²) / δ²",
        "typical_sizes": {
            "small_effect": 64,   # d = 0.5
            "medium_effect": 26,  # d = 0.8
            "large_effect": 12,   # d = 1.2
        },
    },
    "anova": {
        "description": "One-way ANOVA",
        "typical_sizes": {
            "small_effect": 76,   # f = 0.25
            "medium_effect": 20,  # f = 0.4
        },
    },
    "correlation": {
        "description": "Correlation",
        "typical_sizes": {
            "small_effect": 85,   # r = 0.3
            "medium_effect": 30,  # r = 0.5
        },
    },
}


# =============================================================================
# Common Confounding Variables by Field
# =============================================================================

COMMON_CONFOUNDERS = {
    ResearchField.BIOLOGY: [
        {"name": "Age", "mitigation": "Match or stratify by age groups"},
        {"name": "Sex", "mitigation": "Include both sexes equally or analyze separately"},
        {"name": "Genetic background", "mitigation": "Use inbred strains or GWAS correction"},
        {"name": "Housing conditions", "mitigation": "Standardize housing and randomize cage locations"},
        {"name": "Time of day", "mitigation": "Conduct experiments at consistent times"},
        {"name": "Batch effects", "mitigation": "Include batch as covariate or use batch correction"},
    ],
    ResearchField.PSYCHOLOGY: [
        {"name": "Age", "mitigation": "Match or control statistically"},
        {"name": "Education level", "mitigation": "Match or include as covariate"},
        {"name": "Socioeconomic status", "mitigation": "Measure and control for SES"},
        {"name": "Prior exposure", "mitigation": "Screen for prior experience"},
        {"name": "Demand characteristics", "mitigation": "Use deception or double-blind design"},
        {"name": "Order effects", "mitigation": "Counterbalance presentation order"},
    ],
    ResearchField.MEDICINE: [
        {"name": "Age", "mitigation": "Stratified randomization"},
        {"name": "Sex", "mitigation": "Include both sexes, analyze separately if needed"},
        {"name": "Comorbidities", "mitigation": "Exclusion criteria or matching"},
        {"name": "Medication use", "mitigation": "Washout period or include as covariate"},
        {"name": "Disease severity", "mitigation": "Stratify by baseline severity"},
        {"name": "Adherence", "mitigation": "Monitor and report compliance"},
    ],
    ResearchField.PHARMACOLOGY: [
        {"name": "Body weight", "mitigation": "Weight-based dosing"},
        {"name": "Metabolic rate", "mitigation": "Measure drug levels"},
        {"name": "Drug interactions", "mitigation": "Exclusion criteria for concomitant meds"},
        {"name": "Time since last dose", "mitigation": "Standardize timing"},
        {"name": "Food intake", "mitigation": "Fasting state or standardized meals"},
    ],
}


# =============================================================================
# Safety Considerations by Risk Level
# =============================================================================

SAFETY_TEMPLATES = {
    "human_subjects": [
        SafetyConsideration(
            consideration_id="safe-1",
            category="ethical",
            description="Informed consent required",
            risk_level=RiskLevel.MINIMAL,
            mitigation="Provide clear consent forms, allow questions",
            monitoring="Document consent process",
        ),
        SafetyConsideration(
            consideration_id="safe-2",
            category="psychological",
            description="Potential for stress or discomfort",
            risk_level=RiskLevel.LOW,
            mitigation="Provide breaks, allow withdrawal at any time",
            monitoring="Check in with participants regularly",
        ),
        SafetyConsideration(
            consideration_id="safe-3",
            category="physical",
            description="Data privacy and confidentiality",
            risk_level=RiskLevel.LOW,
            mitigation="Anonymize data, secure storage",
            monitoring="Regular security audits",
        ),
    ],
    "animal_research": [
        SafetyConsideration(
            consideration_id="safe-a1",
            category="ethical",
            description="Animal welfare compliance",
            risk_level=RiskLevel.MODERATE,
            mitigation="Follow IACUC guidelines, minimize suffering",
            monitoring="Regular veterinary checks",
        ),
        SafetyConsideration(
            consideration_id="safe-a2",
            category="physical",
            description="Proper handling and housing",
            risk_level=RiskLevel.LOW,
            mitigation="Trained personnel only, appropriate facilities",
            monitoring="Daily health checks",
        ),
    ],
    "chemical_hazards": [
        SafetyConsideration(
            consideration_id="safe-c1",
            category="physical",
            description="Exposure to hazardous chemicals",
            risk_level=RiskLevel.MODERATE,
            mitigation="PPE required, fume hood use, MSDS review",
            monitoring="Exposure monitoring, regular safety training",
        ),
    ],
    "biological_hazards": [
        SafetyConsideration(
            consideration_id="safe-b1",
            category="physical",
            description="Exposure to biological agents",
            risk_level=RiskLevel.MODERATE,
            mitigation="Biosafety cabinet, appropriate BSL practices",
            monitoring="Spill protocols, incident reporting",
        ),
    ],
}


# =============================================================================
# Reproducibility Checklist Items
# =============================================================================

REPRODUCIBILITY_ITEMS = [
    # Methods
    ReproducibilityItem(item="Detailed protocol with all steps documented", category="methods"),
    ReproducibilityItem(item="Specific reagent information (catalog numbers, lot numbers)", category="materials"),
    ReproducibilityItem(item="Equipment specifications and settings", category="materials"),
    ReproducibilityItem(item="Software versions documented", category="analysis"),
    ReproducibilityItem(item="Sample size justification provided", category="methods"),
    ReproducibilityItem(item="Randomization method described", category="methods"),
    ReproducibilityItem(item="Blinding procedures documented", category="methods"),
    
    # Data
    ReproducibilityItem(item="Raw data available", category="data"),
    ReproducibilityItem(item="Data processing steps documented", category="data"),
    ReproducibilityItem(item="Missing data handling described", category="data"),
    
    # Analysis
    ReproducibilityItem(item="Analysis code available", category="analysis"),
    ReproducibilityItem(item="Statistical tests pre-specified", category="analysis"),
    ReproducibilityItem(item="Effect sizes reported", category="analysis"),
    ReproducibilityItem(item="Confidence intervals provided", category="analysis"),
]


# =============================================================================
# Protocol Step Templates
# =============================================================================

PROTOCOL_TEMPLATES = {
    "rct_clinical": [
        ProtocolStep(
            step_number=1,
            title="Participant Recruitment",
            description="Identify and recruit participants meeting inclusion criteria through specified channels.",
            materials=["Recruitment materials", "Screening forms"],
            safety_precautions=["Verify eligibility", "Document informed consent"],
        ),
        ProtocolStep(
            step_number=2,
            title="Baseline Assessment",
            description="Collect baseline measurements and demographic information.",
            measurements=["Primary outcome", "Secondary outcomes", "Demographics"],
            materials=["Measurement instruments", "Case report forms"],
        ),
        ProtocolStep(
            step_number=3,
            title="Randomization",
            description="Randomly assign participants to treatment or control groups.",
            materials=["Randomization schedule", "Group assignment forms"],
            critical_notes="Ensure allocation concealment",
        ),
        ProtocolStep(
            step_number=4,
            title="Intervention Administration",
            description="Administer treatment to intervention group and placebo/standard care to control group.",
            safety_precautions=["Monitor for adverse events", "Document administration"],
        ),
        ProtocolStep(
            step_number=5,
            title="Follow-up Assessments",
            description="Collect outcome measurements at specified time points.",
            measurements=["Primary outcome", "Secondary outcomes", "Adverse events"],
        ),
        ProtocolStep(
            step_number=6,
            title="Final Assessment",
            description="Complete final measurements and debrief participants.",
            measurements=["Primary outcome", "Secondary outcomes", "Participant feedback"],
        ),
        ProtocolStep(
            step_number=7,
            title="Data Analysis",
            description="Perform planned statistical analyses according to analysis plan.",
            materials=["Analysis software", "Statistical code"],
        ),
    ],
    "laboratory": [
        ProtocolStep(
            step_number=1,
            title="Sample Preparation",
            description="Prepare biological samples according to standardized protocol.",
            materials=["Reagents", "Consumables", "Equipment"],
            safety_precautions=["PPE required", "Follow biosafety protocols"],
        ),
        ProtocolStep(
            step_number=2,
            title="Experimental Procedure",
            description="Perform main experimental manipulation.",
            equipment=["Required instruments"],
            critical_notes="Maintain consistent conditions",
        ),
        ProtocolStep(
            step_number=3,
            title="Data Collection",
            description="Record measurements and observations.",
            measurements=["Primary measurements"],
        ),
        ProtocolStep(
            step_number=4,
            title="Data Analysis",
            description="Process and analyze collected data.",
            materials=["Analysis software"],
        ),
    ],
}


# =============================================================================
# Example Research Questions by Field
# =============================================================================

EXAMPLE_RESEARCH_QUESTIONS = {
    ResearchField.BIOLOGY: [
        "Does compound X inhibit cell proliferation in cancer cell lines?",
        "What is the effect of gene knockout on organism development?",
        "How does environmental temperature affect metabolic rate?",
    ],
    ResearchField.PSYCHOLOGY: [
        "Does cognitive behavioral therapy reduce anxiety symptoms?",
        "What is the effect of sleep deprivation on decision-making?",
        "How does social media use affect adolescent well-being?",
    ],
    ResearchField.MEDICINE: [
        "Does drug X improve patient outcomes compared to standard care?",
        "What is the efficacy of intervention Y in treating condition Z?",
        "How does lifestyle modification affect disease progression?",
    ],
    ResearchField.PHARMACOLOGY: [
        "What is the dose-response relationship of compound X?",
        "Does drug combination show synergistic effects?",
        "What are the pharmacokinetic parameters of new compound?",
    ],
}


def get_statistical_test(test_type: str) -> dict:
    """Get information about a statistical test."""
    return STATISTICAL_TESTS.get(test_type, {})


def get_common_confounders(field: ResearchField) -> list:
    """Get common confounding variables for a research field."""
    return COMMON_CONFOUNDERS.get(field, [])


def get_safety_template(template_type: str) -> list:
    """Get safety considerations for a type of research."""
    return SAFETY_TEMPLATES.get(template_type, [])


def get_protocol_template(template_type: str) -> list:
    """Get protocol step templates."""
    return PROTOCOL_TEMPLATES.get(template_type, [])
