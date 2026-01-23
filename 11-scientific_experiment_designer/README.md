# 🔬 Scientific Experiment Designer

An **AI system** that designs scientific experiments, predicts outcomes, suggests controls, identifies confounding variables, and generates protocols using **AutoGen with Nested Team Architecture**.

![AutoGen](https://img.shields.io/badge/Framework-AutoGen-blue)
![Architecture](https://img.shields.io/badge/Architecture-Nested_Teams-purple)
![Complexity](https://img.shields.io/badge/Complexity-⭐⭐⭐⭐⭐-red)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📚 **Literature Review** | Search and synthesize existing research |
| 🔍 **Gap Identification** | Find unexplored research opportunities |
| 📐 **Hypothesis Generation** | Create testable null and alternative hypotheses |
| 🧪 **Protocol Design** | Step-by-step experimental procedures |
| ⚖️ **Controls & Confounds** | Identify variables and mitigation strategies |
| ⚡ **Power Analysis** | Calculate required sample sizes |
| 📊 **Statistical Planning** | Pre-specify analysis methods |
| 🛡️ **Safety Review** | Assess risks and ethical considerations |
| ✅ **Reproducibility** | Generate reproducibility checklists |

## 🏗️ Nested Team Architecture

```
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
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │ Literature  │ │ │ │ Protocol    │ │ │ │ Statistics  │ │
│ │ Reviewer    │ │ │ │ Designer    │ │ │ │ Expert      │ │
│ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │ Gap         │ │ │ │ Controls    │ │ │ │ Data        │ │
│ │ Identifier  │ │ │ │ Specialist  │ │ │ │ Visualizer  │ │
│ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │ Hypothesis  │ │ │ │ Safety      │ │ │ │ Results     │ │
│ │ Generator   │ │ │ │ Reviewer    │ │ │ │ Interpreter │ │
│ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

## 🔬 Research Fields Supported

| Field | Example Questions |
|-------|-------------------|
| **Biology** | Gene knockout effects, cell proliferation |
| **Psychology** | Cognitive interventions, behavior studies |
| **Medicine** | Clinical trials, treatment efficacy |
| **Pharmacology** | Dose-response, drug interactions |
| **Neuroscience** | Neural mechanisms, brain imaging |
| **Chemistry** | Reaction kinetics, synthesis optimization |

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt

# Set API key (optional)
export GOOGLE_API_KEY="your-key"
```

### CLI Usage

```bash
# Design an experiment
python main.py design "Does cognitive training improve memory?" --field Psychology

# Design with custom parameters
python main.py design "Effect of drug X on outcome Y" --field Medicine --type RCT --duration 84

# Save protocol to file
python main.py design "Research question" --field Biology -o protocol.md

# Show example questions
python main.py examples Psychology

# Interactive mode
python main.py interactive

# Web server
python main.py serve
```

### Python API

```python
from experiment_designer import ExperimentDesigner, design_experiment

# Full designer
designer = ExperimentDesigner()
result = designer.design(
    research_question="Does intervention X improve outcome Y?",
    field="Medicine",
    experiment_type="RCT",
    effect_size=0.5,
    duration_days=56,
)

# Generate protocol document
print(result["proposal"].to_protocol_markdown())

# Quick function
result = design_experiment("Does X affect Y?", field="Psychology")
```

## 📁 Project Structure

```
experiment-designer/
├── experiment_designer/
│   ├── __init__.py           # Main ExperimentDesigner class
│   ├── teams/
│   │   └── __init__.py       # Nested AutoGen teams (9 agents)
│   ├── tools/
│   │   └── __init__.py       # Research tools (literature, stats, safety)
│   ├── models/
│   │   └── __init__.py       # Data models (hypotheses, variables, etc.)
│   └── templates/
│       └── __init__.py       # Statistical tests, protocols
├── api.py                     # FastAPI backend
├── frontend/
│   └── index.html            # React dashboard
├── main.py                    # Rich CLI
├── requirements.txt
└── README.md
```

## 📊 Sample Protocol Output

```markdown
# Experiment Protocol: Effect of X on Y

## 1. Objective
Test whether [intervention] affects [outcome] in [population]

## 2. Hypotheses
H₀: No significant difference between groups
H₁: Treatment group shows increased effect (α = 0.05)

## 3. Experimental Design
- **Type:** Randomized Controlled Trial
- **Groups:** Treatment (n=30) vs Control (n=30)
- **Duration:** 8 weeks

## 4. Variables
| Type | Variable | Measurement |
|------|----------|-------------|
| Independent | Treatment dose | mg/kg |
| Dependent | Outcome metric | Scale 1-10 |
| Controlled | Age, sex, baseline | Matching |
| Confounding | Diet, sleep | Questionnaire |

## 5. Protocol Steps
1. Recruit participants meeting inclusion criteria
2. Baseline measurements (Day 0)
3. Randomize into groups
4. Administer treatment/placebo (Days 1-56)
5. Weekly measurements
6. Final assessment (Day 56)
7. Statistical analysis

## 6. Statistical Analysis Plan
- Primary: Two-sample t-test
- Secondary: ANCOVA with baseline covariate
- Effect size: Cohen's d
- Power: 80% to detect d=0.5

## 7. Safety Considerations
- 🟢 Risk Level: Low
- Stopping rules defined
- IRB approval required
```

## 🧪 Experiment Types

| Type | Description |
|------|-------------|
| **RCT** | Randomized Controlled Trial |
| **Factorial** | Multiple factors tested simultaneously |
| **Crossover** | Participants receive all treatments |
| **Cohort** | Observational follow-up study |
| **Case-Control** | Retrospective comparison |

## 📐 Statistical Tests Available

| Test | Use Case |
|------|----------|
| **t-test** | Comparing two group means |
| **ANOVA** | Comparing 3+ group means |
| **ANCOVA** | Controlling for covariates |
| **Chi-square** | Categorical associations |
| **Correlation** | Linear relationships |
| **Regression** | Predicting outcomes |

## 🛡️ Safety Assessment

| Risk Level | Description |
|------------|-------------|
| 🟢 **Minimal** | No foreseeable risks |
| 🟢 **Low** | Minor discomfort possible |
| 🟡 **Moderate** | Requires monitoring |
| 🟠 **High** | Significant precautions needed |
| 🔴 **Critical** | Specialized oversight required |

## ⚙️ Configuration

```python
from experiment_designer import ExperimentDesigner, DesignerConfig

config = DesignerConfig(
    llm_provider="gemini",        # "gemini", "anthropic", "openai"
    default_alpha=0.05,           # Significance level
    default_power=0.80,           # Statistical power
    default_effect_size=0.5,      # Expected effect (Cohen's d)
    default_duration_days=56,     # Experiment duration
    verbose=True,
)

designer = ExperimentDesigner(config)
```

## 🌐 Web API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/fields` | GET | List research fields |
| `/api/experiment-types` | GET | List experiment types |
| `/api/examples` | GET | Example research questions |
| `/api/design` | POST | Design experiment (async) |
| `/api/design/{job_id}` | GET | Get design status |
| `/api/statistical-tests` | GET | Statistical test info |

## 📝 License

MIT License
