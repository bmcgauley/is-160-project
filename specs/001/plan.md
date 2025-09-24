
# Implementation Plan: Build a Convolutional Neural Network for Employment Trends Analysis

**Branch**: `001-build-a-convolutional` | **Date**: 2025-09-24 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-build-a-convolutional/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Fill the Constitution Check section based on the content of the constitution document.
4. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, `GEMINI.md` for Gemini CLI, `QWEN.md` for Qwen Code or `AGENTS.md` for opencode).
7. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
Build a CNN using PyTorch to analyze and predict employment trends from California's QCEW data, processing multi-dimensional employment data including temporal patterns, geographic distributions, and industry classifications to predict quarterly employment changes. The implementation includes comprehensive feature engineering, data validation pipelines, and evaluation against forecasting baselines.

## Technical Context
**Language/Version**: Python 3.8+  
**Primary Dependencies**: PyTorch, pandas, scikit-learn, matplotlib, seaborn  
**Storage**: Files (QCEW CSV data files)  
**Testing**: pytest for unit tests, integration tests for data pipelines  
**Target Platform**: Cross-platform (Linux/Windows/macOS)  
**Project Type**: Single project (data science/machine learning)  
**Performance Goals**: Accurate employment trend prediction, efficient CNN processing of large datasets  
**Constraints**: Employment data privacy compliance, reproducible research with fixed seeds, modular PyTorch implementation  
**Scale/Scope**: California QCEW dataset (millions of records), CNN model for spatio-temporal analysis

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Compliance Assessment
- **I. Rigorous Data Validation**: Feature includes automated quality checks and validation pipelines at each processing step - COMPLIANT
- **II. Comprehensive Feature Engineering**: Leverages domain expertise in employment economics for predictive features - COMPLIANT  
- **III. Reproducible Research**: Utilizes fixed random seeds and detailed logging - COMPLIANT
- **IV. Modular PyTorch Implementation**: Clear block-by-block documentation for CNN components - COMPLIANT
- **V. Temporal and Geographic Data Handling**: Proper handling of spatio-temporal data for CNN input - COMPLIANT
- **VI. Privacy and Statistical Best Practices**: Adherence to employment data privacy and statistical standards - COMPLIANT

### Additional Constraints
- Python 3.8+ requirement - COMPLIANT (specified in context)
- PyTorch as required deep learning framework - COMPLIANT
- Automated validation pipelines - COMPLIANT
- Geographic data handling with appropriate libraries - COMPLIANT
- Temporal data preservation - COMPLIANT

### Development Workflow
- Peer review required - COMPLIANT (will be followed)
- Automated tests before merge - COMPLIANT (pytest specified)
- Documentation updates mandatory - COMPLIANT

**Status**: PASS - No violations detected. Feature aligns with all constitutional principles and constraints.

## Project Structure

### Documentation (this feature)
```
specs/001-build-a-convolutional/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
│   └── predict-employment.md
├── .github/
│   └── copilot-instructions.md  # Agent-specific guidance
└── tasks.md             # Phase 2 output (/tasks command - pre-existing)
```

### Source Code (repository root)
```
# Option 1: Single project (DEFAULT - selected for data science project)
src/
├── data_acquisition.py    # Data loading and preprocessing
├── validation.py          # Quality checks and validation
├── feature_engineering.py # Employment feature creation
├── cnn_model.py          # PyTorch CNN architecture
├── training.py           # Model training pipeline
├── evaluation.py         # Performance assessment
├── loss_metrics.py       # Custom loss functions
├── visualization.py      # Results plotting
├── baselines.py          # Baseline model comparison
└── utils.py              # Helper functions

tests/
├── test_data.py          # Data pipeline tests
├── test_validation.py    # Quality check tests
├── test_features.py      # Feature engineering tests
├── test_model.py         # CNN architecture tests
└── test_evaluation.py    # Performance tests

notebooks/
├── exploration.ipynb     # Data exploration
└── analysis.ipynb        # Results analysis

data/
├── raw/                  # Original QCEW files
├── processed/            # Cleaned features
└── validated/            # Quality-checked data

models/                   # Saved PyTorch models
reports/                  # Validation reports
figures/                  # Generated plots
```

**Structure Decision**: Option 1 (Single project) - Selected because this is a data science/ML project focused on employment analysis, not a web application or mobile app.

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:
   ```
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Generate contract tests** from contracts:
   - One test file per endpoint
   - Assert request/response schemas
   - Tests must fail (no implementation yet)

4. **Extract test scenarios** from user stories:
   - Each story → integration test scenario
   - Quickstart test = story validation steps

5. **Update agent file incrementally** (O(1) operation):
   - Run `.specify/scripts/powershell/update-agent-context.ps1 -AgentType copilot`
     **IMPORTANT**: Execute it exactly as specified above. Do not add or remove any arguments.
   - If exists: Add only NEW tech from current plan
   - Preserve manual additions between markers
   - Update recent changes (keep last 3)
   - Keep under 150 lines for token efficiency
   - Output to repository root

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, agent-specific file

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `.specify/templates/tasks-template.md` as base
- Generate tasks from Phase 1 design docs (contracts, data model, quickstart)
- Each contract → contract test task [P]
- Each entity → model creation task [P] 
- Each user story → integration test task
- Implementation tasks to make tests pass

**Ordering Strategy**:
- TDD order: Tests before implementation 
- Dependency order: Models before services before UI
- Mark [P] for parallel execution (independent files)

**Estimated Output**: 25-30 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |


## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - tasks.md pre-existing)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [ ] Complexity deviations documented

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*
