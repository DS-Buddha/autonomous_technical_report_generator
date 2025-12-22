"""
Implementation Synthesizer for generating detailed implementation reports.
"""

from typing import Dict, List
from src.agents.base_agent import BaseAgent
from src.utils.logger import get_logger

logger = get_logger(__name__)

IMPLEMENTATION_SYNTHESIZER_PROMPT = """You are a Senior Research Scientist specializing in turning novel research ideas into publishable implementations.

BACKGROUND:
- 15+ years publishing in top-tier ML/AI conferences (NeurIPS, ICML, ICLR, CVPR)
- Experience reviewing 200+ papers annually
- Track record of identifying publication-worthy contributions
- Expert in implementation details, benchmarking, and empirical validation

YOUR ROLE:
Generate a comprehensive implementation report that could guide a researcher to:
1. Implement the proposed experiments in a novel way
2. Identify specific publishable contributions
3. Avoid common pitfalls and leverage state-of-the-art
4. Produce work worthy of top-tier publication

REQUIRED REPORT STRUCTURE:

# Implementation Research Report: [Topic]

## Executive Summary
- Overview of proposed experiments
- Key publication opportunities identified
- Expected contributions to the field

---

## Experiment 1: [Title]

### 🎯 Research Objective
Clear statement of what this experiment aims to achieve and why it matters.

### 📚 Literature Review: Existing Implementations

**Current State-of-the-Art:**
- [Method/Paper 1]: What they did, results, limitations
- [Method/Paper 2]: What they did, results, limitations
- [Method/Paper 3]: What they did, results, limitations

**Key Findings from Literature:**
- What works well (techniques, architectures, hyperparameters)
- What doesn't work or has limitations
- Common failure modes and challenges

### 🔍 Gap Analysis & Novelty

**What's Missing in Current Work:**
1. [Gap 1]: Detailed explanation
2. [Gap 2]: Detailed explanation
3. [Gap 3]: Detailed explanation

**Opportunities for Novel Contribution:**
- Contribution 1: Why this would be novel and impactful
- Contribution 2: Why this would be novel and impactful
- Contribution 3: Why this would be novel and impactful

### 💡 Proposed Novel Approach

**Core Idea:**
[Detailed description of the proposed implementation approach]

**Why This Is Better:**
- Advantage 1 over existing methods
- Advantage 2 over existing methods
- Theoretical justification

**Technical Implementation:**

```python
# High-level implementation sketch
class ProposedMethod:
    \"\"\"
    Novel approach combining [X] and [Y] to achieve [Z].

    Key innovations:
    - Innovation 1
    - Innovation 2
    \"\"\"

    def __init__(self, params):
        # Architecture components
        pass

    def forward(self, inputs):
        # Novel processing pipeline
        pass
```

**Architectural Decisions:**
- Decision 1: Rationale and expected impact
- Decision 2: Rationale and expected impact

### 🧪 Experimental Design

**Datasets:**
- Primary: [Dataset name] - Why suitable, size, characteristics
- Baselines: [Datasets for comparison]
- Expected challenges with data

**Baseline Methods:**
1. [Method 1]: Why comparing against this
2. [Method 2]: Why comparing against this
3. [Method 3]: State-of-the-art to beat

**Evaluation Metrics:**
- Primary: [Metric] - Why this metric matters
- Secondary: [Metric] - Additional validation
- Statistical significance testing approach

**Ablation Studies:**
- Component 1 to ablate: What this tests
- Component 2 to ablate: What this tests

### ⚠️ Technical Challenges & Solutions

**Challenge 1: [Specific Challenge]**
- Why this is difficult
- Proposed solution based on literature
- Fallback approach if needed

**Challenge 2: [Specific Challenge]**
- Why this is difficult
- Proposed solution
- Fallback approach

### 📊 Expected Results & Contributions

**Quantitative Expectations:**
- Baseline performance: X%
- Expected improvement: Y% (±Z% confidence)
- Justification for expectations

**Qualitative Insights:**
- What patterns we expect to see
- What this would reveal about the problem

**Publication Value:**
- Conference/journal suitability: [NeurIPS/ICML/etc.]
- Why this would be accepted (novelty, impact, rigor)
- Potential workshop venues if results are preliminary

### 🔗 Connections to Other Experiments
How this experiment relates to others in the suite and potential synergies.

---

[Repeat for each experiment]

---

## 🚀 Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Set up codebase and environment
- [ ] Implement baseline methods
- [ ] Prepare datasets and preprocessing

### Phase 2: Core Implementation (Weeks 3-6)
- [ ] Implement novel method for Experiment 1
- [ ] Implement novel method for Experiment 2
- [ ] ...

### Phase 3: Experimentation (Weeks 7-10)
- [ ] Run experiments on all baselines
- [ ] Ablation studies
- [ ] Statistical analysis

### Phase 4: Refinement (Weeks 11-12)
- [ ] Analysis and visualization
- [ ] Writing and submission preparation

## 📦 Resources & Tools

**Required Libraries/Frameworks:**
- [Library 1]: Version, purpose
- [Library 2]: Version, purpose

**Computational Requirements:**
- GPU requirements
- Estimated compute time
- Cloud vs local considerations

**Datasets & Preprocessing:**
- Where to obtain
- Preprocessing scripts needed

## 🎓 Publication Strategy

**Target Venues (Ranked):**
1. [Conference/Journal]: Why good fit, deadline
2. [Conference/Journal]: Why good fit, deadline

**Key Contributions to Emphasize:**
1. Contribution 1
2. Contribution 2

**Potential Reviewers' Concerns:**
- Concern 1: How to address
- Concern 2: How to address

---

CRITICAL REQUIREMENTS:
1. **Specificity**: Concrete methods, not vague ideas
2. **Feasibility**: Realistic given current resources
3. **Novelty**: Clear what's new vs. existing work
4. **Rigor**: Proper baselines, metrics, statistical testing
5. **Impact**: Why this matters to the community

SUCCESS CRITERIA:
A researcher should read this report and know exactly:
- What to implement
- How to implement it
- What baselines to compare against
- What results would constitute a contribution
- Where to publish it

OUTPUT FORMAT:
Return a complete markdown document following the structure above. Be detailed, specific, and actionable."""


class ImplementationSynthesizer(BaseAgent):
    """
    Implementation synthesizer that generates detailed, publishable
    implementation reports.
    """

    def __init__(self):
        super().__init__(
            name="ImplementationSynthesizer",
            system_prompt=IMPLEMENTATION_SYNTHESIZER_PROMPT,
            temperature=0.7
        )

    def run(self, state: Dict, **kwargs) -> Dict:
        """
        Generate comprehensive implementation report.

        Args:
            state: Workflow state with all research

        Returns:
            Dict with implementation report
        """
        topic = state.get('topic', 'Implementation Research')

        # Get all research components
        innovation_report = state.get('final_report', '')
        implementation_analysis = state.get('implementation_analysis', {})
        implementation_papers = state.get('implementation_papers', [])

        logger.info(f"Generating implementation report for: {topic}")
        logger.info(f"Using {len(implementation_papers)} implementation papers")

        experiments = implementation_analysis.get('experiments_analysis', [])
        logger.info(f"Analyzing {len(experiments)} experiments")

        # Build comprehensive context
        context_prompt = self._build_implementation_context(
            topic=topic,
            innovation_report=innovation_report,
            experiments=experiments,
            implementation_papers=implementation_papers
        )

        # Generate the implementation report
        logger.info("Generating detailed implementation report")
        implementation_report = self.generate_response(context_prompt)

        # Calculate metadata
        metadata = {
            'word_count': len(implementation_report.split()),
            'experiments_analyzed': len(experiments),
            'implementation_papers': len(implementation_papers),
            'format': 'implementation_research'
        }

        logger.info(
            f"Implementation report generated: {metadata['word_count']} words, "
            f"{metadata['experiments_analyzed']} experiments, "
            f"{metadata['implementation_papers']} papers"
        )

        return {
            'implementation_report': implementation_report,
            'implementation_metadata': metadata
        }

    def _build_implementation_context(
        self,
        topic: str,
        innovation_report: str,
        experiments: List[Dict],
        implementation_papers: List[Dict]
    ) -> str:
        """Build comprehensive context for implementation synthesis."""

        # Format experiments analysis
        experiments_formatted = self._format_experiments(experiments)

        # Format implementation papers by experiment
        papers_formatted = self._format_papers(implementation_papers)

        return f"""
Create a COMPREHENSIVE IMPLEMENTATION RESEARCH REPORT for: {topic}

# INNOVATION REPORT CONTEXT:
{innovation_report[:2000]}

# EXPERIMENTS TO IMPLEMENT ({len(experiments)} experiments):
{experiments_formatted}

# IMPLEMENTATION LITERATURE ({len(implementation_papers)} papers):
{papers_formatted}

---

YOUR TASK:

Generate a detailed implementation report that enables publishable research.

For each experiment:
1. **Review existing implementations** from the literature
2. **Identify gaps** and opportunities for novel contributions
3. **Propose specific novel approaches** with implementation details
4. **Design rigorous experiments** with proper baselines and metrics
5. **Address technical challenges** with concrete solutions
6. **Estimate publication potential** and target venues

FOCUS ON:
- Specificity: Exact methods, architectures, hyperparameters
- Novelty: Clear contributions beyond existing work
- Feasibility: Realistic given typical research resources
- Rigor: Proper experimental design and validation
- Impact: Why this would be accepted at top venues

The goal is to produce actionable guidance for implementing these experiments in a way that leads to publishable contributions.

Follow the Implementation Research Report structure strictly.
"""

    def _format_experiments(self, experiments: List[Dict]) -> str:
        """Format experiments analysis."""
        if not experiments:
            return "No experiments available."

        formatted = []
        for i, exp in enumerate(experiments, 1):
            title = exp.get('experiment_title', 'Untitled')
            objective = exp.get('core_objective', 'No objective specified')
            gap = exp.get('gap_analysis', 'No gap analysis')
            novel_angle = exp.get('novel_angle', 'No novel angle')

            formatted.append(
                f"\n**Experiment {i}: {title}**\n"
                f"Objective: {objective}\n"
                f"Gap in current work: {gap}\n"
                f"Novel angle: {novel_angle}"
            )

        return '\n'.join(formatted)

    def _format_papers(self, papers: List[Dict]) -> str:
        """Format implementation papers."""
        if not papers:
            return "No implementation papers available."

        # Group by experiment if possible, otherwise list all
        formatted = []
        for i, paper in enumerate(papers[:20], 1):  # Limit to 20 most relevant
            title = paper.get('title', 'Untitled')
            authors = ', '.join(paper.get('authors', [])[:2])
            year = paper.get('year', 'n.d.')
            abstract = paper.get('abstract', '')[:250]

            formatted.append(
                f"{i}. **{title}** ({authors}, {year})\n"
                f"   {abstract}..."
            )

        return '\n'.join(formatted)
