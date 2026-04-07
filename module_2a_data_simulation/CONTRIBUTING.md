# Contributing to Module 1A

Thank you for your interest in contributing to the Autism Physio-AI Pipeline.

---

## Getting Started

### Fork and clone

```bash
git clone https://github.com/your-org/autism-physio-ai-pipeline.git
cd autism-physio-ai-pipeline/module_1a_data_simulation
pip install -r requirements.txt
```

### Verify your setup

```bash
python main.py --list_emotions
python main.py --duration 60 --n_events 2 --seed 1 --out /tmp/test_run
```

---

## Development Priorities

The following areas are most welcome for contribution:

### High Priority

- **Subject-variability sampling** — Currently each simulation draws a single set of baseline parameters with small random perturbations. A proper population model would draw each simulation from a demographic distribution (age, sex, autism severity, medication status).

- **Sensory overload state** — A common autism-specific state characterised by extreme EDA elevation, high HR, and high-frequency movement. Physiologically distinct from Anger/Fear.

- **Pain state** — Relevant for non-verbal autistic children who cannot communicate pain verbally. Characterised by sustained EDA elevation, moderate HR increase, and guarding-type movement.

- **Unit tests** — A `tests/` directory with pytest-based tests covering signal shape validation, range clipping, reproducibility, and annotation correctness.

### Medium Priority

- **Multi-subject session generation** — A batch API that generates N sessions with inter-subject variability in baseline parameters.

- **Artefact injection** — Structured artefacts beyond random noise: electrode peel-off, device removal, gross motion saturation.

- **Additional device profiles** — Sampling rate and range configurations for Empatica E4+, Biopac, Shimmer, and Polar devices.

### Lower Priority

- **Transition modelling** — Currently emotion events have sharp onset/offset boundaries. Gradual emotion transitions would be more realistic.

- **Co-occurring states** — Simultaneous hunger + tired, or fear + anger. Currently only one state is active at a time.

---

## Code Standards

### Style

- PEP 8 compliant. Line length ≤ 100 characters.
- Type hints on all public function signatures.
- Docstrings on all public classes and methods (NumPy docstring format).

### Configuration

All tuneable parameters belong in `config.py`. No magic numbers in `signal_models.py`, `noise_injector.py`, or other processing files.

### Reproducibility

Any new random process must consume from the passed `rng: np.random.Generator` argument. Do not create independent `np.random` calls or `random` module calls that bypass the seeded generator.

### Data contract

Do not change the structure of `SimulationResult` without bumping `__version__` in `__init__.py` and updating all documentation. Downstream modules depend on this contract.

---

## Adding a New Emotion/State

1. Add an entry to `EMOTION_PROFILES` in `config.py`. See [Extending the Module](README.md#extending-the-module) for the required structure.
2. Add the emotion to the appropriate category list (`EMOTIONS_AFFECTIVE` or `EMOTIONS_NEEDS`).
3. Add an entry to the colour mapping in `visualizer.py` (`SIGNAL_COLORS` does not apply here, but ensure the `color` hex in the profile is unique and distinguishable on plots).
4. Document the new state in `docs/EMOTION_PROFILES.md` with physiological rationale and a literature reference.
5. Test with `python main.py --emotion YourNewState --n_events 3 --duration 120`.

---

## Pull Request Process

1. Create a branch: `git checkout -b feature/your-feature-name`
2. Make your changes with clear, focused commits.
3. Ensure `python main.py` (default run) completes without error.
4. Update relevant documentation files.
5. Open a pull request with:
   - A description of the change and its motivation
   - References to any relevant papers or physiological literature
   - Example output (if adding a new emotion profile or visual change)

---

## Reporting Issues

Use GitHub Issues. Please include:
- Python version and OS
- Command line invocation or code snippet that reproduces the issue
- Full traceback if applicable
- Expected vs. actual behaviour

---

## Code of Conduct

This project follows standard open-source community conduct norms. Contributions should be respectful and constructive. The subject matter (autism, paediatric healthcare) warrants particular sensitivity in all discussions.
