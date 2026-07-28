# GPT-2 validation note

Validation completed on GitHub Actions.

- full v2 unit-test suite: passed
- GPT2_REWARD smoke experiment: passed
- GPT2_ACADEMY_MODEL smoke experiment: passed
- matched no-academy baseline output: generated
- creativity guardrail fields: generated
- v2_complex 30 episodes x 10 seeds benchmark: completed for C5, GPT2_REWARD, GPT2_ACADEMY_MODEL, and GPT2_ACADEMY_FULL
- raw CSV/JSON artifacts: uploaded by workflow run 30325845508
- detailed interpretation: `docs/gpt2_v2_complex_30x10_results.md`

Main result: GPT2_ACADEMY_FULL improved over its matched no-academy baseline without reducing trajectory diversity, but remained below C5. GPT2_REWARD alone degraded performance and requires redesign before larger experiments.
