# Environment notes

## Historically verified runtime

- Python: 3.12.13
- PyTorch: 2.13.0+cpu
- Execution device: CPU
- Captum: 0.8.0
- RapidFuzz: 3.14.1
- DataLoader workers: 0
- PyTorch deterministic algorithms: enabled by the training code

CUDA and ROCm were not used. Historical logs do not contain an authoritative OS edition/build, CPU make/model, installed RAM, explicit AMP flag, or exact versions for the unpinned packages below.

## Reconstructed dependencies

`requirements.txt` preserves the project dependency specification. NumPy, pandas, SciPy, scikit-learn, Matplotlib, seaborn, PyYAML, pytest, and tqdm were not historically pinned, so installing them today may not reproduce the original package set exactly. This is a reproducibility limitation, not evidence that the frozen results are invalid.

## Recommended archival action

Before public release, build a clean environment, record the complete package lock, operating system, and hardware inventory, and run the non-training validation suite plus figure regeneration. Do not use that environment audit to alter frozen scientific outputs.
