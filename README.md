# DDMD-V: Distilling Uniform Discrete Diffusion Models into a One-Step Video Generator

**DDMD-V** (Discrete Distribution Matching Distillation for Video) distills the uniform discrete
video diffusion model [URSA](https://github.com/baaivision/URSA) into a one-step generator **Mei**.

Paper: TODO

## Method

DDMD-V trains the one-step student with two complementary signals.

**On-policy categorical distribution matching.** 

**Spatiotemporal adversarial supervision.** 

## Results

## Installation

```bash
conda create -n ddmdv python=3.14 -y
conda activate ddmdv
pip install -r requirements.txt
pip install -e .
```

## Checkpoints

## Inference

## VBench evaluation

The protocol follows URSA: 946 VBench prompts × 5 samples = 4,730 videos of 49 frames at
320×512. Videos are generated from VBench's dense rewritten captions with the `motion=9.0, `
prefix and named after the raw prompts, and all 16 dimensions are scored.

**1. Generate videos** (training environment, 8 GPUs; interrupted runs resume):

```bash
bash evaluations/vbench/sample.sh student checkpoints/ddmd_v/student.pt samples/vbench/ddmd_v
bash evaluations/vbench/sample.sh teacher - samples/vbench/ursa_50steps
STEPS=8 bash evaluations/vbench/sample.sh teacher - samples/vbench/ursa_8steps
```

**2. Score** in a separate VBench environment:

```bash
bash evaluations/vbench/setup_env.sh vbench
conda activate vbench
NGPUS=8 bash evaluations/vbench/evaluate.sh samples/vbench/ddmd_v results/vbench/ddmd_v
```

`evaluate.sh` runs the 16 dimensions and prints the VBench Total, Quality and Semantic scores
computed by `evaluations/vbench/total_score.py` with VBench's normalisation and weights
(Total = (4 × Quality + Semantic) / 5, `dynamic_degree` weighted 0.5 within Quality).

## Training

## Citation

TODO

## Acknowledgements

The teacher model and the `diffnext` library come from
[URSA](https://github.com/baaivision/URSA). The distillation builds on
[DMD/DMD2](https://github.com/tianweiy/DMD2) and [Di[M]O](https://github.com/yuanzhi-zhu/DiMO);
evaluation uses [VBench](https://github.com/Vchitect/VBench).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
