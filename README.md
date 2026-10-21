# ClinicalGated: Gait Video Classification of Adult Spinal Deformity with Clinician-Defined Priors

PyTorch code for the manuscript

> **ClinicalGated: Gait Video Classification of Adult Spinal Deformity as a Potential Functional Adjunct to Spinal Imaging**
> Kaixu Chen\*, Zihang Xie\*, Weihang You, Tomoyuki Asada, Kousei Miura, Hideki Kadone, Naoya Kikuchi, Yasuhiro Homma, Masashi Yamazaki, Itaru Kitahara, Yoshihiro Kuroda (\* equal contribution)
> Submitted to *Meta-Radiology*, 2026.

ClinicalGated classifies side-view gait videos of patients with adult spinal deformity (ASD), dropped head syndrome (DHS) and lumbar canal stenosis / hip osteoarthritis (LCS-HipOA). Orthopaedic surgeons marked the body regions they attend to when judging gait; these regions are turned into keypoint-based attention priors, injected into a SlowR50 3D-CNN backbone through a learned channel-wise gate at a single intermediate stage, and supervised through a side head so that the learned attention can be checked against the clinician annotations on unseen patients.

## Main results (patient-disjoint held-out protocol, gait-clip level, mean ± std over 3 folds)

| Model | 3-class Acc (%) | Macro Prec (%) | Macro F1 (%) | ASD screening AUROC (%) |
|---|---|---|---|---|
| 3D-CNN, RGB only | 52.6 ± 9.4 | 31.3 ± 12.7 | 27.9 ± 7.9 | 49.1 ± 5.6 |
| Multiplicative input masking (image × ROI) | 68.0 ± 8.2 | 49.5 ± 8.6 | 48.3 ± 6.6 | 61.8 ± 13.1 |
| **ClinicalGated, single-stage [3]** | **69.2 ± 5.5** | **51.6 ± 5.3** | **48.8 ± 4.2** | **80.5 ± 5.3** |

At the untuned operating point the main model reaches 94% sensitivity and 41% specificity for ASD versus non-ASD. The supervised side-head attention lands inside the clinician-annotated region in about 97% of held-out frames (pointing game). Full tables, ablations and the interpretability analysis are in the paper.

## Naming: paper ↔ code

| In the paper | In this repository |
|---|---|
| ClinicalGated | `model.fuse_method=pose_atn` (internal class name `PoseFusionRes3DCNN`, historically "PoseGated") |
| single-stage [3] (main model) | `model.ablation_study=single model.fusion_layers=3` |
| cumulative multi-stage [0..k] | `model.ablation_study=multi model.fusion_layers=k` |
| Add / Mul / Concat / Avg / Late fusion | `model.fuse_method=add|mul|concat|avg|late` |
| Squeeze-and-excitation / cross-attention fusion | `model.fuse_method=se_atn|cross_atn` |
| RGB-only 3D-CNN | `model.fuse_method=none` |
| gate bias *b* | `model.gate_init_bias` (default 2.0) |
| side heads, background and temporal losses | `model.use_side_heads`, `loss.selection`, `loss.w_bg`, `loss.w_temp` |

## Repository layout

```
project/              training / evaluation code
├── train.py          Hydra entry point: K-fold training (python -m project.train)
├── eval.py           evaluation entry point (python -m project.eval)
├── prepare_folds.py  builds and caches the patient-grouped fold split
├── cross_validation.py  StratifiedGroupKFold by patient, held-out protocol
├── dataloader/       LightningDataModule, clinical attention-map construction (med_attn_map.py)
├── models/           SlowR50 backbone, fusion variants, PoseFusionRes3DCNN (= ClinicalGated)
└── trainer/          LightningModules per fusion family
configs/              Hydra configs (config.yaml = training, eval.yaml = evaluation)
pegasus/              PBS job scripts used for every experiment in the paper; EXPERIMENTS.md records them
analysis/             scripts that produce the tables and figures of the paper
prepare_skeleton_dataset/  YOLOv8 keypoint extraction
tests/                shape tests for the models
```

## Installation

```bash
git clone https://github.com/ChenKaiXuSan/ClinicalGated_ASD_PyTorch.git
cd ClinicalGated_ASD_PyTorch
pip install -r requirements.txt   # PyTorch, PyTorch Lightning, pytorchvideo, Hydra, ultralytics, ...
```

The SlowR50 weights (`SLOW_8x8_R50.pyth`) are downloaded automatically from the pytorchvideo model zoo on first use.

## Data

The gait video dataset was collected at the University of Tsukuba Hospital (ethics approval H30-087) and is **not publicly available** because of patient privacy. Anonymised data may be shared on reasonable request to the corresponding author, subject to institutional approval.

The code expects, under `data.root_path`, per-video JSON descriptors (`video_name`, `video_path`, `disease`), extracted keypoint sequences (`data.skeleton_path`) and the clinician region annotations (`data.doctor_results_path`) from which the attention maps are built. See `configs/config.yaml` for every path.

## Reproducing the paper

All reported numbers use the clean held-out protocol: an outer patient-disjoint test fold, an inner patient-grouped validation split for early stopping, three folds, 16-frame chunks at 224×224, chunk probabilities averaged per gait clip. Set `data.heldout_test=True` everywhere.

```bash
# 1. build the fold cache once
python -m project.prepare_folds data.root_path=/path/to/data data.heldout_test=True

# 2. main model (ClinicalGated single-stage [3]); one fold per job, fold_idx 0..2
python -m project.train data.root_path=/path/to/data data.heldout_test=True \
    model.fuse_method=pose_atn model.ablation_study=single model.fusion_layers=3 \
    train.fold=3 train.fold_idx=0 train.gpu=1

# 3. baselines / competitors, e.g. RGB-only and multiplicative masking
python -m project.train data.root_path=/path/to/data data.heldout_test=True model.fuse_method=none train.fold=3 train.fold_idx=0
python -m project.train data.root_path=/path/to/data data.heldout_test=True model.fuse_method=mul  train.fold=3 train.fold_idx=0
```

The PBS scripts that ran every configuration are in `pegasus/`:

| Paper section | Script |
|---|---|
| main comparison (Tables 4–5) | `run_train_heldout_competitors.sh`, `run_train_heldout_late.sh`, `run_train_heldout_location.sh` |
| fusion location (Table 6) | `run_train_heldout_location.sh` |
| gate / loss ablation (Table 7) | `run_train_heldout_ablation.sh` |
| ROI perturbation (Table 8) | `run_analysis_heldout_L3.sh` → `analysis/attention_perturbation.py` |
| attention–ROI alignment (Table 9) | `run_analysis_heldout.sh` → `analysis/attention_alignment.py` |
| qualitative figure (Fig. 3) | `run_fig_qual_heldout.sh` → `analysis/make_combined_fig.py`, `analysis/mask_faces_fig.py` |

Aggregating the per-fold outputs into the paper's tables:

```bash
python -m analysis.aggregate_heldout  data.root_path=/path/to/data   # 3-class accuracy / macro precision / macro F1
python -m analysis.binary_asd_heldout data.root_path=/path/to/data   # ASD-vs-non-ASD sensitivity, specificity, AUROC
```

## Tests

```bash
cd tests && pytest -xvs model/
```

## Citation

Manuscript under review; the entry will be updated on publication.

```bibtex
@article{chen2026clinicalgated,
  title   = {ClinicalGated: Gait Video Classification of Adult Spinal Deformity as a Potential Functional Adjunct to Spinal Imaging},
  author  = {Chen, Kaixu and Xie, Zihang and You, Weihang and Asada, Tomoyuki and Miura, Kousei and Kadone, Hideki and Kikuchi, Naoya and Homma, Yasuhiro and Yamazaki, Masashi and Kitahara, Itaru and Kuroda, Yoshihiro},
  journal = {Meta-Radiology (submitted)},
  year    = {2026}
}
```

Earlier work on the same cohort:

- Chen K, et al. Two-stage video-based convolutional neural networks for adult spinal deformity classification. *Front Neurosci*. 2023. doi:10.3389/fnins.2023.1278584
- Chen K, et al. PhaseMix: a periodic motion fusion method for adult spinal deformity classification. *IEEE Access*. 2024. doi:10.1109/ACCESS.2024.3479165
- Chen K, et al. A clinical knowledge-guided attention framework for gait-based adult spinal deformity diagnosis. *Proc SPIE Medical Imaging*. 2026. doi:10.1117/12.3083822
- Chen K, et al. Clinical knowledge-guided attention for gait-based adult spinal deformity assessment from monocular video. *J Med Imaging*. 2026. doi:10.1117/1.JMI.13.5.054002

## Acknowledgements

Computations were performed on Pegasus at the Center for Computational Sciences, University of Tsukuba (Multidisciplinary Cooperative Research Program; HPCI project hp260146). This research was supported by AMED (JP256f0137008) and the Center for Cyber Medicine Research, University of Tsukuba.

## License

Apache-2.0, see `LICENSE`.
