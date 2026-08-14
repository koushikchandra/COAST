# Baseline Source Code

This directory contains the original source code from the five baseline papers we evaluated
against in COAST. Each subdirectory is a copy of the publicly released code from that paper.

Our adaptation of these models lives in `../baseline_spatial.py`, which replaces each method's
original image front-end with shared UNI patch features so all methods receive identical inputs.
See the module docstring of `baseline_spatial.py` for what specifically was adapted in each case.

---

## Baselines

### BLEEP
**Paper:** Xie et al., "BLEEP: Bi-Level Embeding of Exemplary Proteins for Spatial Transcriptomics", NeurIPS 2023  
**Code:** `BLEEP/`  
**What we kept:** bi-modal retrieval at inference (kNN in UNI embedding space, average expression)  
**What we replaced:** ViT image encoder → shared UNI features

### HisToGene
**Paper:** Pang et al., "HisToGene: Predicting Spatial Gene Expression from Histology Images", bioRxiv 2021  
**Code:** `HisToGene/`  
**What we kept:** ViT self-attention over all spots of a slide + learned (x,y) grid position embeddings  
**What we replaced:** raw image patches → shared UNI features

### Hist2ST
**Paper:** Zeng et al., "Hist2ST: Predicting Spatial Gene Expression from Histology Images", bioRxiv 2022  
**Code:** `Hist2ST/`  
**What we kept:** global transformer blocks + GraphSAGE GCN over kNN(coords) + jumping-knowledge LSTM fusion  
**What we replaced:** raw image patches → shared UNI features

### ST-Net
**Paper:** He et al., "Integrating Spatial Gene Expression and Breast Tumour Morphology via Deep Learning", Nature Biomedical Engineering 2020  
**Code:** `ST-Net/`  
**What we kept:** per-spot MLP regression (no spatial context)  
**What we replaced:** DenseNet-121 patch features → shared UNI features

### TRIPLEX
**Paper:** Chung et al., "TRIPLEX: Predicting Spatial Gene Expression via Three-Resolution Image Fusion", CVPR 2024  
**Code:** `TRIPLEX/`  
**What we kept:** three-resolution fusion (spot / neighbor kNN-mean / global slide-mean) + small transformer fusion head  
**What we replaced:** original feature extraction pipeline → shared UNI features

---

## Citation

If you use these baselines please cite their original papers:

```bibtex
@inproceedings{xie2023bleep,
  title={Spatially Resolved Gene Expression Prediction from {H\&E} Histology Images via Bi-modal Contrastive Learning},
  author={Xie, Minxing and others},
  booktitle={NeurIPS},
  year={2023}
}

@article{pang2021histogene,
  title={{HisToGene}: Predicting Spatial Gene Expression Profiles from Histology Images},
  author={Pang et al.},
  journal={bioRxiv},
  year={2021}
}

@article{zeng2022hist2st,
  title={Spatial Transcriptomics Prediction from Histology jointly through Transformer and Graph Neural Networks},
  author={Zeng et al.},
  journal={bioRxiv},
  year={2022}
}

@article{he2020integrating,
  title={Integrating spatial gene expression and breast tumour morphology via deep learning},
  author={He et al.},
  journal={Nature Biomedical Engineering},
  year={2020}
}

@inproceedings{chung2024triplex,
  title={{TRIPLEX}: Accurate Transcriptome Prediction from Histology Images},
  author={Chung et al.},
  booktitle={CVPR},
  year={2024}
}
```
