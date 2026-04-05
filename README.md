# CausalTriGAN

A tri-generator framework that simultaneously produces a synthetic chest X-ray image, a pathology-localisation heatmap, and a structured radiology report from a single multi-label pathology conditioning input. The central novelty is a causal intervention loss that validates generated heatmaps through sufficiency and necessity testing against a frozen Oracle classifier, providing spatial guarantees built in at generation time rather than applied post-hoc.

The framework uses three generators: G1 (ProjectedGAN) for image synthesis achieving FID 26.28 on CheXpert, G2 (U-Net with Attention Gates and FiLM conditioning) for heatmap generation achieving causal necessity of 0.979 with 7.6% sparsity, and G3 (frozen ViT-BERT) for report generation achieving BLEU-1 of 0.2197 and ROUGE-L of 0.2216. Training follows a three-phase progressive strategy with a causal intervention loss grounded in Pearl's interventional framework.

---

## Dataset

**CheXpert** — Stanford University chest radiograph dataset.

- 40,974 frontal-view chest X-ray images
- 14 pathology labels: No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices
- Labels: binary (0/1) with uncertainty labels (−1) mapped to positive under the U-Ones policy
- Split: 90% training (36,876 samples) / 10% validation (4,098 samples)
- Images resized to 256×256, normalised to [−1, 1]
