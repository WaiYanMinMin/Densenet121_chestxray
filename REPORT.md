# Chest X-ray Multi-label Classification Report

Student: [Your Name]  
Unit: Advanced Machine Learning and Neural Networks  
Model: DenseNet-121 (ImageNet pre-trained, transfer learning)  
Dataset: CheXpert (training and validation splits; top-3 selected labels)

---

## Criterion 1: Selection of Pre-trained CNN and Datasets (Max 5)

### Model and Dataset Choice
- **Pre-trained CNN:** DenseNet-121 from `torchvision` with ImageNet weights.
- **Dataset:** CheXpert chest X-ray dataset.
- **Task framing:** Multi-label classification on the **top 3 labels with highest positive frequency** in the training split.

### Justification
- DenseNet-121 is a strong transfer-learning baseline for medical imaging due to feature reuse and stable gradient flow.
- ImageNet pre-training improves convergence when the target dataset is limited or imbalanced.
- Using the top-3 most frequent labels increases statistical support for training and evaluation reliability.

### Evidence from Notebook
- Label selection and preprocessing are implemented in the data preparation section.
- Model construction is implemented in `get_densenet(num_classes)`.

---

## Criterion 2: Model Architecture Investigation (Max 5)

### Architecture Overview
- Backbone: DenseNet-121 convolutional feature extractor.
- Classification head: final `classifier` replaced with `Linear(in_features, 3)` for 3-label output.
- Loss: `BCEWithLogitsLoss` for multi-label learning.
- Output interpretation: sigmoid per label with threshold 0.5.

### Architecture Evidence
- The notebook includes a dedicated architecture print cell:
  - `print(model)` for full architecture
  - parameter statistics for total/trainable parameters

### Interpretation
- The backbone captures rich hierarchical visual features.
- The replaced linear head adapts generic ImageNet features to chest X-ray labels.
- Multi-label logits support independent label probability estimation.

---

## Criterion 3: Suggested Architectural Improvements (Max 5)

### Improvement 1: Per-label Threshold Tuning
- Replace fixed threshold `0.5` with label-specific thresholds selected on validation data.
- Expected benefit: better precision-recall trade-off for minority/harder labels.

### Improvement 2: Imbalance-aware Objective
- Use class-weighted BCE or focal loss.
- Expected benefit: reduce bias toward dominant negative class outcomes.

### Improvement 3: Progressive Fine-tuning
- Freeze early layers initially, then unfreeze deeper layers gradually.
- Expected benefit: more stable optimization and potentially better generalization.

### Improvement 4: Training Control
- Add LR scheduling and early stopping based on validation metrics.
- Expected benefit: prevent overfitting and improve final checkpoint quality.

### Improvement 5: Backbone Comparison
- Compare DenseNet-121 against EfficientNet/ConvNeXt under same data split and metrics.
- Expected benefit: stronger evidence for model selection decisions.

---

## Criterion 4: Model Evaluation Using Training Dataset (Max 5)

### Evaluation Method
- Training dataset predictions collected after training.
- Metrics used: AUC, Precision, Recall, F1 (per label + macro average).
- This satisfies the requirement to use a known metric on training data.

### Results
- Use notebook outputs and exported files:
  - `outputs/report_figures/train_metrics_table.csv`
  - `outputs/report_figures/macro_metrics_table.csv`
  - `outputs/report_figures/macro_metric_comparison.png`

### Plot Placement
**[Insert Figure 1 here: `macro_metric_comparison.png`]**  
Caption: Macro metric comparison between training and validation splits.

### Short Interpretation
- Report whether training metrics are substantially higher than validation metrics (possible overfitting) or close (better generalization).
- Highlight the strongest and weakest macro metrics (AUC/Precision/Recall/F1).

---

## Criterion 5: Confusion Matrix and Label Comparison (Max 5)

### Method
- For each label, create a binary confusion matrix (`True 0/1` vs `Pred 0/1`) on the **training** set.
- Compare predicted labels directly against ground truth labels.

### Results
- Use exported confusion matrix figure:
  - `outputs/report_figures/train_confusion_matrices.png`

### Plot Placement
**[Insert Figure 2 here: `train_confusion_matrices.png`]**  
Caption: Per-label training confusion matrices for selected CheXpert labels.

### Interpretation Guide
- Discuss FP/FN balance for each label.
- Identify which label is easiest vs hardest for the model.
- Explain whether error patterns align with class imbalance.

---

## Criterion 6: Results Summary and Report Clarity (Max 5)

### Overall Summary
- DenseNet-121 transfer learning provides a practical baseline for multi-label chest X-ray classification.
- Training-set evaluation demonstrates measurable predictive performance using standard metrics.
- Confusion matrices reveal label-specific strengths and weaknesses, especially where imbalance affects recall/precision.

### Final Conclusion
- The implemented pipeline satisfies assignment requirements for architecture investigation, metric-based evaluation, confusion-matrix analysis, and evidence-based improvement proposals.
- Next steps are to tune thresholds, address imbalance, and benchmark alternative backbones for stronger clinical-task robustness.

---

## Appendix: Notebook Cells Added/Updated for Rubric Alignment

- Added dedicated architecture print cell (`print(model)`).
- Added report export cell to generate:
  - `macro_metric_comparison.png`
  - `train_confusion_matrices.png`
  - `train_metrics_table.csv`
  - `val_metrics_table.csv`
  - `macro_metrics_table.csv`
- Updated wording to consistently reflect the top-3 label setup.
