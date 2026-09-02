# Literature Review: Skin Cancer Detection Using Deep Learning

> Scope: Vision Transformer (ViT), CNN, hybrid CNN–Transformer, GRU-fusion, and related deep-learning approaches for dermatological cancer and melanoma detection.

## 1. Literature Review Overview

The reviewed studies show a clear progression from conventional CNN-based models toward Transformer-based and hybrid architectures for dermoscopic image analysis. Across the selected papers, the dominant research focus is improving accuracy, feature extraction, generalization, interpretability, and real-world clinical usefulness.

### Major Themes

- Vision Transformers are increasingly favored for global feature learning in skin lesion images.
- Hybrid CNN–Transformer models often outperform single-architecture models by combining local and global patterns.
- Data augmentation and synthetic image generation are used to address class imbalance and improve robustness.
- Model interpretability and demographic diversity remain major concerns in clinical adoption.
- Real-world deployment, workflow integration, and long-term evaluation are recurring research gaps.

---

## 2. Detailed Study Review

### 1. AI-Driven Dermatological Cancer Detection: A Vision Transformer Based Novel Approach

**Publication:** 2025 · B. Ramasubramanian et al.

#### Summary
This study proposes a Vision Transformer model for dermatological cancer detection from dermoscopic images. The model achieved 98.45% accuracy, 97.74% precision, and 98.17% recall, outperforming CNN-based approaches and demonstrating the effectiveness of transformer-based architectures in medical image analysis.

#### Future Research Directions
- Integrate transformer-based models with other AI techniques to enhance detection performance.
- Test the system in real clinical environments.
- Expand the dataset to include diverse skin types and lesion categories.
- Design clinician-friendly interfaces for AI-assisted diagnosis.
- Evaluate the long-term effect of AI-based diagnosis on treatment and patient outcomes.

#### Research Gaps
- Limited discussion of ViT limitations for specific skin conditions.
- Minimal analysis of performance across different demographics.
- Lack of evidence on workflow integration in clinical practice.
- No long-term assessment of model reliability over time.
- Limited attention to model interpretability.

#### Dataset
- Diverse collection of labeled skin lesion images

---

### 2. Vision Transformers (ViT) for Enhanced Skin Cancer Classification

**Publication:** 2024 · Mohamed Dahmani et al.

#### Summary
This study applies ViT models for skin cancer classification and reports strong performance, with accuracy reaching 96% and 94% on two different datasets. The findings suggest that ViT-based models can outperform CNNs and other deep learning models in dermatological image analysis.

#### Future Research Directions
- Improve ViT performance for skin cancer detection and classification.
- Combine ViTs with other advanced AI techniques.
- Increase dataset diversity for better generalization.
- Explore real-time dermoscopic analysis.
- Study interpretability to increase trust among clinicians.

#### Research Gaps
- Limited discussion of ViT limitations in edge-case clinical scenarios.
- Insufficient generalization analysis across populations and skin types.
- Limited exploration of computational cost and deployment challenges.
- Lack of interpretability evaluation.
- No long-term assessment of patient-level impact.

#### Dataset
- Benign versus malignant skin cancer dataset
- HAM10000 dataset

---

### 3. Comparative Analysis of Single and Hybrid CNN-Transformer Models for Melanoma Detection Using the ISIC Dataset

**Publication:** 2025 · Chenghan Wang · Applied and Computational Engineering

#### Summary
This study compares single and hybrid CNN-Transformer models for melanoma detection using the ISIC dataset. The findings show that hybrid architectures outperform standalone models, although they require more computational resources. The paper highlights the need to balance performance and efficiency for practical deployment.

#### Future Research Directions
- Optimize hybrid models to reduce computational cost while preserving accuracy.
- Include patient demographics and clinical history for improved decision support.
- Develop real-time diagnostic tools for early melanoma detection.
- Test model generalizability across varied populations and skin tones.
- Develop lightweight models for resource-constrained environments.

#### Research Gaps
- High computational cost remains a major challenge.
- Performance-efficiency trade-off is not fully resolved.
- Dataset diversity may limit generalizability.
- Interpretability is not deeply examined.
- No long-term clinical workflow evaluation is presented.

#### Dataset
- ISIC dataset

---

### 4. Gradient-guided Multi-attention Fusion: An Artificial Intelligence-enhanced Melanoma Diagnosis System Integrating Convolutional Neural Network and Vision Transformer Architectures

**Publication:** 2025 · Nouf Alzahrani et al.

#### Summary
This paper proposes Gradient-guided Multi-attention Fusion (GGMAF), an AI-based melanoma diagnosis system that integrates CNNs and ViTs. The model improves diagnostic accuracy by focusing on fine-grained lesion details and structural patterns in dermoscopic images.

#### Future Research Directions
- Improve attention-guided interpretability.
- Integrate GGMAF with other imaging modalities.
- Assess scalability across diverse clinical settings.
- Develop real-time analysis tools for clinical workflows.
- Perform longitudinal studies to assess patient outcomes.

#### Research Gaps
- Limited evaluation across different clinical environments.
- Lack of integration with broader diagnostic systems.
- No long-term performance assessment in real-world settings.
- Insufficient discussion of hardware and computational requirements.
- Limited analysis of robustness under image noise and artifacts.

#### Dataset
- Not specified in the provided study notes

---

### 5. Robust Skin Cancer Detection through CNN-Transformer-GRU Fusion and Generative Adversarial Network Based Data Augmentation

**Publication:** 2025 · Alex Varghese et al. · CMES – Computer Modeling in Engineering & Sciences

#### Summary
This study proposes a CNN-Transformer-GRU fusion model supported by GAN-based augmentation for robust skin cancer detection. The model achieved 90.61% accuracy on HAM10000 and improved precision, recall, and AUC compared with baseline methods.

#### Future Research Directions
- Explore additional augmentation methods to address imbalance.
- Integrate other architectures to improve generalization.
- Extend the model to additional lesion and cancer types.
- Develop real-time clinical diagnostic tools.
- Evaluate performance across demographic groups.

#### Research Gaps
- Limited discussion of real-world clinical robustness.
- No long-term evaluation beyond the HAM10000 dataset.
- Limited interpretability for medical users.
- Insufficient comparison among augmentation strategies.
- No strong discussion of clinical workflow integration.

#### Dataset
- HAM10000 dataset

---

### 6. Vision Transformer and Attention-Based Melanoma Disease Classification

**Publication:** 2023 · Prakhar Shobhit et al.

#### Summary
This paper investigates the role of self-attention in Vision Transformers for melanoma classification and examines how attention weights capture fine lesion features in dermoscopic images.

#### Future Research Directions
- Improve fine-tuning strategies for medical image data.
- Combine ViTs with other architectures for stronger performance.
- Test on other skin lesion types beyond melanoma.
- Study attention-map interpretability.
- Expand the dataset to improve robustness.

#### Research Gaps
- Limited discussion of computational cost and training time.
- Little evidence on cross-population generalizability.
- Interpretability of attention maps is not fully explored.
- No clinical workflow integration analysis.
- Bias and fairness concerns are not sufficiently examined.

#### Dataset
- Not specified in the provided study notes

---

### 7. Multi-Branch Vision Transformer for Early Skin Cancer Detection

**Publication:** 2025 · Jirachat Rujiwan et al.

#### Summary
This study proposes a multi-branch Vision Transformer for early skin cancer detection. The model achieved 92.0% accuracy and 94.0% recall in melanoma detection and was designed to handle limitations of conventional CNN-based approaches.

#### Future Research Directions
- Improve interpretability of decision-making.
- Extend the model to rare skin conditions beyond Dermatofibroma.
- Include demographic and clinical metadata for better accuracy.
- Test deployment in real clinical settings.
- Evaluate long-term impact on patient outcomes and healthcare efficiency.

#### Research Gaps
- Existing CNN weaknesses in global pattern recognition remain unresolved in some settings.
- Interpretability remains limited.
- Rare lesion classification remains underdeveloped.
- Scalability in resource-limited healthcare settings is not fully explored.
- Long-term adaptability across diverse populations needs study.

#### Dataset
- 8,318 dermoscopic images covering seven lesion categories

---

### 8. Enhancing Melanoma Diagnosis with Advanced Deep Learning Models Focusing on Vision Transformer, Swin Transformer, and ConvNeXt

**Publication:** 2024 · Serra Aksoy et al. · Dermatopathology

#### Summary
This study compares ConvNeXt, Vision Transformer, and Swin Transformer for melanoma diagnosis using dermoscopic images. ConvNeXt achieved 91.5% accuracy with balanced precision and recall, outperforming the other models in early melanoma detection.

#### Future Research Directions
- Explore hybrid architectures for medical image analysis.
- Improve model efficiency using better preprocessing and richer datasets.
- Apply these models to other cutaneous tumors.
- Expand datasets to improve generalizability.
- Develop real-time clinical tools for early diagnosis.

#### Research Gaps
- Dataset diversity remains limited.
- Interpretability is not fully explored.
- Clinical workflow integration is not studied.
- Real-time deployment feasibility is not addressed.
- There is no clear roadmap for future improvements.

#### Dataset
- Dataset sourced from Kaggle
- 13,900 uniformly sized images

---

### 9. Skin Cancer Detection Using Vision Transformers: An AI-Driven Approach for Precision Dermatology

**Publication:** 2025 · Manjot Kaur Sidhu et al.

#### Summary
This paper presents a ViT-based framework for skin cancer detection and reports 91.2% accuracy and 0.94 AUC-ROC. The model is reported to outperform traditional CNNs and offers strong potential for automated skin cancer screening.

#### Future Research Directions
- No specific future research suggestions were provided in the available notes.

#### Research Gaps
- Limited generalization to diverse clinical settings.
- No long-term comparison with traditional approaches.
- No workflow integration study.
- Attention visualizations are not deeply evaluated for clinical use.
- Limited concern for variation across skin types and lesion conditions.

#### Dataset
- Not specified in the provided study notes

---

### 10. Innovative Melanoma Diagnosis: Harnessing VI Transformer Architecture

**Publication:** 2024 · Sreelakshmi Jayasankar et al. · International Journal of Advanced Computer Science and Applications

#### Summary
This study introduces a Vision Transformer architecture for melanoma diagnosis and reports 96.23% accuracy on dermoscopy images. The model outperforms comparable methods and shows strong potential for improving diagnostic precision in clinical settings.

#### Future Research Directions
- Combine the ViT model with other machine learning techniques.
- Evaluate performance in real-world clinical environments.
- Expand datasets beyond Kaggle to cover broader lesion diversity.
- Develop clinician-friendly interfaces for real use.
- Assess long-term effects on patient outcomes and healthcare efficiency.

#### Research Gaps
- Limited discussion of ViT performance across diverse skin types and conditions.
- Insufficient interpretability analysis.
- No practical workflow integration analysis.
- No assessment of performance in long-term real-world use.
- No direct comparison with dermatologists or expert clinicians.

#### Dataset
- Dermoscopy images from Kaggle repository
- Large collection of high-quality medical imagery

---

## 3. Cross-Study Comparison

| # | Architecture / Approach | Reported Result | Dataset Mentioned | Main Gap |
|---|---|---|---|---|
| 1 | ViT | 98.45% accuracy; 97.74% precision; 98.17% recall | Diverse labeled skin-lesion images | Generalization, interpretability, clinical validation |
| 2 | ViT | 96% and 94% accuracy on two datasets | Benign vs. malignant; HAM10000 | Computational cost, dataset diversity |
| 3 | CNN–Transformer | Hybrid models outperform single models | ISIC | Accuracy–efficiency balance |
| 4 | CNN + ViT + attention | Improved melanoma diagnosis | Not specified | Deployment and robustness |
| 5 | CNN + Transformer + GRU + GAN | 90.61% accuracy, improved AUC/precision/recall | HAM10000 | Clinical validation and augmentation study |
| 6 | ViT + attention | Strong melanoma classification performance | Not specified | Interpretability and fairness |
| 7 | Multi-branch ViT | 92.0% accuracy; 94.0% recall | 8,318 dermoscopic images | Rare lesion recognition and scalability |
| 8 | ConvNeXt + ViT + Swin | 91.5% accuracy | Kaggle, 13,900 images | Diversity and deployment practicality |
| 9 | ViT | 91.2% accuracy; 0.94 AUC-ROC | Not specified | Generalization and long-term reliability |
| 10 | ViT | 96.23% accuracy | Kaggle dermoscopy dataset | Human-expert comparison and diversity |

---

## 4. Common Research Gaps

Across the studies, several limitations appear repeatedly:

- Limited validation across diverse skin tones, ethnicities, ages, and lesion types.
- Insufficient testing on external or real-world clinical datasets.
- Lack of interpretability and explainability in transformer-based models.
- High computational cost of hybrid and transformer systems.
- Limited integration into existing dermatology workflows.
- Weak long-term evaluation of reliability and patient outcomes.
- Need for lightweight models suitable for practical deployment.
- Inadequate comparison with dermatologists or expert clinicians.

---

## 5. Overall Conclusion

The literature strongly suggests that Vision Transformers and hybrid CNN–Transformer models are highly effective for skin cancer detection and melanoma classification. These architectures consistently perform well because they combine strong local feature extraction with global contextual reasoning. However, their practical usefulness in healthcare still depends on resolving interpretability, generalization, fairness, and clinical deployment issues.

For future work, a promising direction is to design a robust and explainable skin cancer detection model that is validated on diverse populations, benchmarked using multiple metrics, and tested for real-world clinical suitability rather than relying only on overall accuracy.

---

## 6. Note on Source Quality

This document is a structured refactoring of the supplied notes. The original information was preserved as closely as possible, while placeholders and duplicated fragments were cleaned up to produce a readable research-summary draft.
