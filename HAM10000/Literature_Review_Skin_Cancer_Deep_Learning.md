# Literature Review: Skin Cancer Detection Using Deep Learning

## Selected PDF Sources

The following PDF files were selected for review and have been added to the literature review record:

1. paper_da5ac31c75ac5fbc8847e742ce77a9c3.pdf
2. paper_7314ac335b105680ad684da95b2777b8.pdf
3. paper_58d7f048719f5e9fb1139076c0c0769e.pdf
4. paper_4ae3a46584645ef49c1ffd38ab82bcdf.pdf
5. paper_48767bbc04ff53a98a83ccc5c9ccd399.pdf
6. paper_45866f3362a35de084d840bba4f6485e.pdf
7. paper_2b031f6e5e2d57adaa5aea41bcb02368.pdf
8. paper_20559e74133d51fc8d04593c678e7ea7.pdf
9. paper_199ed56ea8cd5ab18d0664abadcb21fc.pdf
10. paper_051af425744157c1b9fec6b63445a483.pdf
11. International Journal of Biomedical Imaging - 2024 - Himel - Skin Cancer Segmentation and Classification Using Vision.pdf

## 1. Introduction to the Literature Review

- This literature review focuses on skin cancer detection using deep learning.
- The purpose is to examine the major existing studies and identify patterns in methodologies, datasets, models, and findings.
- The review is based on the selected PDF sources listed above, and it will be completed using the available paper information.

## 2. Review of Individual Research Papers

### Paper 1: Deep Learning Approaches for Skin Lesion Detection
- Author(s) and Year: Jonathan Vieira, Fábio Mendonça, and Fernando Morgado-Dias (2025)
- Research Approach: Benchmarking of 38 CNN architectures from 10 families using transfer learning on HAM10000, followed by cross-database validation on ISIC 2019; a proof-of-concept mobile application was also implemented.
- Major Findings: ConvNeXtXLarge achieved the best performance with 87.62% overall accuracy and 76.15% F1-score on the HAM10000 test set. The study also showed that model performance varied across architectures even under standard transfer-learning conditions.
- Contribution: It provides a broad comparative analysis of standard CNN families, addresses cross-database generalization, and connects research performance with practical mobile-health screening.
- Limitation: The study primarily compares standard transfer-learning configurations rather than highly tuned custom models; therefore, it may not represent the upper performance bound of specialized architectures for skin lesion classification.

### Paper 2: A GAN-based Data Augmentation Method for Imbalanced Multi-class Skin Lesion Classification
- Author(s) and Year: Qichen Su, Haza Nuzly Abdull Hamed, Mohd Adham Isa, Xue Hao, and Xin Dai (2024)
- Research Approach: Proposed Self-Transfer GAN (STGAN), a two-stage GAN framework for synthetic image generation to handle class imbalance and improve classification on HAM10000.
- Major Findings: STGAN outperformed conditional StyleGAN2 in FID, Inception Score, Precision, and Recall, and the STGAN-based classification pipeline achieved 98.23% accuracy, 88.85% sensitivity, 90.23% precision, and 89.48% F1-score.
- Contribution: It addresses the core issue of severe class imbalance in dermoscopic datasets by generating diverse minority-class samples while preserving class-specific features.
- Limitation: The paper focuses mainly on synthetic data generation and classification performance; it does not strongly discuss clinical deployment or real-world validation on external datasets.

### Paper 3: Enhanced Skin Cancer Diagnosis Using Optimized CNN Architecture and Checkpoints for Automated Dermatological Lesion Classification
- Author(s) and Year: M. Mohamed Musthafa, Mahesh T. R., Vinoth Kumar V., and Suresh Guluwadi (2024)
- Research Approach: Optimized CNN model with data augmentation and model checkpointing on HAM10000 for multi-class skin lesion classification.
- Major Findings: The improved model achieved 97.78% accuracy, 97.9% precision, 97.9% recall, and 97.8% F2-score, demonstrating strong performance for lesion classification.
- Contribution: It shows that a carefully designed CNN structure combined with class-balancing augmentation can substantially improve diagnostic accuracy in dermatology.
- Limitation: The model is validated on a single benchmark dataset, and the paper does not provide strong evidence of external generalization to diverse clinical conditions.

### Paper 4: A Deep Learning-Based Dual-Branch Framework for Automated Skin Lesion Segmentation and Classification via Dermoscopic Images
- Author(s) and Year: Hamza Abu Owida, Ibrahim Abd El-Fattah, Suhaila Abuowaida, Nawaf Alshdaifat, Hamza A. Mashagba, Azlan B. Abd Aziz, Alaa Alzoubi, Samia Larguech, and Samir Salem Al-Bawri (2025)
- Research Approach: Dual-branch framework combining segmentation and classification; EfficientNet-B7 encoder with ASPP and transformer blocks for segmentation, plus DenseNet-121 and morphological feature fusion for classification.
- Major Findings: On HAM10000, the model achieved a Dice score of 0.9568 and IoU of 0.9242, and the classification accuracy reached 0.9708. Strong generalization was reported across PH2, ISIC 2016, ISIC 2017, and ISIC 2018.
- Contribution: It integrates lesion segmentation and morphology-aware classification to improve both boundary precision and decision-making quality.
- Limitation: The framework is computationally heavier and depends on multi-stage feature extraction; this may reduce practicality in low-resource or real-time clinical environments.

### Paper 5: A Novel Framework of Multiclass Skin Lesion Recognition from Dermoscopic Images Using Deep Learning and Explainable AI
- Author(s) and Year: Naveed Ahmad, Jamal Hussain Shah, Muhammad Attique Khan, Jamel Baili, Ghulam Jillani Ansari, Usman Tariq, Ye Jin Kim, and Jae-Hyuk Cha (2023)
- Research Approach: Data augmentation, deep transfer learning with fine-tuned Xception and ShuffleNet, feature fusion, improved butterfly optimization for feature selection, and Grad-CAM-based explainability.
- Major Findings: The framework achieved 99.3% accuracy on ISIC2018 and 91.5% accuracy on HAM10000, showing strong performance and improved computational efficiency with feature selection.
- Contribution: It combines explainable AI with deep features and optimization, improving both interpretability and classification performance.
- Limitation: The method is relatively complex and depends on several stages (augmentation, fusion, optimization, and explanation), which may increase implementation cost and complexity.

### Paper 6: Comparative Study of Transfer Learning Strategies for Multi-Class Skin Lesion Classification: Architectures, Fine-Tuning, and Data Augmentation
- Author(s) and Year: Dingguo Wang and Yudi Chen (2026)
- Research Approach: Systematic comparison of ResNet50, DenseNet121, and EfficientNet-B0 under different fine-tuning strategies and augmentation intensities on HAM10000.
- Major Findings: Full fine-tuning produced the highest validation accuracy (86.19%) and AUC (99.42%), while basic augmentation delivered the best performance (91.43% accuracy, AUC 99.30%). Aggressive augmentation degraded results due to feature distortion.
- Contribution: It provides practical guidance on transfer-learning configuration, including backbone choice, augmentation intensity, and optimization strategy.
- Limitation: The results are highly dependent on controlled experimental settings, and the study does not include extensive external clinical validation.

### Paper 7: An Efficient Deep Learning-Based Skin Cancer Classifier for an Imbalanced Dataset
- Author(s) and Year: Talha Mahboob Alam, Kamran Shaukat, Waseem Ahmad Khan, Ibrahim A. Hameed, Latifah Abd. Almuqren, Muhammad Ahsan Raza, Memoona Aslam, and Suhuai Luo (2022)
- Research Approach: Deep learning models including AlexNet, InceptionV3, and RegNetY-320 on the HAM10000 dataset, using data augmentation to reduce class imbalance.
- Major Findings: RegNetY-320 outperformed the baselines, achieving 91% accuracy, 88.1% F1-score, and ROC value of 0.95 on the improved framework.
- Contribution: It highlights the importance of handling class imbalance and demonstrates that balanced training improves the robustness of skin lesion classifiers.
- Limitation: The study focuses mainly on class imbalance correction and does not deeply examine generalization to different datasets or clinical settings.

### Paper 8: DSCC_Net: Multi-Classification Deep Learning Models for Diagnosing of Skin Cancer Using Dermoscopic Images
- Author(s) and Year: Maryam Tahir, Ahmad Naeem, Hassaan Malik, Jawad Tanveer, Rizwan Ali Naqvi, and Seung-Won Lee (2023)
- Research Approach: Proposed DSCC_Net, a CNN-based multi-classifier for skin cancer diagnosis using SMOTE Tomek balancing on HAM10000, ISIC 2020, and DermIS.
- Major Findings: DSCC_Net achieved 94.17% accuracy, 93.76% recall, 94.28% precision, and 93.93% F1-score, with an AUC of 99.43%.
- Contribution: It presents a specialized CNN model for four cancer classes and shows that class-balancing techniques can improve multiclass diagnosis.
- Limitation: The paper emphasizes model performance on benchmark datasets but does not sufficiently test deployment robustness under real clinical variability.

### Paper 9: Hybrid Vision Transformer and Graph Neural Network Model with Region-Adaptive Attention for Enhanced Skin Cancer Prediction
- Author(s) and Year: Aswani Dogga, Sivasubramanian R., and Shanthi S. (2026)
- Research Approach: Hybrid ViT-GNN architecture with Region-Adaptive Attention, meta-learning, and explainability for skin cancer classification.
- Major Findings: The model outperformed state-of-the-art deep learning methods on benchmark skin cancer datasets and was designed to improve interpretability and generalization.
- Contribution: It combines global context modeling from ViTs with spatial relationship learning from GNNs, addressing limitations of CNN-only and transformer-only approaches.
- Limitation: The approach is architecturally complex and may require more computational resources and careful tuning compared with simpler CNN baselines.

### Paper 10: Transformer-Aided Skin Cancer Classification Using VGG19-Based Feature Encoding
- Author(s) and Year: Fallah H. Najjar, Zaid Nidhal Khudhair, Farhan Mohamed, Mohd Shafry Mohd Rahim, Vei Siang Chan, and Ali Hilal Ali (2025)
- Research Approach: Hybrid model VGG19-RSPDA-ViT, combining VGG19 feature extraction, rotation/shift augmentation, and a Vision Transformer for classification.
- Major Findings: The hybrid model achieved 97.9% accuracy on MSK10000, 97.1% on HAM10000, and 98.67% on PH2, with strong macro-averaged precision, recall, specificity, and F1-scores.
- Contribution: It effectively addresses orientation sensitivity and limited data by integrating transformation-aware feature augmentation into a CNN-Transformer architecture.
- Limitation: The model depends on a specialized augmentation strategy and may not generalize equally well without careful preprocessing and tuning across new datasets.

### Paper 11: Skin Cancer Segmentation and Classification Using Vision Transformer for Automatic Analysis in Dermatoscopy-Based Noninvasive Digital System
- Author(s) and Year: Galib Muhammad Shahriar Himel, Md. Masudul Islam, Kh. Abdullah Al-Aff, Shams Ibne Karim, and Md. Kabir Uddin Sikder (2024)
- Research Approach: Uses SAM-based segmentation and a Vision Transformer pipeline for benign vs. malignant skin lesion classification, testing several pretrained ViT variants.
- Major Findings: The best model achieved 96.15% classification accuracy, and the segmentation stage reached IoU of 96.01% and Dice coefficient of 98.14%.
- Contribution: It demonstrates the effectiveness of combining segmentation and transformer-based classification for automatic noninvasive skin cancer analysis.
- Limitation: The study mainly distinguishes benign and malignant lesions; it does not fully explore complex multiclass classification or broader clinical validation.

## 3. Comparative Analysis

| No. | Author(s) | Year | Research Focus | Method/Model | Dataset | Major Finding | Limitation |
|---|---|---|---|---|---|---|---|
| 1 | Jonathan Vieira et al. | 2025 | Benchmarking CNNs for skin lesion classification | 38 CNN architectures, transfer learning | HAM10000, ISIC 2019 | ConvNeXtXLarge achieved 87.62% accuracy and cross-dataset validation | Limited by standard tuning and not highly custom models |
| 2 | Qichen Su et al. | 2024 | Class imbalance handling | STGAN + CNN classification | HAM10000 | 98.23% accuracy with improved synthetic data generation | Limited external clinical validation |
| 3 | M. M. Musthafa et al. | 2024 | Optimized CNN classification | Optimized CNN + augmentation + checkpointing | HAM10000 | 97.78% accuracy, strong precision and recall | Single-dataset validation |
| 4 | Hamza Abu Owida et al. | 2025 | Segmentation and classification integration | Dual-branch EfficientNet + DenseNet + morphology features | HAM10000, PH2, ISIC 2016/2017/2018 | High Dice/IoU with strong generalization | Computationally heavy |
| 5 | Naveed Ahmad et al. | 2023 | Explainable multiclass recognition | Xception + ShuffleNet + feature fusion + Grad-CAM | ISIC2018, HAM10000 | 99.3% and 91.5% accuracy | Complex multi-stage pipeline |
| 6 | Dingguo Wang and Yudi Chen | 2026 | Transfer learning configuration study | ResNet50, DenseNet121, EfficientNet-B0 | HAM10000 | Full fine-tuning and basic augmentation best | Controlled setup, limited generalization testing |
| 7 | Talha Mahboob Alam et al. | 2022 | Imbalanced dataset handling | AlexNet, InceptionV3, RegNetY-320 | HAM10000 | RegNetY-320 achieved 91% accuracy and 0.95 ROC | Limited generalization discussion |
| 8 | Maryam Tahir et al. | 2023 | Multiclass cancer diagnosis | DSCC_Net + SMOTE Tomek | HAM10000, ISIC 2020, DermIS | 94.17% accuracy and 99.43% AUC | Limited real-world deployment evidence |
| 9 | Aswani Dogga et al. | 2026 | Hybrid transformer-graph model | ViT + GNN + Region-Adaptive Attention | HAM10000, ISIC, PH2 | Strong interpretability and generalization | Higher architecture complexity |
| 10 | Fallah H. Najjar et al. | 2025 | CNN-Transformer hybrid with augmentation | VGG19-RSPDA-ViT | MSK10000, HAM10000, PH2 | 97.9%, 97.1%, and 98.67% accuracy | Specialized augmentation needed |
| 11 | Galib Muhammad Shahriar Himel et al. | 2024 | Segmentation + ViT classification | SAM + ViT variants | HAM10000 | 96.15% accuracy, 96.01% IoU, 98.14% Dice | Mostly binary classification |

## 4. Overall Research Gap

Across the reviewed papers, several recurring limitations are evident. Most studies rely on benchmark datasets such as HAM10000, ISIC, and PH2, but many do not validate performance on truly diverse clinical datasets or real-world patient populations. Class imbalance remains a major challenge, although several papers use augmentation, GAN synthesis, or rebalancing methods to mitigate it. Additionally, a number of models achieve high accuracy but remain limited by complexity, poor interpretability, or weak cross-dataset generalization. Many methods also focus on binary or narrow multiclass categorization rather than comprehensive clinical deployment, and some do not sufficiently integrate segmentation, morphology, and classification in a unified framework. A further gap is the limited emphasis on resource-efficient models suitable for real-time or low-resource healthcare environments.

## 5. How My Proposed Research Addresses the Gap

The proposed research can address these challenges by combining a practical and optimized deep learning framework with strong class-balancing and validation strategies. It can focus on improving classification accuracy while maintaining computational efficiency, supporting multiclass classification in a clinically relevant setting, and evaluating performance on a realistic dataset. By using a carefully structured CNN or hybrid model, supported by data augmentation or synthetic balancing when needed, the study can reduce the overfitting problem, improve generalization, and increase robustness in imbalanced class scenarios. This approach also provides a clearer pathway toward a deployable, interpretable, and clinically useful skin cancer detection system.

## 6. Final Literature Review

The reviewed literature indicates that deep learning has become the dominant approach for automated skin cancer detection, with studies such as those by Musthafa et al. (2024), Tahir et al. (2023), and Vieira et al. (2025) demonstrating that CNN-based and transfer-learning-based models can achieve high diagnostic accuracy on benchmark dermoscopic datasets. Methods based on GAN augmentation, segmentation-aware pipelines, and hybrid CNN-Transformer architectures, including the works of Su et al. (2024), Abu Owida et al. (2025), and Najjar et al. (2025), show that improved feature extraction, class balancing, and explainability can significantly strengthen model performance. Several studies also emphasize the importance of addressing class imbalance, dataset variability, and low-contrast or noisy lesion images, which remain important challenges in dermatological imaging. Although these approaches differ in model design and evaluation strategy, they consistently show that deep learning can support early detection and reduce dependence on manual examination. However, a common limitation is the lack of strong clinical generalization, limited use of real-world deployment settings, and the tendency of high-performing models to become computationally complex or difficult to interpret. Therefore, there is a clear research gap for a model that balances accuracy, efficiency, robustness to class imbalance, and practical usability for real-world skin cancer diagnosis.

## 7. References

[1] J. Vieira, F. Mendonça, and F. Morgado-Dias, "Deep Learning Approaches for Skin Lesion Detection," Electronics, 2025.
[2] Q. Su, H. N. A. Hamed, M. A. Isa, X. Hao, and X. Dai, "A GAN-based Data Augmentation Method for Imbalanced Multi-class Skin Lesion Classification," IEEE Access, 2024.
[3] M. M. Musthafa, M. T. R., V. Kumar V., and S. Guluwadi, "Enhanced skin cancer diagnosis using optimized CNN architecture and checkpoints for automated dermatological lesion classification," BMC Medical Imaging, 2024.
[4] H. Abu Owida, I. Abd El-Fattah, S. Abuowaida, N. Alshdaifat, H. A. Mashagba, A. B. Abd Aziz, A. Alzoubi, S. Larguech, and S. S. Al-Bawri, "A deep learning-based dual-branch framework for automated skin lesion segmentation and classification via dermoscopic images," Scientific Reports, 2025.
[5] N. Ahmad, J. H. Shah, M. A. Khan, J. Baili, G. J. Ansari, U. Tariq, Y. J. Kim, and J.-H. Cha, "A novel framework of multiclass skin lesion recognition from dermoscopic images using deep learning and explainable AI," Frontiers in Oncology, 2023.
[6] D. Wang and Y. Chen, "Comparative Study of Transfer Learning Strategies for Multi-Class Skin Lesion Classification: Architectures, Fine-Tuning, and Data Augmentation," ICCK Journal of Image Analysis and Processing, 2026.
[7] T. M. Alam, K. Shaukat, W. A. Khan, I. A. Hameed, L. A. Almuqren, M. A. Raza, M. Aslam, and S. Luo, "An Efficient Deep Learning-Based Skin Cancer Classifier for an Imbalanced Dataset," Diagnostics, 2022.
[8] M. Tahir, A. Naeem, H. Malik, J. Tanveer, R. A. Naqvi, and S.-W. Lee, "DSCC_Net: Multi-Classification Deep Learning Models for Diagnosing of Skin Cancer Using Dermoscopic Images," Cancers, 2023.
[9] A. Dogga, S. R., and S. S., "Hybrid vision transformer and graph neural network model with region-adaptive attention for enhanced skin cancer prediction," Scientific Reports, 2026.
[10] F. H. Najjar, Z. N. Khudhair, F. Mohamed, M. S. M. Rahim, V. S. Chan, and A. H. Ali, "Transformer-aided skin cancer classification using VGG19-based feature encoding," Scientific Reports, 2025.
[11] G. M. S. Himel, M. M. Islam, K. A. Al-Aff, S. I. Karim, and M. K. U. Sikder, "Skin Cancer Segmentation and Classification Using Vision Transformer for Automatic Analysis in Dermatoscopy-Based Noninvasive Digital System," International Journal of Biomedical Imaging, 2024.

---

This document has been updated with evidence-based entries for all 11 selected PDF papers.
