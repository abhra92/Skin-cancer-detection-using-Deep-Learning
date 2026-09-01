# Skin Cancer Detection Using Deep Learning: Research Paper Collection

## PAPER 1

Title: Comparative Study of Transfer Learning Strategies for Multi-Class Skin Lesion Classification: Architectures, Fine-Tuning, and Data Augmentation

Authors: Dingguo Wang and Yudi Chen

Year: 2026

Abstract: This study evaluates three CNN architectures (ResNet50, DenseNet121, and EfficientNet-B0), three fine-tuning strategies (full, partial, and classifier-only), and three augmentation intensities on the HAM10000 dataset. The paper examines which combination provides the best trade-off between accuracy and computational cost in multi-class skin lesion classification. The experimental analysis includes statistical tests, ablation studies, and evaluation of training stability. The findings show that full fine-tuning achieved the highest validation accuracy, while basic augmentation produced the best practical performance. Aggressive augmentation was found to degrade performance because it distorted key medical image features.

Conclusion: The paper concludes that transfer learning configuration is critical for skin lesion classification and that architecture choice, fine-tuning strategy, and augmentation intensity must be balanced carefully. It provides practical guidance for building reliable deep learning workflows for medical image classification, particularly when working with limited and imbalanced data.

---

## PAPER 2

Title: A novel framework of multiclass skin lesion recognition from dermoscopic images using deep learning and explainable AI

Authors: Naveed Ahmad, Jamal Hussain Shah, Muhammad Attique Khan, Jamel Baili, Ghulam Jillani Ansari, Usman Tariq, Ye Jin Kim, and Jae-Hyuk Cha

Year: 2023

Abstract: This paper proposes a multiclass framework for skin lesion recognition that combines data augmentation, deep transfer learning, and explainable AI. Two pretrained models, Xception and ShuffleNet, are fine-tuned and fused to extract deep features. The most important features are then selected using an improved butterfly optimization algorithm before classification. Grad-CAM is used to visualize the regions contributing to model predictions. The method is evaluated on ISIC2018 and HAM10000 datasets and reports strong performance, with accuracy reaching 99.3% on ISIC2018 and 91.5% on HAM10000.

Conclusion: The study concludes that integrating data augmentation, feature fusion, optimization-based feature selection, and explainable AI can improve multiclass skin lesion recognition. The model demonstrates that deep learning can be made more accurate and interpretable for dermatological diagnosis.

---

## PAPER 3

Title: A deep learning-based dual-branch framework for automated skin lesion segmentation and classification via dermoscopic images

Authors: Hamza Abu Owida, Ibrahim Abd El-Fattah, Suhaila Abuowaida, Nawaf Alshdaifat, Hamza A. Mashagba, Azlan B. Abd Aziz, Alaa Alzoubi, Samia Larguech, and Samir Salem Al-Bawri

Year: 2025

Abstract: This study develops a dual-branch deep learning architecture for simultaneous skin lesion segmentation and classification. The segmentation branch uses a modified EfficientNet-B7 encoder with Atrous Spatial Pyramid Pooling and transformer blocks, while the classification branch fuses DenseNet-121 features with morphological information extracted from predicted segmentation masks. The method is evaluated on HAM10000, PH2, ISIC 2016, ISIC 2017, and ISIC 2018 datasets. It achieves high segmentation performance and strong classification accuracy, demonstrating the value of combining appearance information with lesion morphology.

Conclusion: The authors conclude that an integrated segmentation-classification framework improves the clinical usefulness of automated skin lesion analysis. The method shows strong generalization and promising performance for diagnostic support in skin cancer screening.

---

## PAPER 4

Title: Enhanced skin cancer diagnosis using optimized CNN architecture and checkpoints for automated dermatological lesion classification

Authors: M. Mohamed Musthafa, Mahesh T. R., Vinoth Kumar V., and Suresh Guluwadi

Year: 2024

Abstract: This paper uses the HAM10000 dataset to propose a CNN-based skin lesion classification model tailored for the challenges of skin cancer diagnosis. The model incorporates multiple convolutional, pooling, and dense layers, class balancing through augmentation, and model checkpointing to preserve the best training state. The proposed method achieved high performance metrics, including 97.78% accuracy, 97.9% precision, 97.9% recall, and 97.8% F2-score. The study emphasizes the importance of architecture design and careful training strategy in dermatological image analysis.

Conclusion: The paper concludes that an optimized CNN architecture with augmentation and checkpointing can produce highly accurate and reliable results for automated skin cancer diagnosis. The model supports early detection and can assist clinicians in decision-making.

---

## PAPER 5

Title: Hybrid vision transformer and graph neural network model with region-adaptive attention for enhanced skin cancer prediction

Authors: Aswani Dogga, Sivasubramanian R., and Shanthi S.

Year: 2026

Abstract: This paper presents a hybrid model that combines a Vision Transformer (ViT) and a Graph Neural Network (GNN) with Region-Adaptive Attention for enhanced skin cancer prediction. The ViT branch captures global contextual dependencies, while the GNN models spatial relationships between lesion regions. Region-Adaptive Attention further emphasizes clinically relevant regions. The framework is evaluated on benchmark skin cancer datasets and aims to improve both classification performance and interpretability. The paper highlights the complementary strengths of ViTs and GNNs in medical imaging.

Conclusion: The authors conclude that the hybrid ViT-GNN design offers a more robust and interpretable solution for skin cancer prediction than standalone CNN or transformer models. The framework addresses key limitations in lesion context modeling and feature selection.

---

## PAPER 6

Title: Transformer-aided skin cancer classification using VGG19-based feature encoding

Authors: Fallah H. Najjar, Zaid Nidhal Khudhair, Farhan Mohamed, Mohd Shafry Mohd Rahim, Vei Siang Chan, and Ali Hilal Ali

Year: 2025

Abstract: This paper proposes a hybrid model named VGG19-RSPDA-ViT, which combines VGG19-based local feature extraction with a Vision Transformer. The model incorporates rotation and shift patch data augmentation (RSPDA) to increase robustness to orientation changes and improve generalization on small datasets. It is evaluated on three benchmark datasets: MSK10000, HAM10000, and PH2. The model achieves high accuracy across binary and multiclass classification settings, outperforming existing baselines and demonstrating strong generalization capability.

Conclusion: The paper concludes that combining CNN-based local feature encoding with transformer-based global attention is effective for skin lesion classification. The inclusion of RSPDA also helps address orientation sensitivity and limited data availability.

---

## PAPER 7

Title: An Efficient Deep Learning-Based Skin Cancer Classifier for an Imbalanced Dataset

Authors: Talha Mahboob Alam, Kamran Shaukat, Waseem Ahmad Khan, Ibrahim A. Hameed, Latifah Abd. Almuqren, Muhammad Ahsan Raza, Memoona Aslam, and Suhuai Luo

Year: 2022

Abstract: This paper addresses the issue of class imbalance in skin cancer datasets by proposing a deep learning framework evaluated on the Skin Cancer MNIST: HAM10000 dataset. Several models, including AlexNet, InceptionV3, and RegNetY-320, are trained and compared. Data augmentation is used to balance the classes and improve model learning. The proposed framework achieves better accuracy, F1-score, and ROC values than previous state-of-the-art methods, with RegNetY-320 outperforming the other architectures.

Conclusion: The study confirms that class imbalance is a major challenge in skin lesion classification and that careful balancing strategies can significantly improve performance. RegNetY-320 is identified as the best-performing architecture in the evaluated setting.

---

## PAPER 8

Title: DSCC_Net: Multi-Classification Deep Learning Models for Diagnosing of Skin Cancer Using Dermoscopic Images

Authors: Maryam Tahir, Ahmad Naeem, Hassaan Malik, Jawad Tanveer, Rizwan Ali Naqvi, and Seung-Won Lee

Year: 2023

Abstract: This paper introduces DSCC_Net, a CNN-based deep learning model for multi-class skin cancer diagnosis. The framework is evaluated on three public datasets: ISIC 2020, HAM10000, and DermIS. SMOTE Tomek is applied to address class imbalance, and the model is compared with six baseline networks including ResNet-152, VGG-16, VGG-19, MobileNet, EfficientNet-B0, and Inception-V3. The proposed DSCC_Net achieves strong classification metrics, including 94.17% accuracy, 93.76% recall, 94.28% precision, 93.93% F1-score, and 99.43% AUC.

Conclusion: The paper concludes that DSCC_Net offers a strong multi-class classification solution for skin cancer and outperforms several common baseline models. It demonstrates the importance of model design and class balancing in diagnostic accuracy.

---

## PAPER 9

Title: Deep Learning Approaches for Skin Lesion Detection

Authors: Jonathan Vieira, Fábio Mendonça, and Fernando Morgado-Dias

Year: 2025

Abstract: This work benchmarks 38 CNN architectures from ten families using transfer learning on the HAM10000 dataset for seven-class skin lesion classification. The study evaluates models with frozen pretrained weights and then performs cross-database validation on ISIC 2019 to assess generalization. ConvNeXtXLarge performs best among the tested models and demonstrates notable robustness. The paper also presents a client-server mobile application prototype that transmits dermoscopic images to a server for classification.

Conclusion: The paper concludes that deep learning architectures can be useful for both research and real-world screening tools, but model selection and validation across datasets remain essential. The study also highlights the practical value of bringing AI-based diagnostics into mobile healthcare workflows.

---

## PAPER 10

Title: A GAN-based Data Augmentation Method for Imbalanced Multi-class Skin Lesion Classification

Authors: Qichen Su, Haza Nuzly Abdull Hamed, Mohd Adham Isa, Xue Hao, and Xin Dai

Year: 2024

Abstract: This paper addresses the problem of imbalanced multi-class skin lesion classification by proposing a two-stage GAN-based method called STGAN. The approach learns shared global knowledge first and then combines it with class-specific knowledge to generate realistic skin lesion images. The method is evaluated on HAM10000 using Fréchet Inception Distance, Inception Score, Precision, Recall, and classification metrics. The proposed framework demonstrates significant improvement over conditional StyleGAN2 and achieves high classification accuracy.

Conclusion: The authors conclude that STGAN effectively mitigates class imbalance and improves skin lesion classification by generating more diverse, high-quality synthetic training data. The study emphasizes the potential of GAN-based augmentation in medical image analysis.

---

## PAPER 11

Title: Skin Cancer Segmentation and Classification Using Vision Transformer for Automatic Analysis in Dermatoscopy-Based Noninvasive Digital System

Authors: Galib Muhammad Shahriar Himel, Md. Masudul Islam, Kh. Abdullah Al-Aff, Shams Ibne Karim, and Md. Kabir Uddin Sikder

Year: 2024

Abstract: This paper proposes a noninvasive dermatoscopy-based system for automatic skin cancer analysis using Vision Transformer-based deep learning. The study uses the HAM10000 dataset, which contains 10,015 skin lesion images classified as benign and malignant. The authors apply preprocessing, normalization, and augmentation before training several transformer-based models. A Segment Anything Model (SAM) is used for lesion segmentation, achieving an IoU of 96.01% and a Dice coefficient of 98.14%. Among six models tested, ViT-Google with patch-32 achieved the best performance, reaching 96.15% classification accuracy with a low false negative rate.

Conclusion: The study concludes that Vision Transformer-based models can perform strongly in skin cancer classification when combined with segmentation and careful hyperparameter tuning. The proposed system is presented as a practical digital diagnostic support tool that can assist dermatologists and enable early detection using dermoscopic images.

---

## Additional Notes

This document compiles the research papers in the requested format and includes the extracted details for Paper 11. The entries are written in a concise academic style and are based on the information available in the supplied PDFs.
