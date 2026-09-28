PAPER 1

Title:
[Skin Cancer Segmentation and Classification Using Vision
Transformer for Automatic Analysis in Dermatoscopy-Based
Noninvasive Digital System]

Authors:
[Galib Muhammad Shahriar Himel , Md. Masudul Islam , Kh. Abdullah Al-Aff ,
Shams Ibne Karim , and Md. Kabir Uddin Sikder ]

Year:
[2024]

Abstract:
[1. Introduction
Cancer is a condition that arises when cells undergo uncon-
trolled division and extend into nearby tissues. The develop-
ment of cancer is triggered by alterations and mutations in
the DNA. The majority of DNA changes responsible for can-
cer occur within specific regions known as genes. Among the
various types of cancers, skin cancer is among the five on the
list. If we disregard breast and prostate cancers which are
gender-dependent, skin cancer will remain in the third larg-
est cancer category among many others. Based on the statis-
tics released by the American Cancer Society (ACS) [1],
there were 58,120 recorded cases of skin cancer among
males and 39,490 cases among females. An intriguing obser-
vation is that the incidence of skin cancer has been steadily
rising from 1992 to 2019, with a notable exception in 2020
Hindawi
International Journal of Biomedical Imaging
Volume 2024, Article ID 3022192, 18 pages
https://doi.org/10.1155/2024/3022192
[2]. This exception can be attributed to the understandable
decrease in cases during the COVID-19 pandemic, as people
were mostly confined to their homes. This decline is reason-
able considering that exposure to ultraviolet (UV) radiation
is a significant contributing factor to the development of
skin cancer.
More people are diagnosed with skin cancer each year in
the U.S. than all other cancers combined [3]. More than
5,400 people worldwide die of nonmelanoma skin cancer
every month [4]. The number of melanoma deaths is
expected to increase by 4.4 percent in 2023 [3]. More people
develop skin cancer because of indoor tanning than develop
lung cancer because of smoking [5]. Skin cancer represents
approximately 2 to 4 percent of all cancers in Asians, 4 to
5 percent of all cancers in Hispanics, and 1 to 2 percent of
all cancers in Black people [6–8]. Skin cancers account for
3 percent of pediatric cancers [9].
There are various factors involving the initiation of
developing skin cancer such as ultraviolet rays, arsenic
consumption, cigarette smoking, exposure to nuclear radi-
ation, excessive X-ray exposure, indoor tanning, drinking
alcohol, viruses, changing environments, sunburn, abnor-
mal swelling, eczema, weak immune system, and human
papilloma virus. There are also environmental factors that
work as a catalyst for developing skin cancer such as
working coal, tar, petroleum, and shale oils [10]. Some dis-
eases may fasten the development rate of skin cancer but
not necessarily the root cause of cancer themselves such
as AIDS and other diseases that involve the weakening
of the immune system.
Skin cancer may form on the upper part of the skin
(which is visible to us) as well as in the inner parts of differ-
ent layers of the skin. More than 90% of skin cancers are
caused by direct sunlight exposure, to be specific, UV expo-
sure. Almost all of the cases are related to cancer develop-
ment on the upper visible part of the skin. The affected
skin areas can be easily captured by a dermatoscope or
high-resolution smartphone camera. On the other hand,
skin cancers that develop inside the skin layers require inva-
sive sample collection methods. Since more than 90% of skin
cancer is caused by UV rays and these cancers occur primar-
ily on the surface of the skin through UV exposure, it can be
said that the majority of skin cancers can be identified
through noninvasive methods using a dermatoscope, as it
allows for the collection of samples from the skin.
Traditionally, the ABCDE [11] method is used for the
primary diagnosis (classification) of skin cancer. The
ABCDE acronym is used as a mnemonic for recognizing
potential signs of melanoma, a type of skin cancer that is
shown in Figure 1. Each letter corresponds to a characteristic
feature that may indicate the presence of melanoma. Here is
the breakdown of the ABCDE criteria:
(i) A—asymmetry: Melanomas often exhibit irregular
or asymmetrical shapes, where one half does not
match the other half
(ii) B—border irregularity: The borders of a melanoma
may be uneven, ragged, or notched, rather than
smooth and well-defined
(iii) C—color variation: Melanomas can have a range of
colors within the same lesion, such as different
shades of brown, black, blue, red, or white
(iv) D—diameter: While melanomas can be smaller, any
mole or lesion with a diameter larger than 6 milli-
meters (about the size of a pencil eraser) should be
closely examined
(v) E—evolution or changes over time: Watch out for
any changes in size, shape, color, elevation, or other
characteristics of a mole or lesion
After the primary classification, the sample is transferred
to the pathological analysis team for a definitive result. The
pathological analysis is an invasive method. This method
can be used for classifying all kinds of cancer cells. However,
it has its disadvantages. This method is very costly, time-
consuming, and painful. Pathological diagnosis of skin can-
cer typically involves an invasive method known as a biopsy.
The biopsy procedure is performed by a healthcare profes-
sional, usually a dermatologist or a surgeon, and it involves
the removal of a small sample of suspicious skin tissue for
further examination.
There are different types of skin biopsies, including the
following:
(i) Excisional biopsy: This type of biopsy involves the
complete removal of the suspicious skin lesion,
along with a small margin of surrounding healthy
tissue. The excised sample is then sent to a pathol-
ogy laboratory for analysis
(ii) Incisional biopsy: In this method, only a portion of
the suspicious skin lesion is removed for examina-
tion. It is typically done when the lesion is large or
deep, and the entire lesion cannot be easily excised
(iii) Punch biopsy: A punch biopsy involves using a spe-
cial tool called a punch to remove a small circular
piece of skin tissue, including the suspicious lesion
and a small portion of normal skin around it
(iv) Shave biopsy: This method involves shaving offthe
top layers of the skin using a surgical blade to obtain
a sample. It is commonly used for superficial lesions
or those located on the surface of the skin
Asymmetry Borders Color Diameter Evolution
6 mm
Figure 1: The ABCDE method for primary diagnosis of skin cancer.
2 International Journal of Biomedical Imaging 2973, 2024, 1, Downloaded from https://onlinelibrary.wiley.com/doi/10.1155/2024/3022192, Wiley Online Library on [31/08/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License
Once the skin sample is obtained through the biopsy
procedure, it is sent to a pathology laboratory, where it
undergoes microscopic examination by a pathologist. The
pathologist examines the tissue sample, looking for charac-
teristic features of skin cancer, such as abnormal cell growth
patterns, cellular atypia, and invasion of surrounding tissues.
Based on the examination findings, the pathologist provides
a de finitive diagnosis and may further classify the specific
type and stage of skin cancer.
Due to the time-consuming nature of even the primary
diagnosis process used in the conventional method, it
often becomes very difficult to identify cancer within the
target time. At the same time, handling a lot of patients
within the expected timeframe becomes a nightmare. From
the statistics, it is also clear that day by day, skin cancer
patients are increasing rapidly as a result of the depression
of the ozone layer creating an inability to block the UV
rays. So, soon, the hospital and the diagnosis centers will
face more difficulties in managing the situation in due
time. To solve this issue, artificial intelligence can play a
vital role. Using a dermatoscope, the suspected areas of
the skin can be captured at up to 20x magnification very
easily. These images can be used to train an intelligent sys-
tem that can classify cancer cells accurately. Computers,
smartphones, or any other smart devices can be used to
do the task.
In this paper, we are proposing a vision transfer-based
machine learning model that will classify the cancerous cells
on the upper part of the skin. This model can be mounted on
any smart device which can be used to capture images using
a dermatoscope and instantly generate a classification deci-
sion regarding the suspected area. In Figure 2, the workflow
of rapid cancer cell classification is shown.
In this research, the HAM10000 [12] dataset is used for
training the model. At first, we manually annotated the sus-
pected areas and then applied the Segment Anything Model
(SAM) [13] to train the segmentation model. This model
will be used to detect the specific area using the concept of
semantic segmentation. In the second stage, the classifica-
tion model will be used to differentiate between the benign
and malignant categories. The major contribution of our
research:
(i) Creating the segmentation model using the Segment
Anything Model (SAM)
(ii) For classification, we have used Google’s ViT model
(patch 32) [14], Visual Attention Network (VAN)
[15], ResNet50 [14], Facebook’s MAE model [16],
Document Image Transformer (DiT) [17], and
BERT pretraining of image transformer (BEiT)
[18] to train the vision transformer model. Compar-
ing these models, we have found out that Google’s
ViT model (patch-32) generates better results than
the other models in skin cancer classification
(iii) We meticulously fine-tuned the hyperparameters
for our model, exploring various optimization strat-
egies, learning rates, and loss functions. Specifically,
we evaluated the performance under different opti-
mizers, including “Adam,”“Adamax,”“RMSProp,”
and “SGD.”The learning rates considered spanned
a range of values: 0.1, 0.01, 0.001, 0.0001, 0.00002,
and 0.00001. Additionally, we experimented with
different loss functions, namely, “Cross Entropy
Loss”and “Hinge.”After a thorough analysis, we
Patient
Machine
learning
model
DermatoscopeSample
Malignant
Rapid skin cancer
Forwarded for
Pathological
Diagnosis
Figure 2: Rapid cancer cell classification using dermatoscope.
3International Journal of Biomedical Imaging 2973, 2024, 1, Downloaded from https://onlinelibrary.wiley.com/doi/10.1155/2024/3022192, Wiley Online Library on [31/08/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License
determined that the most effective combination for
our vision transformer (ViT) model involved
employing the “Adam”optimizer with a learning
rate of 0.00002 and using “Cross Entropy Loss”for
calculating the loss. These optimized hyperpara-
meters were instrumental in achieving superior per-
formance and model efficiency
The following paper is structured in the following way:
Section 2 describes the existing literature on the topic and
comparison among them. Section 3 describes the methodol-
ogy of our works which includes dataset description, exper-
imental model, and experimental setup. Section 4 provides
various graphical representations of the result analysis and
a discussion of the challenging part of our study. Section 5
provides the concluding remarks.]

Conclusion:
[In this research, we utilize Google’s 32-patch-based vision
transformer (ViT) model to address the identification of skin
cancer. The prevalence of skin cancer has increased signifi-
cantly worldwide due to the depletion of the ozone layer,
posing a substantial threat to communities. Delayed testing
procedures, limited facilities, and a lack of early-stage diag-
nosis have led to numerous deaths. To solve this, intelligent
smart technologies are necessary to establish rapid and effec-
tive testing procedures. Our experimental approach involves
employing the ViT model in conjunction with the SAM seg-
mentation model for skin cancer identification and classifi-
cation. The application of the Segment Anything Model
(SAM) enables the segmentation of cancerous regions in
the images, yielding an Intersection over Union (IOU) of
96.01% and a Dice coefficient of 98.14%. Also, our classifica-
tion experiments with Google’s ViT model yield impressive
results, achieving 96.15% accuracy and 99.49% ROC score.
Consequently, we conclude that the ViT-Google (patch 32)
model can be trained as an efficient skin cancer detectionmodel using the vision transformer and implemented in
smart devices for swift detection, thereby providing valuable
support to pathologists. However, it is essential to acknowl-
edge that our ViT model-based method is more suitable for
fair-skinned individuals due to the dataset acquired from
Harvard University in America, where the predominant
population is white. Our future work will expand the dataset
to encompass a more diverse range of cases from individuals
of various ethnic backgrounds. Additionally, federated
learning can be employed to update the existing model with new data, allowing for iterative improvements without
retraining the entire model.]



Write a concise and well-structured Literature Review for my research topic in 1 paragraph of 10–20 lines. Summarize the most relevant previous studies and research works related to the topic, mentioning the authors/researchers, year of publication, methods or technologies used, datasets (if applicable), and major findings/results. Briefly compare the existing approaches and highlight their important contributions. Finally, identify the research gap or limitation in the existing studies and explain how my proposed research addresses or improves upon that gap. Use formal academic English, maintain a clear logical flow, avoid unnecessary details, and do not include fabricated references or information.