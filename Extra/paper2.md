PAPER 2

Title:
[A GAN-based Data Augmentation Method for 
Imbalanced Multi-class Skin Lesion 
Classification]

Authors:
[QICHEN SU, HAZA NUZLY ABDULL HAMED, MOHD ADHAM ISA, XUE HAO, and  
XIN DAI]

Year:
[year]

Abstract:
[Skin  cancer  is  one  of  the  most  common  types  of  cancer  globally.  Despite  the  remarkable 
advancements of deep learning methods in computer vision, automatic diagnosis of skin diseases still faces 
challenges such as limited data and class imbalance. Generative Adversarial Networks (GANs), which can 
synthesize  realistic  data,  appear  as  an  alternative  to  mitigate  these  issues.  However,  for  imbalanced  data, 
unconditional GANs either generate uneven data distribution or neglect universal knowledge of the whole 
dataset,  while  state-of-the-art  (SOTA)  conditional  GANs  suffer  from  performance  degradation  due  to  the 
mode collapse of minority classes. This paper proposes a two-stage GAN-based method to synthesize fine-
grained  and  diverse  256×256  pixels  skin  lesion  images  for  the  imbalanced  dataset,  named  Self-Transfer 
GAN  (STGAN).  STGAN  first  learns  universal  knowledge  from  all  classes  then  transfers  this  shared 
knowledge  to  each  class  and  fuses  it  with  class-specific  knowledge  to  synthesize  high-quality  images. 
Furthermore, based on STGAN, a framework to enhance the classification performance is established. Both 
data generation and classification tasks are evaluated on HAM10000 dataset. In terms of Fréchet Inception 
Distance  (FID),  Inception  Score,  Precision,  and  Recall,  STGAN  improved  by  16%,  16%,  4%,  33% 
compared  with  SOTA  conditional  StyleGAN2.  For  classification,  the  STGAN-based  framework  achieves 
remarkable results, with  an Accuracy of  98.23%,  Sensitivity of 88.85%, Precision of 90.23%,  F1-score of 
89.48%, and Specificity of 98.34%.]

Conclusion:
[n  this  paper,  a  two-stage  GAN-based  data  augmentation 
method is proposed to enhance the classification performance 
of  imbalanced  skin  lesion  dataset.  The  proposed  method 
decouples  the  learning  of  universal  knowledge  and  class-
specific  knowledge  to  mitigate  mode  collapse  of  minority 
classes.  Additionally,  self-supervised  learning  is  introduced 
to  improve  sample  efficiency,  and  Freeze-D  is  applied  to 
enhance stability during the learning of class-specific 
knowledge. Based on the proposed GAN-based data 
augmentation method, a framework to enhance the 
classification performance is established. Experimental 
results  on  HAM10000  dataset  showing  that  the  proposed 
method  can  generate  diverse  and  visually-appealing  images 
and can provide additional information to improve the 
classification  results.  Accordingly,  this  study  can  aid  in  the 
development of automated skin disease diagnosis system and 
contribute to improving the survival rates of individuals with 
skin  cancer.  A  limitation  of  this  study  is  that  due  to  the 
presence  of  natural  artifacts,  such  as  hair  and  veins,  in  the original  training  images,  images  generated  by  the  proposed 
method also retain these artifacts. These artifacts may 
occlude the skin lesions and potentially cause the degradation 
of  the  classification  performance.  As  a  future  direction,  we 
intend  to  explore  the  use  of  image  inpainting  techniques  to 
remove  these  natural  artifacts  before  training  and  further 
boost  the  performance  of  GAN-based  data  augmentation  in 
CNN-based classification. ]


Write a concise and well-structured Literature Review for my research topic in 1 paragraph of 10–15 lines. Summarize the most relevant previous studies and research works related to the topic, mentioning the authors/researchers, year of publication, methods or technologies used, datasets (if applicable), and major findings/results. Briefly compare the existing approaches and highlight their important contributions. Finally, identify the research gap or limitation in the existing studies and explain how my proposed research addresses or improves upon that gap. Use formal academic English, maintain a clear logical flow, avoid unnecessary details, and do not include fabricated references or information.`