# Experiment 2 — Literature Review Using Digital Libraries

## 1. Purpose

This guide explains **exactly how to complete Experiment No. 2** for the research project:

> **Skin Cancer Detection Using Deep Learning**

The goal is to find relevant research papers, download them, collect specific sections, provide those sections to an LLM, and generate a properly structured literature review.

---

## 2. Experiment Details

| Field | Details |
|---|---|
| Experiment No. | 2 |
| Title | Literature Review using Digital Libraries |
| Learning Outcome | Search and review research papers using databases such as Google Scholar, IEEE Xplore, Springer, and ACM Digital Library. |
| Research Topic | Skin Cancer Detection Using Deep Learning |

---

# 3. What You Have to Do

The complete workflow is:

**Search → Select Papers → Download Papers → Read Papers → Extract Information → Give Information to LLM → Generate Literature Review → Verify → Finalize**

### Minimum recommended papers

For a good practical submission, collect **5–10 relevant research papers**.

Prefer papers that are:

- Directly related to skin cancer or skin lesion detection/classification.
- Related to Deep Learning or Computer Vision.
- Published in a recognized journal/conference.
- Preferably recent, especially **2020–2026**.
- Based on identifiable datasets and methods.
- Useful for identifying limitations or research gaps.

---

# 4. Step 1 — Understand Your Research Topic

Your topic is:

> **Skin Cancer Detection Using Deep Learning**

Break it into search concepts:

| Concept | Keywords |
|---|---|
| Disease | Skin cancer, melanoma, skin lesion |
| Task | Detection, classification, diagnosis |
| Technology | Deep learning, machine learning, computer vision |
| Models | CNN, ResNet, DenseNet, EfficientNet, VGG, MobileNet, Vision Transformer |
| Dataset | HAM10000, ISIC, PH2, Derm7pt |

Do not search only:

> `skin cancer`

That is too broad.

Instead, use combinations such as:

> `skin cancer detection deep learning`

> `skin lesion classification CNN`

> `melanoma detection deep learning`

> `skin cancer classification transfer learning`

---

# 5. Step 2 — Search for Research Papers

You can use:

1. Google Scholar
2. IEEE Xplore
3. SpringerLink
4. ACM Digital Library
5. PubMed
6. ScienceDirect

For this experiment, your teacher specifically mentioned:

- Google Scholar
- IEEE Xplore
- Springer
- ACM Digital Library

## Recommended Search Queries

Copy and use these searches:

```text
"skin cancer detection" AND "deep learning"
```

```text
"skin lesion classification" AND CNN
```

```text
"skin cancer classification" AND "transfer learning"
```

```text
"melanoma detection" AND "deep learning"
```

```text
"skin lesion" AND "convolutional neural network"
```

```text
"skin cancer" AND EfficientNet
```

```text
"skin lesion classification" AND ResNet
```

```text
"skin cancer" AND "Vision Transformer"
```

---

# 6. Step 3 — Select Relevant Papers

Do not download every paper you find.

Check the following before selecting a paper:

### Selection Checklist

- [ ] Is the paper directly related to the research topic?
- [ ] Does it use Deep Learning or Computer Vision?
- [ ] Does it discuss skin cancer or skin lesion detection/classification?
- [ ] Is the paper from a credible academic source?
- [ ] Is the publication year available?
- [ ] Are the authors clearly listed?
- [ ] Is the abstract available?
- [ ] Is the conclusion available?
- [ ] Does the paper provide useful methodology/results?
- [ ] Can the paper help identify a research gap?

If most answers are **Yes**, keep the paper.

---

# 7. Step 4 — Download the Research Paper

Whenever possible, download the **full PDF** from the official publisher or a legitimate academic source.

### Typical process

1. Open the research paper.
2. Verify the title.
3. Verify the authors.
4. Check the publication year.
5. Look for a **PDF / Download PDF / Full Text** option.
6. Download the PDF.
7. Rename it clearly.

### Recommended file naming

```text
01_Author_Year_ShortTitle.pdf
02_Author_Year_ShortTitle.pdf
03_Author_Year_ShortTitle.pdf
```

Example:

```text
01_Smith_2023_SkinCancerCNN.pdf
```

---

# 8. Step 5 — Create a Paper Collection Folder

Create one main folder:

```text
Skin_Cancer_Literature_Review/
│
├── Research_Papers/
│   ├── 01_Paper.pdf
│   ├── 02_Paper.pdf
│   ├── 03_Paper.pdf
│   ├── 04_Paper.pdf
│   └── 05_Paper.pdf
│
├── Extracted_Content/
│   ├── Paper_01.txt
│   ├── Paper_02.txt
│   └── Paper_03.txt
│
├── Literature_Review/
│   └── Literature_Review.md
│
└── References/
    └── References.md
```

This makes it much easier for the entire group to work together.

---

# 9. Step 6 — Extract Information from Every Paper

For **each paper**, collect the following:

1. Paper Title
2. Authors
3. Publication Year
4. Abstract
5. Dataset
6. Methodology
7. Deep Learning Model
8. Preprocessing
9. Data Augmentation
10. Results/Evaluation Metrics
11. Main Contribution
12. Conclusion
13. Limitations
14. DOI/Publisher information, if available

### Important

Your teacher specifically asked you to provide:

- **Paper Name/Title**
- **Author Name**
- **Abstract**
- **Conclusion**

These four items are the minimum required input.

The additional information is recommended because it makes the literature review much better.

---

# 10. Step 7 — Prepare Each Paper in a Standard Format

Use exactly the same format for every paper.

```text
PAPER 01

Title:
[Complete paper title]

Authors:
[Complete author names]

Publication Year:
[Year]

Abstract:
[Paste the abstract from the paper]

Dataset:
[Dataset name, if mentioned]

Method/Model:
[Model or methodology, if mentioned]

Results:
[Important results, if mentioned]

Conclusion:
[Paste the conclusion from the paper]

DOI/Publisher:
[DOI or publisher information, if available]
```

Repeat this for Paper 02, Paper 03, etc.

---

# 11. Step 8 — Create a Master Research Table

Before generating the final literature review, maintain a table.

| No. | Authors | Year | Research Focus | Model | Dataset | Major Finding | Limitation |
|---|---|---:|---|---|---|---|---|
| 1 | Author(s) | 2023 | Skin lesion classification | CNN | HAM10000 | Finding | Limitation |
| 2 | Author(s) | 2022 | Melanoma detection | ResNet | ISIC | Finding | Limitation |
| 3 | Author(s) | 2024 | Skin cancer classification | EfficientNet | ISIC | Finding | Limitation |
| 4 | Author(s) | 2021 | Lesion classification | DenseNet | HAM10000 | Finding | Limitation |
| 5 | Author(s) | 2025 | Skin lesion detection | Transformer | ISIC | Finding | Limitation |

**Do not fill this table with guesses.**

If a paper does not report something, write:

> `Not Reported`

---

# 12. Step 9 — Give the Papers to an LLM

Now take the information collected from all papers and provide it to an LLM.

You can use ChatGPT or another suitable LLM.

## Copy This Universal Prompt

```text
Act as an academic research assistant specializing in Deep Learning, Computer Vision, and Medical Image Analysis.

I am conducting Experiment No. 2:

Title:
"Literature Review using Digital Libraries"

Learning Outcome:
"Search and review research papers using databases such as Google Scholar, IEEE Xplore, Springer, and ACM Digital Library."

My research topic is:

"Skin Cancer Detection Using Deep Learning"

I will provide information collected directly from multiple research papers.

For each paper, I will provide:

- Paper Title
- Author(s)
- Publication Year
- Abstract
- Dataset
- Method/Model
- Results
- Conclusion
- DOI/Publisher information, if available

Your task is to analyze the provided papers and prepare an academic Literature Review.

For each research paper:

1. Identify the author(s) and publication year.
2. Explain the main research objective.
3. Summarize the methodology or deep learning approach.
4. Mention the dataset used, if provided.
5. Mention the model/architecture used, if provided.
6. Summarize the major findings/results.
7. Explain the main contribution.
8. Identify limitations or research gaps explicitly mentioned in the paper.
9. Do not invent missing information.

After reviewing all papers, compare them and identify:

- Common approaches
- Common datasets
- Common deep learning models
- Important findings
- Strengths of existing approaches
- Limitations of existing approaches
- Common challenges
- Research gaps

Pay particular attention to:

- Class imbalance
- Dataset limitations
- Image quality
- Overfitting
- Dataset bias
- Lack of diverse images
- Model interpretability
- Computational cost
- Generalization to real-world clinical images

Create the following output:

## 1. Introduction

Write a short introduction explaining the importance of research on skin cancer detection using deep learning.

## 2. Review of Individual Research Papers

For every paper, provide:

### Paper 1 — [Author(s), Year]

- Research Objective:
- Dataset:
- Method/Model:
- Major Findings:
- Contribution:
- Limitation:

Repeat for all papers.

## 3. Comparative Literature Review

Create a table:

| No. | Author(s) | Year | Research Focus | Model | Dataset | Major Finding | Limitation |

## 4. Critical Analysis

Compare the research papers instead of merely listing them.

Explain:
- Which approaches are commonly used?
- Which models appear effective?
- Which datasets are frequently used?
- What limitations repeatedly occur?
- What problems remain unresolved?

## 5. Research Gap

Identify a clear research gap based ONLY on the papers provided.

Do not invent a research gap without evidence from the papers.

## 6. Proposed Research Direction

Explain how a proposed project on "Skin Cancer Detection Using Deep Learning" could address the identified research gap.

## 7. Final Literature Review

Write a concise, well-connected academic literature review in ONE paragraph of approximately 5–8 lines.

The paragraph must:
- Mention important authors and years.
- Summarize major approaches.
- Mention datasets/models where relevant.
- Compare existing approaches.
- Mention important findings.
- End with the research gap and how the proposed research can address it.

## 8. References

Format the provided papers in IEEE reference style.

IMPORTANT RULES:

- Use ONLY the information provided in the papers.
- Do NOT fabricate authors, years, datasets, models, accuracy values, results, DOI numbers, or citations.
- If information is missing, write "Not Reported".
- Do NOT simply copy the abstracts and conclusions into the literature review.
- Paraphrase and synthesize the research.
- Clearly identify which author/year is associated with each study.
- Preserve the meaning of the original research.
- Maintain an academic and objective tone.
- Make the final work suitable for a BCA research project.
```

---

# 13. Step 10 — Paste Paper Information

After the prompt, paste your papers.

Use this format:

```text
==============================
PAPER 01
==============================

Title:
[Title]

Authors:
[Authors]

Year:
[Year]

Abstract:
[Abstract]

Dataset:
[Dataset]

Method/Model:
[Method/Model]

Results:
[Results]

Conclusion:
[Conclusion]


==============================
PAPER 02
==============================

Title:
[Title]

Authors:
[Authors]

Year:
[Year]

Abstract:
[Abstract]

Dataset:
[Dataset]

Method/Model:
[Method/Model]

Results:
[Results]

Conclusion:
[Conclusion]
```

Continue until all papers have been added.

---

# 14. Step 11 — Verify the LLM Output

This step is **mandatory**.

LLMs can make mistakes even when the source material is provided.

Compare the generated literature review against the original papers.

### Verification Checklist

- [ ] Author names are correct.
- [ ] Publication years are correct.
- [ ] Paper titles are correct.
- [ ] Dataset names are correct.
- [ ] Models are correct.
- [ ] Reported results are correct.
- [ ] Conclusions are represented accurately.
- [ ] No fabricated information was added.
- [ ] Research gaps are supported by the papers.
- [ ] References correspond to the correct papers.

---

# 15. Step 12 — Identify the Research Gap

Your final literature review should not stop at:

> "Many researchers used CNN and achieved high accuracy."

That is not a research gap.

Look for repeated limitations such as:

```text
Existing Studies
       ↓
CNN / Transfer Learning / Transformers
       ↓
Good Classification Performance
       ↓
BUT
       ↓
Class Imbalance
Dataset Bias
Limited Diversity
Overfitting
Poor Generalization
Limited Explainability
       ↓
RESEARCH GAP
       ↓
Need for a more robust and generalizable approach
```

The exact research gap must come from the papers you reviewed.

---

# 16. Step 13 — Write the Final Literature Review

A good final literature review should follow this logic:

```text
Author + Year
      ↓
What they did
      ↓
Dataset / Model
      ↓
Main Finding
      ↓
Limitation
      ↓
Comparison with other studies
      ↓
Research Gap
      ↓
Your Proposed Research
```

### Example Structure

> Previous studies have explored deep learning techniques for automated skin lesion and skin cancer classification using datasets such as HAM10000 and ISIC. [Author, Year] employed [model] and reported [finding], while [Author, Year] investigated [different approach]. Transfer learning architectures such as [models] have demonstrated strong classification performance, although several studies have reported challenges related to class imbalance, dataset diversity, and generalization. More recent approaches have explored advanced architectures to improve classification and interpretability. However, limitations in dataset representativeness and real-world generalization remain important concerns. Therefore, the proposed research focuses on developing and evaluating a deep-learning-based skin cancer classification approach while addressing the identified limitations.

**Do not copy this example as your final literature review. Replace it with findings from your actual papers.**

---

# 17. How to Divide the Work Among Group Members

If you are the group mentor, divide the work.

### Example for a 5-member group

| Member | Responsibility |
|---|---|
| Member 1 | Find Papers 1–2 |
| Member 2 | Find Papers 3–4 |
| Member 3 | Find Papers 5–6 |
| Member 4 | Extract abstracts, conclusions, datasets and models |
| Group Mentor | Combine information, run LLM analysis, verify and finalize |

### Better workflow

Each member should submit:

```text
Paper PDF
+
Paper Information
+
Source Link/DOI
+
Extracted Abstract
+
Extracted Conclusion
```

Then the mentor combines everything into one master document.

---

# 18. Recommended Google Drive Structure

If your group is collaborating through Google Drive:

```text
SKIN CANCER RESEARCH
│
├── 01_Research_Papers
│
├── 02_Paper_Information
│
├── 03_Literature_Table
│
├── 04_LLM_Output
│
├── 05_Verification
│
└── 06_Final_Submission
```

Use consistent filenames:

```text
Paper_01.pdf
Paper_02.pdf
Paper_03.pdf
```

And:

```text
Paper_01_Information.docx
Paper_02_Information.docx
Paper_03_Information.docx
```

---

# 19. What NOT to Do

### Do NOT:

- Download random blogs instead of research papers.
- Copy a literature review from another student's project.
- Invent accuracy values.
- Invent datasets.
- Invent authors or publication years.
- Ask an LLM to create fake references.
- Use only one research paper.
- Use only papers with positive results.
- Treat "high accuracy" as the only measure of a good study.
- Submit an AI-generated literature review without checking it against the papers.

### Especially avoid this prompt:

```text
Write a literature review on skin cancer detection using deep learning.
```

This is too vague. It can produce generic or fabricated references.

Instead, **provide the actual papers and source information to the LLM.**

---

# 20. Final Submission Checklist

Before submitting Experiment 2, make sure you have:

### Research Papers

- [ ] 5–10 research papers downloaded.
- [ ] Papers are relevant to the research topic.
- [ ] Sources/databases are recorded.

### Paper Information

- [ ] Title
- [ ] Authors
- [ ] Year
- [ ] Abstract
- [ ] Dataset
- [ ] Model/Method
- [ ] Results
- [ ] Conclusion
- [ ] DOI/Publisher

### Literature Review

- [ ] Introduction
- [ ] Individual paper analysis
- [ ] Comparative table
- [ ] Critical analysis
- [ ] Research gap
- [ ] Proposed research direction
- [ ] Final literature review
- [ ] IEEE references

### Verification

- [ ] All author names checked.
- [ ] All years checked.
- [ ] Results checked against original papers.
- [ ] No fabricated information.
- [ ] Research gap supported by the reviewed studies.

---

# 21. Final Workflow — One-Line Version

```text
Choose Topic
→ Search Digital Libraries
→ Select 5–10 Papers
→ Download PDFs
→ Extract Title + Authors + Year + Abstract + Dataset + Model + Results + Conclusion
→ Organize Papers
→ Paste Information + Universal Prompt into LLM
→ Generate Literature Review
→ Compare with Original Papers
→ Correct Errors
→ Identify Research Gap
→ Format References
→ Submit Experiment 2
```

---

# 22. Deliverables for Experiment 2

At the end of the experiment, your group should have:

```text
1. Downloaded Research Papers
2. Paper Information Sheet
3. Comparative Literature Table
4. AI-Assisted Literature Review
5. Identified Research Gap
6. IEEE References
7. Final Practical Record Entry
```

## Core Principle

**The papers are the evidence. The LLM is only the analysis and writing assistant.**

Do not allow the LLM to replace the actual literature search. The group should be able to show **which papers were selected, where they came from, what they said, and how the final research gap was derived.**
