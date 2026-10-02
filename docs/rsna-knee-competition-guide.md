# RSNA Knee Abnormality Detection — Complete Competition Guide

*Compiled 29 Sep 2026 from every tab of the competition (Overview, Data, Code, Models, Discussion, Leaderboard, Rules, Team, Submissions) plus the most important discussion threads. Numbers are as shown on the site that day.*

Competition: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection

---

## 0. TL;DR

- **Task:** from a knee MRI study (several DICOM series), output a probability for each of **12 findings**. It is multi-label, so one knee can have several findings at once.
- **Metric:** mean ROC AUC across the 12 labels (macro AUC). Every label counts equally.
- **The catch:** only **58 of 4,407** training studies have real labels, which is 1.3%. Every other study has only a free-text radiology report in one of several languages. Everyone turns the reports into training labels, and **label quality is the main lever** in this competition.
- **Test data has no reports.** Report text can be used to *create training targets*, never as a model input at inference time.
- **The hidden test labels come from radiologists reading the images, not from the reports.** The host has confirmed that reports and labels disagree fairly often.
- **Submission:** a Kaggle Notebook, GPU or CPU, **≤ 9 h runtime, internet off**. Max 5 submissions per day, and you pick 2 as final.
- **Leaderboard today:** #1 scores 0.961, #10 scores 0.957, #50 scores 0.953. It is extremely tight. The public LB uses only about 30% of the test set, so expect a shake-up on the private LB.
- **What works (per the community):** clean report-derived labels (an LLM beats regex), pseudo-labelling, and surprisingly **simple 2.5D CNNs at 224–288 px** (ResNet34, EfficientNet-B0, CoAtNet), which reach 0.94–0.95. Bigger backbones help little.
- **LLM APIs are explicitly allowed** for extracting labels from reports (host ruling).
- **Your status:** rules accepted, solo team (you are the leader), **0 submissions**. The final deadline is **22 Oct**, 23 days away.

---

## 1. Key facts

| Item | Value |
|---|---|
| Host | Radiological Society of North America (RSNA) |
| Type | Research **code** competition (submissions run as Kaggle Notebooks) |
| Prize pool | **$77,000**. Main LB: 10 places, $9k down to $5k. Efficiency track: $7k / $6k / $5k |
| Participation | ~22.8k entrants, ~5.2k participants, **~4,570 teams**, ~69.5k submissions |
| Metric | Macro-averaged ROC AUC over 12 labels |
| Test set | ~1,300 studies. Public LB ≈ 30%, private LB ≈ 70% |
| Runtime limit | ≤ 9 h on CPU or GPU, internet disabled |
| Submissions | 5 per day, 2 final selections |
| Team size | Max 5 |
| External data / pretrained models | Allowed if free and reasonably accessible to all (details in §6) |
| Winner licence | CC-BY-NC 4.0 (code + weights) |
| Data licence | RSNA MIRA licence (commercial + academic use allowed) |
| Tags | Image classification, computer vision, text, medicine |

---

## 2. Timeline

All deadlines are 11:59 PM UTC, which is 12:59 AM the next day in London while BST is in effect.

| Date | Event | From today (29 Sep) |
|---|---|---|
| 30 Jul 2026 | Start | — |
| **15 Oct 2026** | Entry deadline + **team merger deadline** | 16 days |
| **22 Oct 2026** | **Final submission deadline** | 23 days |
| 5 Nov 2026 | Winners submit code, video and method write-up | — |

You have already accepted the rules, so the entry deadline doesn't affect you. It only matters if you want to **join or merge with a team**, which must happen by 15 Oct.

At most about **115 submissions** remain for you (5/day × 23 days).

---

## 3. The 12 labels and how they were graded

Two musculoskeletal radiologists labelled each annotated study independently from the images, and a third radiologist settled any disagreement. Labels apply to the whole exam for one knee. **Borderline or "on the fence" findings were graded negative**, which favours specificity.

The table below paraphrases the host's pinned Overview post (thread 733343).

| Column | Finding | Counts as positive | Counts as negative |
|---|---|---|---|
| `ACL` | Anterior cruciate ligament tear | High-grade partial or full tear: complete discontinuity or >50% of fibres disrupted | Mild signal change, degeneration or thickening without discontinuity |
| `MCL` | Medial collateral ligament tear | High-grade partial or complete **acute** tear, with disrupted fibres and edema | Low-grade sprain, chronic or old changes |
| `Medial Meniscus` | Medial meniscus tear | Abnormal signal that clearly reaches the meniscal surface on **≥2 images**, or a truncated, diminutive or displaced fragment | Internal degeneration that doesn't reach the surface |
| `Lateral Meniscus` | Lateral meniscus tear | Same criteria as the medial meniscus | Same |
| `Medial OA` | Medial compartment osteoarthritis | Moderate or large area (≈1 cm+) of high-grade cartilage loss (>50% thickness) | Anything milder |
| `Lateral OA` | Lateral compartment OA | Same | Same |
| `PF OA` | Patellofemoral OA (kneecap and trochlear groove) | Same | Same |
| `Effusion` | Joint effusion | Moderate or large fluid distending the joint | Small or trace fluid |
| `Synovitis` | Inflamed, thickened joint lining | Present | — |
| `Baker's` | Baker's (popliteal) cyst | Moderate or large fluid collection behind the knee | Small cysts |
| `Contusion` | Bone bruise / bone-marrow edema from impact | Edema-like marrow signal **without** a fracture line | — |
| `Fracture` | Acute fracture | Acute cortical break or fracture line | — |

**Host clarifications** from thread 733826:
- Labels were assigned **from the images, independently of the reports**.
- When image and report disagree, the **image-based label is authoritative**.
- A negative label means the finding was annotated as **absent**, not "not annotated".
- Some studies were bilateral (both knees). The host adjusted the report text or metadata so that the relevant knee can be identified.
- Report–label disagreements are **expected**. A clinical report is written by one radiologist, while the labels used several readers and stricter image-based thresholds.

**MRI basics** (from the Overview post):
- An exam is a set of **series**, and each series has a plane and a pulse sequence.
  - **Sagittal** is a side view. **Coronal** is a front view. **Axial** is a cross-section.
- **Fluid-sensitive** sequences (PD or T2, usually fat-suppressed) make fluid, edema and tears appear bright. Most findings are detected on these.
- Which plane shows a structure best:
  - Cruciate ligaments and menisci are well seen on sagittal and coronal images.
  - Patellofemoral cartilage is best seen on axial images.

---

## 4. The data

**Total: 569.76 GB, about 820k files.** DICOMs are organised as `train_series/<StudyUID>/<SeriesUID>/<SOPInstanceUID>.dcm`, with one slice per file.

### Files

| File | What it is |
|---|---|
| `train.csv` (5.69 MB) | One row per training study: `StudyInstanceUID`, `Report` (free text), and the 12 label columns |
| `train_series.csv` | One row per series: `StudyInstanceUID`, `SeriesInstanceUID`, `Fluid_Sensitive` (0/1), `Fat_Suppression` (0/1), `Anatomical_Plane` (Sagittal/Coronal/Axial) |
| `train_series/` | Training DICOMs |
| `test.csv`, `test_series.csv`, `test_series/` | **Only 3 example studies.** The hidden set of about 1,300 studies replaces them during scoring. **No `Report` column at test time.** |
| `sample_submission.csv` | All labels set to 0.5. This is also the Efficiency-track benchmark. |

### Training set statistics

**Studies and series**
- **4,407 studies** and **24,371 series**, about 5.5 series per study.
- Planes: Sagittal ≈ 40%, Coronal ≈ 35%, Axial ≈ 24%.
- `Fluid_Sensitive` = 1 for about 57% of series. The two flags are often correlated but are not identical.
- A series usually has 20–45 slices (median 30), with some reaching a few hundred.

**Reports**
- There are 4,276 unique reports among 4,407 studies. Some reports are templated "normal knee" boilerplate; the most common one is Turkish.
- Reports come in several languages. Spanish, Dutch, English, French, German, Turkish and Bulgarian were all seen, and one thread counts nine.

**Images**
- Images come from about 20 institutions worldwide, and one participant counted 265 distinct scanner/software fingerprints.
- Intensities, orientations and resolutions vary.
- Transfer syntaxes are mixed: uncompressed, JPEG Lossless, JPEG 2000 and Implicit VR. Your DICOM reader must handle all of them.
- Metadata is stripped down to 86 allowed tags.

### The 58 "gold" labelled studies

| Label | Positives / 58 | Prevalence |
|---|---|---|
| ACL | 24 | 41% |
| MCL | 9 | 16% |
| Medial Meniscus | 26 | 45% |
| Lateral Meniscus | 23 | 40% |
| Medial OA | 15 | 26% |
| Lateral OA | 11 | 19% |
| PF OA | 21 | 36% |
| Effusion | 35 | 60% |
| Synovitis | 27 | 47% |
| Baker's | 12 | 21% |
| Contusion | 19 | 33% |
| Fracture | 18 | 31% |

Every gold study has at least one positive finding, with an average of about 4.1 per study (per thread 733932). These prevalences therefore **do not** reflect the real prevalence in the data. The host also warns that prevalence may differ between train, public LB and private LB.

---

## 5. Scoring and submitting

### Main metric

**Score = mean over the 12 labels of the ROC AUC.** Because AUC depends only on ranking, well-calibrated probabilities aren't required; what matters is the order of the predictions within each label.

### Submission format

```
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
<uid>,0.5,0.5,...
```

The file must be named `submission.csv` and written by the notebook.

### Notebook constraints

- Runtime ≤ 9 h on CPU or GPU.
- **Internet off.** Model weights must be attached as Kaggle datasets or models.
- Free, public external data and pretrained models are allowed.

### Efficiency track

**Who is eligible**
- Only your final selected (or auto-selected) submissions count.
- The submission must beat the 0.5 benchmark on the private LB.

**The formula as shown on the page (lower is better):**

`Efficiency = AUC / (Benchmark − maxAUC) + RuntimeSeconds / 32400`

- **Benchmark** is the score of the all-0.5 sample submission (about 0.5).
- **maxAUC** is the best AUC anyone gets on the private LB.
- **32400** is 9 hours in seconds.
- The denominator is negative, so a higher AUC lowers your score.

**Worked example (my own arithmetic, assuming maxAUC ≈ 0.96):**
- Each extra hour of runtime costs about as much as **losing ~0.05 AUC**.
- So runtime is heavily weighted: a fast single model can beat a slow ensemble on this track.

**Other points**
- A public notebook tracks the Efficiency LB daily. During the competition it shows ranks only: https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb
- Participants note that your **two main-LB selections are also what the Efficiency LB uses**. Selecting a heavy ensemble hurts you there.

### Final ranking

- Only the **private** LB (about 70% of test) decides prizes and medals.
- Ties go to whoever submitted first.

---

## 6. Rules that actually matter

### Teams and sharing

- Max 5 members per team.
- Mergers are allowed until 15 Oct, but only if the combined submission count is ≤ 5 × (days elapsed).
- **No private sharing** of code or data outside your team. Public sharing on Kaggle forums or notebooks is fine, and publicly shared code is treated as open-source licensed.
- **Data security:** don't give the competition data to anyone who hasn't accepted the rules.
- One Kaggle account per person.

### LLMs on the report text — explicitly allowed (host, thread 733965)

- Sending reports to a commercial LLM API (OpenAI, Anthropic, Google, etc.) to extract labels **is not treated as prohibited private sharing**.
- The host confirmed in replies that you can use an LLM API to read the reports and generate labels.
- LLM-derived labels or embeddings are OK as model inputs, but only from information that exists at inference time. Reports don't exist at test time.
- The tool must still be low-cost and reasonably accessible to everyone. One participant said all their API labelling cost **under $5**.

### External datasets — partly settled

- The host said a **non-commercial licence alone doesn't exclude a dataset**. Winning prize money doesn't by itself count as commercial use.
- The deciding test is **accessibility**:
  - Datasets behind a simple registration or click-through agreement are generally OK.
  - Datasets that need institutional approval, IRB approval, negotiated agreements or long credentialing may not be allowed.
- **KneeCoT has been ruled not allowed** because it requires a formal institutional agreement. This is reported in thread 743416.
- **Still unresolved** as of today (thread 743416, very active): a per-dataset yes/no for MRNet, SKM-TEA, fastMRI/fastMRI+, OAI, KneeMRI (CC BY-NC-ND) and KneeXNet-2.5D. The same thread asks whether publicly released weights trained on OAI are allowed.
- Participants warn that external knee MRI data could give a big boost, so this ruling may matter a lot.
- You remain responsible for complying with each dataset's own licence.

### Winners' obligations (only relevant if you win a prize)

- Release training code, inference code and weights under **CC-BY-NC 4.0**. Weights go on a public Kaggle dataset, with a `requirements.txt` plus a Kaggle image or Dockerfile.
- Write a reproducible method description.
- Record a short video.
- Post your code and weights links on the forum.
- Sign the prize documents. Foreign residents need a W-8BEN tax form.

### Eligibility

- You must be 18+ and not in a sanctioned region. UK residents are fine.
- Employees of RSNA or Kaggle can take part but can't win prizes.

---

## 7. Your account status (as of today)

- ✅ Rules accepted.
- Team: **solo**, you are the leader, 4 open slots, no invitations received.
- Submissions: **none yet**.

---

## 8. Leaderboard snapshot (public, ~30% of test)

| Rank | Team | Score | Entries |
|---|---|---|---|
| 1 | rhoskeri | 0.961 | 79 |
| 2 | Wassim Dobbi | 0.960 | 85 |
| 3 | Scott Willis | 0.958 | 166 |
| 4–8 | TomKroumov, Parag & Ian, YOKKOISHO, Brandon Low, takoi & charmq | 0.958 | — |
| 10 | Attention Is All You Knee | 0.957 | 98 |
| 20 | kreininmv | 0.956 | 137 |
| 30 | Tucker Arrants | 0.955 | 69 |
| 49 | Optimo | 0.953 | 39 |

- There are 4,571 teams in total.
- Ranks 1 to 50 are separated by only **0.008 AUC**.
- The public LB covers only about 390 studies, so gaps of 0.002–0.003 are mostly noise. Several top competitors say openly that they expect a shake-up.

---

## 9. What the community has learned (key discussion threads)

### 9.1 Labels are the bottleneck

**LLM labels vs regex** (thread 733932, stevenleehans):
- LLM-extracted labels scored **0.878** macro AUC against the 58 gold studies. Regex/keyword extraction scored **0.814**.
- About **25% of label cells are "report doesn't mention it"**, and what that silence means depends on the finding:
  - **Baker's cyst:** if the report is silent, the cyst is almost certainly absent (~3% positive). Treat silence as negative.
  - **Synovitis:** about 84% of reports never mention it, yet 27 of the 58 gold studies are positive. When a report is silent on synovitis, there is still about a 34% chance it is present, so silence carries little information.
  - **Effusion predicts synovitis better than the synovitis field does.** Filling only the undecided synovitis cells from effusion improved that column from 0.678 to 0.790.
  - Filling *every* finding's gaps with a learned model made things **worse**. Only fill gaps where silence is genuinely uninformative.
- Takeaway: have your labeller output "not addressed" as its own answer, then decide separately for each finding what that means.

**Reports vs gold labels** (thread 733826):
- A manual audit found report-only labels agreed with gold in 82.5% of 240 decisions.
- Examples of disagreement:
  - A report says "Baker-Zyste" but the label is 0.
  - A report says the lateral meniscus is normal but the label is 1.

**Models can beat their own training labels** (threads 740610 and 735304):
- One participant's image model scored higher against gold than the report labels it was trained on for Effusion (0.90 vs 0.70), MCL and Baker's.
- Another (#38) found his out-of-fold predictions were already closer to the radiologists than the extracted labels.
- This is why **pseudo-labelling** works: train a teacher model, relabel the data with its predictions, then train a student on the new labels.

**Recipe used by a top-80 competitor:**
1. Collect several weak-label sources: public label sets, API LLM labels and local LLM labels.
2. Train several teacher models on them.
3. Use the teachers to generate pseudo-labels, and train the final model on those.

They found cheap and expensive LLMs made little difference.

**Caution on "single model" claims (Chris Deotte, #92):** a single model trained on pseudo-labels already contains an ensemble's knowledge. Treat "single model" claims with that in mind.

### 9.2 No shortcut in the metadata

Thread 733517 checked whether DICOM header metadata alone could predict the labels:
- Metadata alone reaches only about 0.65 macro AUC with random folds.
- It drops to about 0.60 when folds are split by scanner, because the extra signal was just memorising sites.
- **Leaderboard scores reflect genuinely reading the images.**

### 9.3 What models score (reported by participants)

| Setup | Reported score |
|---|---|
| ResNet, 5-fold, 224 px (#10 on LB) | 0.954 LB |
| **Qwen 3.5 2B vision-language model**, LoRA on the language part + fine-tuned vision encoder, 384 px, single fold, ~3.5 h training | 0.950 LB |
| 2.5D CoAtNet at 224 px (a few neighbouring slices at a time), single fold | 0.950 LB |
| CoAtNet 288 px, gains mostly from label work | 0.949 LB |
| Small ResNet (Scott Willis, #3, also #1 on Efficiency LB at the time): single fold 0.938, then 5-fold | 0.947 LB |
| 5-fold model at 224 px (Tucker Arrants) | 0.943 LB |
| 288 px + test-time augmentation (#6) | 0.943 LB |
| Small ResNet at 224 px, single fold | 0.936 LB |
| DINOv2-based models | 0.887 – 0.932 LB |

**Takeaways**
- Resolution above ~288 px and larger backbones gave little. One team got only +0.001 from scaling the encoder (thread 735154).
- The **DINOv2-small** pretrained model is the most-used on the Models tab (161 users, best score 0.945). However, simple ImageNet CNNs match or beat it.

### 9.4 Preprocessing details people shared

**Physical crop around the knee:** 130–150 mm is typical.
- Tom Aindow: 392 px, 150 mm centre crop, bag of 32 random slices per study.
- Will: 130 mm crop, 336 px, 6 sequence slots, 12 slices per slot.

**The "Raptor" pipeline (Dread Development):** 80 slices per study spread across 2–98% of each series, 140 mm crop, 336 px. The 80 slices are split across 5 slots:

| Slot | Slices |
|---|---|
| Sagittal fluid | 22 |
| Sagittal | 18 |
| Coronal fluid | 15 |
| Coronal | 10 |
| Axial | 15 |

- **Slice density mattered more than span.** Spreading the same number of slices over a wider range *cost* 0.009.
- Evaluating more windows per study helps up to a point, then stops paying.

**Input and pooling details you can get wrong** (from 737696):
- **RadImageNet** weights (pretrained on medical images) expect a single slice copied into all 3 channels. Feeding them 3 adjacent slices instead cost about 0.02.
- Swapping a learned attention head for mean pooling cost 0.014.
- Changing the pooling method at inference only cost about 0.02.

**Runtime tip (Efficiency LB):** in one pipeline the biggest time cost was ordering slices by geometry, which meant opening about 186 files per study. Sorting by `InstanceNumber` agreed 94.5% of the time but didn't speed things up, because opening each file is the slow part.

### 9.5 How to validate (strong consensus, thread 740610)

- **Validate with cross-validation on your report-derived labels.** Don't rely on the 58 gold studies or the public LB. Neither is big enough to detect 0.002–0.003 differences.
- Set a noise threshold, around **0.003 AUC**, and only accept changes above it. Use multiple seeds if you need a tighter estimate.
- **Change one thing at a time.** Submit only real improvements, and use them to check that CV and LB move together.
- The 58 gold studies work as a sanity check. They can't separate differences smaller than about 0.02.

### 9.6 Advice from a top-30 competitor (Tucker Arrants, thread 740610)

1. Reproduce the best **single-model** public notebook locally. Ignore the blends.
2. Use documented public LLM labels, or ideally your own so you know where they fail.
3. Start at **224/288 px with a simple CNN** such as ResNet34 or EfficientNet-B0, with plain pooling and no attention. This alone can reach 0.94+.
4. Work through the augmentations one at a time, including MRI-specific ones from past RSNA competitions.
5. Only then try other backbones, resolutions, attention or newer encoders.
6. Be sceptical of public notebooks. Many are over-engineered LB-chasing blends, and some are copies of other people's work (see thread 735348).

---

## 10. Public notebooks worth knowing

The notebook bodies render in a sandboxed frame I couldn't read (one crashed on load). This list comes from the Code tab (sorted by votes) plus what discussion threads say about each notebook.

| Notebook | Author | LB | Notes |
|---|---|---|---|
| [RSNA Knee Abnormalities – Efficiency LB](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb) | Ryan Holbrook (Kaggle) | — | Pinned. Official daily Efficiency LB |
| [RSNA Knee baseline v1](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1) | Pilkwang Kim | 0.891 | Most-voted (644). An early baseline paired with the first public LLM labels |
| [RSNA Knee: Data structure, EDA, baseline](https://www.kaggle.com/code/romanrozen/rsna-knee-data-structure-eda-baseline) | Roman Rozen | 0.894 | Good EDA. Its report extractor is used by others |
| [Knee MRI: twelve findings from a single model](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model) | Dread Development | 0.924 | "Raptor" CoAtNet pipeline, praised as a genuinely original approach |
| [Knee MRI: training the twelve-finding model](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model) | Dread Development | — | Training code for the above |
| [Bend the Knee to the Dinosaurs](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-the-dinosaurs) | Mattia Angeli | 0.941 | DINOv2-based |
| [Bend the Knee to DinoV3 (ensembled)](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-dinov3-ensembled) | Mattia Angeli | 0.922 | DINOv3 |
| [RSNA Knee: read the report, then the knee](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee) | prvsiyan | 0.906 | Report → labels → image model |
| [Head and shoulders, knees and toes](https://www.kaggle.com/code/prvsiyan/head-and-shoulders-knees-and-toes) | prvsiyan | 0.937 | |
| [RSNA_Versia_5](https://www.kaggle.com/code/evgendvorkin/rsna-versia-5) | evgendvorkin | 0.943 | Blend |
| RSNA Knee Fast 2xT4 Inference | — | 0.943 | Fast inference on 2 T4 GPUs |
| Bend the Knee to Speedy Raptors – The Original | — | 0.943 | Raptor-based blend |
| rsna-knee-0942-restructured | — | 0.942 | |
| RSNA Knee DINO-RadImageNet Rank Ensemble | — | 0.940 | |
| [rsna-metadata-probe](https://www.kaggle.com/code/zhukovoleksiy/rsna-metadata-probe) | Oleksii Zhukov | — | Shows metadata isn't a shortcut |

### Public label sets and weights mentioned in discussions

- **RSNA Knee LLM Report Labels** (stevenleehans). These include per-finding probabilities and a "not addressed" = 0.5 value.
- **rsna-knee-llm-labels** (Pilkwang Kim), the first public LLM label set.
- **Stratified Folds & LLM Soft Labels** (barun2104).
- **LLM Report Labels** (lixin73).
- [rsna-knee-abnormality-qwen3-8b-weak-labels](https://www.kaggle.com/datasets/laymond/rsna-knee-abnormality-qwen3-8b-weak-labels) (laymond). Note that a participant has asked whether including report text in a public dataset breaks the redistribution rule.
- Raptor weights from Dread Development, e.g. [raptor-knee-widedense](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense) and [raptor-knee-finespacing](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-finespacing), a single model that scored 0.932. The dataset pages list the preprocessing settings each checkpoint expects, and you must match them.

---

## 11. Models tab (pretrained models used by competitors)

| Model | Users | Best public score |
|---|---|---|
| DINOv2 small | 161 | 0.945 |
| DINOv2 base | 16 | 0.919 |
| BiomedCLIP | 1 | 0.906 |
| EfficientNet-B0 | 1 | 0.81 |
| DINOv3 ViT-B/16 | 1 | 0.771 |

Others on the tab include multilingual MiniLM (text), Qwen3-8B and Gemma-3-4B (used to label reports), Swin-Tiny, ResNet34, and a Pillar-0 MRI foundation model fine-tuned on knee MRI.

---

## 12. Other threads (titles only, for browsing)

- 730709 — How to get started + official Discord (discord.gg/kaggle)
- 733375 — Host's welcome post
- 735767 — What could the final ceiling be?
- 735826 — What's the real bottleneck: labels, image pipeline or model capacity?
- 735154 — Scaling the encoder bought us nothing (+0.0011)
- 736268 — Too many noisy public notebooks
- 735348 — Byte-for-byte public notebook copying
- 736678 — Worth checking the Efficiency LB (many top public-LB teams are slow ensembles)

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **Study** | One MRI exam of one knee. This is what you predict for. |
| **Series** | One acquisition within a study, with one plane and one sequence. A study has about 5–6. |
| **Slice** | One 2D image, stored as one `.dcm` file. |
| **2.5D** | Feeding a 2D CNN a few adjacent slices stacked as channels, a cheap way to add 3D context. |
| **Fluid-sensitive / fat-suppressed** | Sequences where fluid and edema look bright. Most findings are seen on these. |
| **Weak labels** | Labels derived from reports rather than expert image reads. They are noisy. |
| **Pseudo-labels** | Model predictions used as new training targets. |
| **OOF** | Out-of-fold predictions: each sample is predicted by the model that didn't train on it. |
| **Gold / gold58** | The 58 expert-labelled training studies. |
| **Public / Private LB** | Scores on about 30% and about 70% of the test set. Only the private one counts. |

---

## Sources

- Overview: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview
- Data: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data
- Rules: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules
- Leaderboard: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard
- Code: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/code
- Models: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/models
- Host overview and label definitions: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343
- LLM use ruling and external-data ruling: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965
- Report vs label discrepancies (host answers): https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826
- External datasets question (open): https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416
- Rules clarification thread: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733652
- "Not addressed" LLM labels study: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932
- Metadata shortcut test: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733517
- Best single-model scores: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304
- Plateau at 0.903 + advice: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740610
- Raptor weights thread: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737696
- Efficiency LB thread: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/736678
