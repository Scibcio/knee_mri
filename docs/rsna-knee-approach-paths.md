# RSNA Knee — How to Approach It (Paths to Test)

*Written 30 Sep 2026. There are 22 days left until the final deadline (22 Oct), and the last day to merge teams is 15 Oct.*

---

## 0. The short version

- **"Limited data" is really "limited labels."** You already have 4,407 training MRIs. Only 58 of them are labelled, but every one of them has a report. The cheapest way to get more training signal is to **turn those reports into better labels** (§2 Path B), not to look for new images.
- **External MRI data can help, but mostly for ACL and meniscus.** No public dataset covers all 12 findings. The host still hasn't said which datasets are allowed, and that question has gone unanswered for 5 days. Treat external data as a **side experiment with rules risk**, not the main plan (§3).
- **Pretrained weights are the safe way to add outside knowledge.** RadImageNet is MIT/CC-BY licensed, and ImageNet and DINOv2 are also fine. None of these carry rules risk.
- **Model choice matters less than people think.** Simple 2.5D CNNs at 224–288 px reach 0.94–0.95, and bigger models add very little (§4).
- **Recommended order:**
  1. **A** — simple baseline
  2. **B** — better labels and pseudo-labels
  3. Then **one** of C (external data), D (vision-language model) or E (efficiency play), based on what A and B show.

---

## 1. Your setup and what it allows

**Your RTX 5070 (12 GB)** is fine for Paths A, B and E. Path D is tight but possible with LoRA.

**Don't download 570 GB.** Instead, pre-process on Kaggle and download a compact cache:
1. Run a Kaggle CPU notebook that reads the DICOMs, sorts the slices, crops about 140 mm around the knee, resizes, and saves `uint8` arrays.
2. Size estimate: 64 slices × 224×224 × 4,407 studies ≈ **14 GB**. That fits within Kaggle's notebook output limit.
3. Save the output as a private Kaggle dataset, then download it and train locally.

**Kaggle GPUs:**
- The free weekly GPU quota is best spent on **inference submissions**. Train at home.
- Kaggle's T4 and P100 GPUs have **no bf16 support**. One participant lost 0.14 AUC because a bf16 inference notebook ran about 50× slower and hit the time limit. **Use fp16 or fp32 at inference.**

**Test set:** about **1,322 hidden studies**. Your notebook must process all of them within 9 hours, including DICOM reading.

**Medals** (Kaggle rules for 1,000+ teams, with 4,571 teams here):

| Medal | Cut-off | Approximate rank |
|---|---|---|
| Bronze | Top 10% | ~457 |
| Silver | Top 5% | ~228 |
| Gold | Top 10 + 0.2% | ~19 |

Check the leaderboard at those ranks to see what score each medal needs today.

---

## 2. The paths

### Path A — Strong simple baseline (do this first, everyone needs it)

**Goal:** a trustworthy 0.93–0.94 pipeline and a validation setup you can trust. **Time:** days 1–6.

1. **Labels:** start with a documented public LLM label set, such as stevenleehans' "RSNA Knee LLM Report Labels". Replace them with your own in Path B.
2. **Pre-processing:**
   - Order slices physically, using `ImagePositionPatient` projected onto the slice normal. Sorting by `InstanceNumber` agrees about 95% of the time and is fine to start with.
   - Use 3–5 "slots" by plane and sequence type, from `train_series.csv`: sagittal fluid-sensitive, sagittal other, coronal fluid, coronal other, and axial.
   - Take about 12–16 slices per slot, sampled densely from the central part of the stack. Crop about 140 mm and resize to 224 px.
3. **Model:**
   - A 2D backbone applied to each 2.5D input (3 adjacent slices as the 3 channels). Use ResNet34 or EfficientNet-B0 with ImageNet weights.
   - Mean or attention pooling over slices, then over slots, then 12 sigmoid outputs.
4. **Loss:** binary cross-entropy (BCE). **Mask out** targets the report doesn't address rather than forcing them to 0.
5. **Validation:**
   - 5-fold cross-validation (CV) against the report-derived labels, with folds grouped by scanner or site if you can.
   - Use the 58 gold studies only as a sanity check.
   - Set a noise threshold of about **0.003 AUC** and only accept changes above it.
6. **First submission:** a single fold in fp16, to learn how CV relates to the leaderboard (LB) and how long inference takes.

**Expected result:** 0.92–0.94 LB. **Risk:** low. **Hardware:** 12 GB is enough.

---

### Path B — Better labels and pseudo-labels (the highest-value lever)

**Goal:** +0.01 to +0.02 AUC over A. This is where top teams say their gains came from. **Time:** days 4–14, running in parallel with A.

**B1. Your own LLM labels.**
- Send each report to a cheap LLM API, which is allowed. One team spent under $5 in total.
- Ask for a probability per finding, **plus an explicit "not addressed" option**. The per-finding rules in step B2 then decide what "not addressed" means.
- **Put the competition's label definitions into the prompt.** Examples: a "high-grade" ACL tear only; ignore small effusions; osteochondral injuries don't count as acute fractures. The gold labels were graded strictly, so borderline findings count as negative.
- Score your labels against the 58 gold studies (macro AUC). The public LLM labels score about 0.878 on this; regex scores about 0.814.

**B2. Decide what silence means for each finding** (thread 733932):

| When the report doesn't mention it… | Treat as | Findings |
|---|---|---|
| …the finding is almost always absent | **0** | Baker's (≈3% positive when unmentioned), Medial OA (0%). Imputing Contusion and Lateral OA also hurt, so treat them as 0 too |
| …it's genuinely unknown | **mask it, or impute it** | Synovitis (≈34% positive when unmentioned), PF OA, Fracture (imputing these helped) |
| Special case | fill silent Synovitis from Effusion | these co-occur, so Effusion predicts Synovitis better than the Synovitis field does |

**B3. Blend label sources.** Average your LLM labels with 1–2 public label sets. The errors partly cancel out.

**B4. Teacher → pseudo-label → student loop.**
1. Train Path A models on the blended labels. These are the teachers.
2. Take their out-of-fold predictions. They are often **closer to the radiologists' labels than the report labels are**.
3. Mix them into the targets. A soft blend works, e.g. 0.5 × report label + 0.5 × teacher prediction.
4. Retrain a student model on the mixed targets. Repeat once or twice.

**B5. Check each label column against gold.** If the image model beats the report labels for a finding against gold (for example, Effusion 0.90 vs 0.70), lean harder on pseudo-labels for that finding.

**Expected result:** 0.94–0.95. **Risk:** low. **Hardware:** fine. The LLM calls go through an API, or run locally with Qwen3-8B, which fits in 12 GB at 4-bit.

---

### Path C — External knee MRI data (your "gather more from the web" idea)

**Goal:** add images the competition doesn't have, mainly to improve ACL and meniscus. **Time:** days 7–16. **Risk:** medium to high, because of the rules (see §3).

Four ways to use it, from safest to riskiest:

- **C1. Medical pretrained weights (no rules risk).**
  - Swap the ImageNet backbone for **RadImageNet ResNet50**, trained on 1.35 M CT, MRI and ultrasound images, MIT/CC-BY.
  - **Gotcha:** it expects **one slice repeated in all 3 channels**, not 3 adjacent slices. Getting this wrong cost one team about 0.02, and getting it right gave them +0.04 over ImageNet ResNet34 in their pipeline.
- **C2. MRNet (Stanford) as extra training data.**
  - **1,370 knee exams**, each with 3 planes: sagittal T2, coronal T1 and axial PD.
  - Labels: **ACL tear, meniscal tear, abnormal**. They were extracted from reports, so they are also noisy.
  - Format: about 6 GB of 256×256 `.npy` files. Access is a click-through research-use agreement.
  - Use it for **pre-training** or as **extra heads**: an MRNet head for ACL and one for any-meniscus, trained alongside your 12 outputs.
  - Domain gap: one scanner vendor (GE) and only 3 fixed sequences, so map them into your slots and normalise intensities carefully.
- **C3. Self-supervised pre-training on unlabelled MRIs.**
  - This teaches the backbone "what knees look like" before fine-tuning.
  - **The easiest version needs no external data at all:** pre-train (e.g. MAE, SimCLR or DINO-style) on the **competition's own ~800k training slices**.
  - fastMRI's **10,012 clinical knee DICOM studies** would be the ideal external source. However, its application now asks for institutional affiliation and proposed use, which the host may treat as an accessibility barrier.
  - On 12 GB and with 3 weeks left, this is a stretch. Only do it if A and B are done early.
- **C4. OAI with MOAKS scores (high reward, heavy lift).**
  - 4,796 participants, with bilateral knee MRI at several time points.
  - MOAKS radiologist readings exist for subsets, for example the FNIH cohort of about 600 knees. They cover cartilage damage (≈ OA labels), bone-marrow lesions, meniscus damage, and effusion- and Hoffa-synovitis. That makes OAI the **only** public source with image-based labels for synovitis and effusion.
  - The catch:
    - Access is through the NIMH Data Archive login.
    - The data volume is huge.
    - The subjects are an older OA cohort scanned with a research protocol, so the domain shift is large.
    - A participant has explicitly asked the host about it (thread 741819), with no answer yet.
  - Probably not realistic in 22 days.

**Things to skip:**
- **KneeCoT:** the host explicitly banned it.
- **SKM-TEA:** 155 scans of a 3D research sequence (qDESS), too different from clinical scans.
- **KneeMRI (Rijeka):** 917 sagittal ACL studies, but its no-derivatives (ND) licence clashes with the requirement for winners to publish weights.
- **Scraping images from the web** (Radiopaedia, Google Images, papers): these aren't a "reasonably accessible dataset" under the rules, the licences and terms of service forbid it, the volume is tiny, and there are no reliable study-level labels. **Don't.**

**Expected result:** unknown. Top competitors *suspect* external data could add a lot. For ACL and meniscus, which are already about 0.95 AUC, the upside is probably modest.

**How to test it:** keep everything else fixed and compare CV and gold AUC **per finding** with and without the external data.

---

### Path D — Fine-tune a small vision-language model (high ceiling, experimental)

**Goal:** a different model family for the ensemble, or a single model at 0.95. **Time:** days 10–20.

- One participant (rank 53) fine-tuned **Qwen 3.5 2B**: LoRA on the language model plus the vision encoder, 384 px inputs, 12 outputs. That gave **0.950 on a single fold**, with about 3.5 hours of training.
- **On 12 GB** you'll need LoRA, gradient checkpointing, a batch size of 1–2 and fp16 or bf16 (your card supports bf16; Kaggle's T4 does not). It's feasible but slow to iterate.
- **Inference risk:** 1,322 studies × many slices on a T4 in fp16 has to fit in 9 hours. Time it on a subset first.
- **Why try it:** its errors will differ from a CNN's, which is what makes an ensemble gain real.

**Expected result:** 0.94–0.95. **Risk:** medium, mostly engineering and time. **Hardware:** tight.

---

### Path E — Efficiency-track play (a realistic prize angle for a solo entrant)

**Goal:** a strong score with **runtime under ~15 minutes**. **Time:** it builds on A and B at no extra cost.

- The efficiency score heavily penalises runtime: **1 extra hour costs about as much as 0.05 AUC**. A fast 0.945 model beats a slow 0.955 ensemble on that track.
- Scott Willis (#3 on the main leaderboard) got there with a small ResNet and was #1 on the Efficiency LB at the time.
- Speed tricks:
  - Read only the series and slices you use.
  - Decode in parallel with threads; opening files is the bottleneck.
  - Use small inputs (224 px), 1–3 folds and fp16.
  - Skip heavy test-time augmentation.
- **Final-selection strategy:** the Efficiency LB uses your 2 final picks, so pick **one fast model and one best-score model**.

**Expected result:** same AUC as B at a fraction of the runtime. **Risk:** low.

---

### Path F — Blend with strong public notebooks (a cheap medal insurance)

- Chris Deotte (rank 97) says blending your own model with legitimate public models usually gets a medal.
- **Rank-average** your predictions with 1–2 public single-model notebooks, such as the Raptor CoAtNet or a DINOv2 notebook.
- Only keep a blend if it also improves your **CV**, not just the public LB. Many public blends are overfitted to the public LB.

---

## 3. External data — the rules situation (as of 30 Sep)

| Status | What |
|---|---|
| ✅ Allowed | Public pretrained weights (ImageNet, DINOv2, RadImageNet…), LLM APIs for labelling reports, hand-labelling the *training* images (community reading of the rules, not confirmed by the host) |
| ✅ Allowed in principle | Datasets with only a **non-commercial** licence. The host said prize money doesn't make use "commercial" |
| ⚠️ Probably OK, but no ruling | Datasets behind a **simple registration or click-through** (MRNet, SKM-TEA). The host said these "generally" meet the accessibility test |
| ❓ Unclear | Datasets needing **institutional or identity verification** (fastMRI's institutional application, OAI's NDA login) and OAI-trained public weights. The thread asking for a per-dataset ruling (743416) has been unanswered for 5 days |
| ❌ Not allowed | **KneeCoT**, because it requires a formal institutional agreement |

**Practical advice:**
- If you use external data, **stick to click-through datasets like MRNet**.
- Keep a **no-external-data** submission as one of your two final picks.
- Keep a written note of each dataset's licence and how you accessed it.
- Check thread 743416 before the final selection on 22 Oct.

---

## 4. Model and architecture menu

| Backbone | Why | Reported | Fits 12 GB |
|---|---|---|---|
| **ResNet34 / EfficientNet-B0** (ImageNet) | Fast and strong. Good for experiments and the efficiency track | 0.936–0.954 LB | ✅ Easily |
| **CoAtNet** (`coatnet_rmlp_2_rw_384`) | Current top single-model family, used in the Raptor pipeline | 0.937–0.950 LB | ✅ At 224–288, batch small at 384 |
| ConvNeXt-Tiny | Solid modern CNN, good for ensemble diversity | — | ✅ |
| **RadImageNet ResNet50** | Medical pretraining, safe outside knowledge | Helped some pipelines and hurt others (input-format dependent) | ✅ |
| DINOv2-S (ViT) | Most used on the Models tab | 0.887–0.945 LB | ✅ |
| **Qwen 3.5 2B VLM (LoRA)** | Different family, high ceiling | 0.950 single fold | ⚠️ Tight |

**Architecture choices that mattered, according to participants:**
- **2.5D inputs.** Use 3 neighbouring slices as channels. RadImageNet is the exception: repeat a single slice ×3.
- **Dense slice sampling beats wide coverage.** More slices over the same range helped; spreading the same number of slices wider hurt.
- **Learned attention pooling** beat mean pooling by about 0.014 in one pipeline. Don't change the pooling at inference.
- **Resolution:** 224–288 px is enough. 384 px gave little for the extra cost.
- **Folds:** a single fold is often within noise of 5 folds on the LB, so use 1 fold for speed and 5 folds for the final submission.

---

## 5. Suggested 22-day schedule

| Dates | Work |
|---|---|
| 30 Sep – 2 Oct | Kaggle pre-processing notebook → compact cache → download. Read the public baseline notebooks |
| 2 – 5 Oct | **Path A** baseline and 5-fold CV. First LB submission (fp16). Start **B1** LLM labelling in parallel |
| 5 – 9 Oct | **Path B** (B1–B3): own labels, silence rules, blends. Swap in RadImageNet or CoAtNet (**C1**). Submit the best |
| 9 – 13 Oct | **B4** pseudo-label loop. Decide on **C2** (MRNet) or **D** (VLM). Only one; there isn't time for both |
| by 15 Oct | **Team-merge deadline.** Consider teaming up with someone whose models are different from yours; several are asking in the Discussion |
| 15 – 20 Oct | Chosen extra path. Ensemble your models (rank-average), plus an optional public blend (**F**) |
| 21 – 22 Oct | Pick the 2 finals: **best CV model** + **fast model** (**E**). Check thread 743416 for any external-data ruling |

---

## 6. Experiment log template

Change **one** thing per run and write down:
- run id, what changed, CV macro AUC, CV per finding, gold-58 AUC, LB (if submitted), inference minutes.

Keep a change only if CV improves by more than 0.003.

---

## Sources

- Competition rules and discussion threads:
  - 743416 (external datasets, still unanswered)
  - 734109 (KneeCoT banned)
  - 733965 (LLM ruling, non-commercial ruling)
  - 740718 (hand-labelling train images)
  - 733932 (label silence)
  - 735304 and 740610 (model scores and advice)
  - 737696 (Raptor and RadImageNet findings)
  - 743698 (Kaggle GPU)
  - 744230 (bf16 on T4)
  - https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion
- MRNet dataset: https://aimi.stanford.edu/datasets/mrnet-knee-mris and https://stanfordmlgroup.github.io/competitions/mrnet/
- fastMRI (access terms, 10,012 DICOM studies): https://fastmri.med.nyu.edu/ and https://pubs.rsna.org/doi/10.1148/ryai.2020190007
- fastMRI+ annotations: https://github.com/microsoft/fastmri-plus and https://pmc.ncbi.nlm.nih.gov/articles/PMC8983757
- OAI access: https://datacatalog.med.nyu.edu/dataset/10162
- MOAKS in the FNIH cohort: https://link.springer.com/article/10.1186/s12891-016-1310-6
- SKM-TEA summary: https://arxiv.org/pdf/2607.02428
- RadImageNet: https://pmc.ncbi.nlm.nih.gov/articles/PMC9530758, https://github.com/BMEII-AI/RadImageNet and https://huggingface.co/Lab-Rasool/RadImageNet
