# WildGuard design questions

This document shows how WildGuard's system design changes after the alignment work, stage by stage,
and what is still to decide before the pipeline is rebuilt. It does not change the alignment:
report §3 and [`DATA/ALIGNING_LEMIS_AND_CITES.md`](../DATA/ALIGNING_LEMIS_AND_CITES.md)
(*ALIGNING*) stay as they are. It takes their consequences for the system as given.

It has three parts:
- **Part A, the pipeline:** what scores the records, stage by stage. Report §4 is written from it.
- **Part B, model selection:** how the supervised models, the unsupervised detectors and the way
  they are combined are chosen, trained, tuned and compared. Report §4 explains the choices from
  it, with a flow diagram.
- **Part C, the evaluation:** how the report shows that the chosen system works. Report §6 is
  written from it.

Each item is tagged with the research question it serves (report §1.3).

**Where the changes come from.**
- **The alignment (report §3, ALIGNING)** changes stages 0, 1, 2 and 4 of the pipeline.
- **The review of transfer validity** (raised in a discussion with ChatGPT) changes mainly the
  evaluation: the evidence for transfer is indirect, so Part C sets out what can be measured. It
  also suggested a domain-adversarial network (S3.1), which was considered and not pursued.
- **Choosing models by comparison rather than by assumption** is Part B. The current pipeline
  picked XGBoost first and compared two other models afterwards; Part B compares candidates taught
  in the course under one protocol and keeps only what the evidence supports.
- **Checking the current pipeline against all of these** raised the remaining questions, mostly in
  stages 5 and 6.

**How open questions are decided.**
- **Settled:** follows from ALIGNING; implemented in the rebuild, no experiment needed.
- **To test:** an experiment decides, with its decision rule written before its results are seen,
  as in notebooks 04 and 05. The rules sketched here are drafts until the experiment's notebook
  states them. Experiments go in `notebooks/` as 06 onward.
- **Choice:** data cannot decide; the choice and its reasons are recorded.

Findings are cited by number: L, C and X (EDA notebooks 01–03), P (notebook 04), M (notebook 05).
Figures quoted from report §6 come from the current pipeline, which predates the alignment; they
show why a question is open, not what its answer will be. Course material is cited by module and
day: PSUPR (Problem Solving Using Pattern Recognition) and PRMLS (Pattern Recognition and Machine
Learning Systems).

| # | Research question | Report |
|---|---|---|
| RQ1 | Can LEMIS and CITES be aligned so that the same trade looks the same in both, and what cannot be aligned? | §3.4 |
| RQ2 | Does it work: can enforcement outcomes be predicted on shipments the model never saw, overall and on the CITES-like part of LEMIS (CITES-listed species, commercial trade)? | §6.2 |
| RQ3 | Is it useful: how much inspection effort does it save compared with random or rule-based inspection, measured on held-out LEMIS (CITES has no outcomes)? | §6.3 |
| RQ4 | Does combining supervised and unsupervised evidence help? | §6.4 |
| RQ5 | Does it transfer: which parts of the shift from LEMIS to CITES can be measured, how much does each cost, and which adaptation steps help? | §6.6 |
| RQ6 | What does it surface in CITES, and is that plausible against independent evidence? | §6.7 |

---

## Part A. The pipeline

### At a glance

| Stage | Current pipeline | After the rebuild (settled) | Open |
|---|---|---|---|
| **0 · Select records** (new) | All LEMIS line items; all CITES animal records, both reporter types | LEMIS animals with a class; CITES rule D, Appendix `N` removed, source `I` / purpose `L` out of the ranking | — |
| **1 · Align** | Codes mapped by name; `P`/`O`, `Y`/`N` and some countries unmapped; latest appendix | Full crosswalk; higher taxa one way; appendix in force; product groups by coding (done on branch `rework`) | — |
| **2 · Features** | Peer norms per population; peer groups on the current product groups | Peer groups on the new product groups and higher taxa; CITES described against LEMIS norms (S2.1); permit ID carried, not a feature | feature selection (B8) |
| **3 · Supervised** | XGBoost on all line items, chosen in advance | XGBoost (B5) and a 100-tree random forest (B6), no class weights, trained on animals with a class | — |
| **4 · Unsupervised** | Peer deviation, Isolation Forest, K-Means, refitted on all CITES records | Not part of the score: no detector improved it (B7). The peer-deviation score is shown as context | — |
| **5 · Combine** | Meta-learner over all four signals, fitted on 2011–12, reused on CITES; reasons from the supervised model | The average of the two models' percentile ranks (B6) | S5.2 reasons |
| **6 · Rank and review** (new) | Line items, within each year | Permits (consignments) scored by their highest record, ranked within each year into tiers by budget; flag card with context and a US-law warning | — |

### Stage 0 · Select records (new)

Serves RQ1 and RQ6. The current description of the system has no such stage; the alignment
created it.

Settled:
- LEMIS: train on animals with a class, dropping plants, fungi, bacteria and rows without a class
  (ALIGNING §2.2; notebook 04).
- CITES: score rule D's records: every exporter-reported record, plus importer-reported records
  with no exact exporter-reported counterpart (ALIGNING §2.2; notebook 05).
- CITES: remove Appendix `N`; keep source `I` and purpose `L` out of the ranking (ALIGNING §1.2).
- CITES counts and shares (UNODC legal-trade shares, risk-pattern pages) use exporter-reported
  records only (ALIGNING §1.2).

No open questions.

### Stage 1 · Align

Serves RQ1.

Settled (ALIGNING §2.1; code changes in ALIGNING §3):
- LEMIS `P` → `O` (pre-Convention) and `Y` → `N` (reintroduction).
- Country codes `VS`, `ZZ`, `ZR`, `TP` mapped; CITES `microgrammes` converted to mg.
- One higher-taxon convention: CITES `Genus` filled from `Taxon` when blank.
- Class names mapped (`Actinopterygii` = `Actinopteri`, …).
- Appendix: the listing in force in the shipment year, genus fallback, Appendix `N` ignored.
- Fields with no counterpart (value, port, month, companies, permits) stay out of the shared
  model.

**S1.1 Product groups.** *Decided* (31 October 2026, notebook 06). RQ1. One leather group for
small and large leather goods, shoes and trim; live rock joins raw coral; plant codes go to
`other` and CITES *fibres* to `hair`. The remaining tenfold gaps are differences in trade (G5).
The notebook also found that the appendix feature's genus fallback was wrong for named species,
and the rule was corrected (G6; ALIGNING §2.1). The original question, for the record:
- Why open: the current 31 groups match names. LEMIS shoes (220,574 line items) and live rock
  (19,624) sit in groups no CITES record falls into (ALIGNING §3 #14, X5). The rule is already set
  in ALIGNING §2.1: group by how each vocabulary codes the object. This question applies it.
- Keep caviar, extracts, ivory, horn and bone apart (L9).
- Draft rule: no group holds at least 0.1% of one dataset and nothing in the other. Every group
  whose shares differ more than tenfold is explained in ALIGNING as a coding difference (fix it)
  or a difference in trade (keep it).
- Peer groups and combination rarity are built on product groups, so this comes first.

### Stage 2 · Features

Serves RQ2 and RQ5.

Settled:
- Peer groups (genus × product group × unit) use the new product groups and the higher-taxon
  convention.
- The appendix feature is the listing in force.
- The export permit ID is carried through for grouping and comparing records, not used as a
  feature (LEMIS has none).
- Which features the chosen models keep is decided in Part B (B8).

**S2.1 Peer norms on CITES: CITES's own or LEMIS's?** *Decided: LEMIS norms* (31 October 2026,
notebook 08). RQ5. Norms learned from the target helped on three regional holdouts but were
significantly worse on Oceania (−0.066 PR-AUC) and on the time holdout (−0.007), so the rule
fails and CITES records are described against LEMIS's training-year norms. ALIGNING §1.2 step 11
and `cites_features.py` changed accordingly; the decision is rechecked with Part B's model in C6.
The original question, for the record:
- Why open: the design re-learns peer norms on CITES, on the argument that relative features
  transfer better (report §4.2). But in the regional holdout, re-normalising on the target did not
  help (PR-AUC 0.528 vs 0.539, §6.6 A).
- Test on the regional holdout and on the US-overlap set (C5) if it is usable.
- Draft rule: CITES norms (from rule D's records) are kept only if no worse than LEMIS norms on
  every test set.
- This is the one open question that can change a rule in ALIGNING: if LEMIS norms win, §1.2 step
  11 changes.

**S2.2 Consignment features.** *Decided: `cons_sources` adopted* (31 October 2026, notebook 09);
*then dropped by B8* (notebook 12: it adds nothing to the final ensemble).
RQ2. Of five candidate features, only the numbers of purposes and of sources per consignment
mean the same thing in LEMIS shipments and CITES permits; the number of purposes is constant in
LEMIS training (every shipment has one), so only the number of distinct sources carries
anything. It raises PR-AUC on listed 2011 line items by 0.002, with an interval just above zero:
it passes the rule narrowly, and B8 tests it again. The original question, for the record:
- Why open: a LEMIS shipment groups line items with one exporter, date and set of companies (§3.2,
  L1). A CITES export permit groups records the same way (M2, M9). Features describing the
  consignment (number of line items, number of taxa, mix of purposes and sources) could be learned
  on LEMIS shipments and computed on CITES permits.
- Caveat: a LEMIS shipment also holds non-CITES items, while a permit holds only listed
  specimens, and each CITES record is one country's report.
- Draft rule: adopted only if (a) the features are distributed comparably on LEMIS listed line
  items and on CITES permits (exporter-reported), and (b) they improve the primary metric of Part
  B (B4).
- If adopted, the limitation "items travelling on one permit" (ALIGNING §2.3, report §8 item 7)
  narrows. The two countries' reports and re-export chains still cannot be learned.

### Stage 3 · Supervised

Serves RQ2.

Settled:
- Trained on animals with a class (stage 0), on the shared fields only. A model that adds the
  LEMIS-only fields (value, port, month) is kept only as the RQ1 comparison (C3), not deployed.
- **Which model, how it is trained and tuned: Part B (B2–B6).**

**S3.1 A domain-adversarial network (DANN).** *Decided: not pursued* (31 October 2026). RQ5.
- Origin: the ChatGPT discussion. DANN targets the domain shift by learning features that cannot
  tell LEMIS from CITES.
- Not pursued because the MLP was dropped from the candidates (B2), and a DANN needs an MLP
  baseline: without one, a gain from adversarial training cannot be told apart from a gain from
  using a neural network. It belongs in report §9 (future work).
- If it is taken up later, our findings constrain it:
  - its domain discriminator must see raw fields only; otherwise it separates the datasets through
    the peer-group statistics (fingerprinting, §6.6 B) and training removes the relative features;
  - its invariance strength has to be tuned on the regional holdout, because CITES has no labels;
  - the largest shifts are in appendix, class and product form (§6.6 B), which are also
    predictive, so forcing invariance may cost signal.

### Stage 4 · Unsupervised

Serves RQ4.

Settled (B7, notebook 11, 31 October 2026):
- No unsupervised detector improved the ranking once XGBoost and the random forest were in it, so
  none is part of the score. Isolation Forest, K-Means, the Gaussian mixture and DBSCAN are
  reported as tested and rejected (RQ4).
- The peer-deviation score, the strongest detector on its own and the most transparent (how
  unusual a record's quantity, route, purpose and source are for its peers), is shown on the flag
  card as context, computed with the LEMIS norms (S2.1).

### Stage 5 · Hybrid and reasons

Serves RQ4 and RQ3.

Settled (B6, notebook 11, 31 October 2026):
- The score is the unweighted average of the percentile ranks of XGBoost and the random forest
  (100 trees of depth 16). A fitted meta-learner was no better than the average.
- Percentile ranks need a reference population. On CITES both models' scores are ranked within
  each year (as the tiers are), averaged, and ranked again.

**S5.1 Do the stack's weights hold on CITES?** *Moot* (31 October 2026): the score is a rank
average with no fitted weights to reuse, and no detector is in it. The original question, for the
record:
- Why open: the weights are fitted on LEMIS (currently supervised 0.85, Isolation Forest 1.18,
  peer deviation 0.36, K-Means 0.01). On CITES the detectors and their percentile scales are
  refitted (`score_cites.py`), but the weights are reused, so a percentile may not mean what it
  meant on LEMIS.
- Test within LEMIS on the regional holdout (§6.6 A): refit the detectors on the target region,
  reuse the source weights, and compare with the supervised part alone and with weights fitted on
  target labels. Repeat on the US-overlap set (C5) if usable.
- Draft rule: the stack is used on CITES only if the reused weights are no worse than its
  supervised part alone on the target.

**S5.2 Where the reasons come from.** *Decided* (31 October 2026): reasons from XGBoost's exact
contributions, with the peer-deviation signal beside them; the report says the reasons explain one
of the two models. RQ3. The options, for the record:
- Why open: the plain-language reasons come from XGBoost's TreeSHAP contributions (report §4.6),
  but the score is now the average of XGBoost's and the random forest's ranks (B6). Reasons from
  XGBoost alone explain half of the score.
- Options: (a) keep XGBoost's contributions as the reasons, and say in the report that they
  explain one of the two models; (b) add the forest's contributions through TreeSHAP for
  scikit-learn forests (the `shap` package), averaging the two models' contributions per
  feature; (c) as (a), with the peer-deviation context (stage 4) shown beside the reasons, so the
  reviewer also sees how unusual the record is for its peers.

### Stage 6 · Rank and review (new)

Serves RQ3 and RQ6. The current description ends at "risk score and review tier"; the alignment and
notebook 05 made the ranking unit and the review view design questions of their own.

Settled:
- Records are ranked within each year (CITES reports lag, C3); priority = top 1%, review = next
  4%.
- Records already marked as confiscated (source `I`) or law enforcement (purpose `L`) are set aside.

**S6.1 The review unit: line items or permits.** *Decided: permits* (31 October 2026, notebook
14). RQ3. A consignment's score is its highest record score (the maximum tied with the mean of the
top two and beat noisy-or on LEMIS shipments). The queue ranks CITES export permits; a record
without a permit stands alone. On LEMIS 2011, a 1% budget of shipments put a case behind 410 of
411 slots, while ranking line items spent slots on repeat lines of the same shipments. The
original question, for the record:
- Why open: under rule D, 10.9% of top-1% slots repeat a permit already in the top 1%, and 85% of
  the multi-line permits in it are split by the cut.
- LEMIS shipments are the equivalent of permits, and the pipeline already scores a shipment as the
  maximum over its line items (`analysis.py`). So the rule that combines line scores into one
  consignment score **can** be tested on LEMIS. Report §4.8 says it cannot; that sentence needs
  correcting.
- Draft rule: compare the maximum, the mean of the top two, and noisy-or on shipment-level recall
  at a 1% and 5% budget on the listed selection records (B1); choose the best, ties to the simplest
  (maximum). Then decide the unit from M9 and that result.

**S6.2 What the flag card shows beyond the score.** *Decided: context only* (31 October 2026):
the consignment's other records, and the importer's report beside the exporter's where both
exist, without scoring their disagreement. RQ3. The options, for the record:
- Why open: signals only CITES has, such as the two countries' reports of a consignment
  disagreeing (M8) and one-sided reporting (M5, M6), cannot be learned from LEMIS (ALIGNING §2.3).
- Options: show them as context (both reports side by side, the consignment's other lines); or
  add mirror disagreement as an unsupervised signal. M8 is partly mechanical and cannot be
  validated with labels, which argues for context only.

**S6.3 US-specific taxa.** *Decided: a warning on the flag card* (31 October 2026): marine-mammal
products stay ranked, with a note that the pattern may reflect US law when the importer is not
the US; the report measures their share of flags. RQ6. The options, for the record:
- Why open: marine-mammal products are refused at ten times the average rate in LEMIS, consistent
  with the US Marine Mammal Protection Act. They make up 12.7% of current priority flags, and the
  Greenland → Denmark route is flagged in 95% of its records (§6.7). Report §8 item 3 already
  recommends importer-specific rules.
- Options: a warning on the flag card; ranking them separately when the importer is not the US;
  leaving them as they are and reporting the effect.

### The corrected description of the system

The app's "How it works" text and report §4.1 change as follows once the open questions are
decided. Brackets mark wording that depends on a decision.

1. **Select records.** Learn from LEMIS animal imports; score CITES records once per trade where
   possible (rule D), setting aside Appendix N and records already marked as confiscated or law
   enforcement.
2. **Align.** Map LEMIS and CITES codes onto one schema: product forms grouped by what the object
   is, units, source and purpose codes (including the two that differ), countries, higher taxa, and
   the CITES listing in force at the time of the shipment.
3. **Features.** Deviation from the peer norms of LEMIS's training years:
   quantity against peers, route rarity, unusual purpose or source [, consignment make-up].
4. **Supervised.** Two models, chosen from six candidates, learn from LEMIS outcomes (refused,
   seized, abandoned or re-exported vs cleared): XGBoost (boosting) and a random forest (bagging).
5. **Unsupervised.** A peer-deviation score shows how unusual a record is for its peers. It is
   context for the reviewer, not part of the score: no unsupervised detector improved the ranking.
6. **Combine.** The risk score is the average of the two models' percentile ranks. The reasons
   come from XGBoost's contributions, shown with the peer-deviation signal.
7. **Rank.** Consignments (CITES export permits), each scored by its highest record, are ranked
   within each year into review tiers by inspection budget. The flag card shows the consignment's
   records, both countries' reports where they exist, and a warning where a pattern may reflect
   US law.

The paragraph on what a flag means still holds. It gains one sentence: some signals exist only in
CITES, such as the two countries' reports of a consignment disagreeing, and cannot be learned from
US enforcement outcomes.

---

## Part B. Model selection

How the supervised models, the detectors and their combination are chosen. Every step is decided
on LEMIS, where labels exist; the chosen configuration then has to pass the transfer test (C6)
before it is deployed on CITES.

**Principles.**
- Candidates are chosen for what the research questions need, not to use every method taught. The
  project requirement (at least three of: supervised/unsupervised learning, machine or deep
  learning, hybrid/ensemble, intelligent sensing) is already met by the design's structure.
- A candidate left out is left out for a stated reason, which the report gives.
- Rules are written before results, as in notebooks 04 and 05. Ties go to the simpler model.
- The test years are used once, at the end (B10).

**B1 What each year range is used for.** *Decided* (31 October 2026). RQ2.

The current pipeline uses 2011–2012 for early stopping, thresholds and fitting the meta-learner at
once, and the choice between the weighted and unweighted transfer models (§6.6 C) was made on
test-year records. Tuning, choosing and fitting the stack on
the same records overstates how well the choice will do. Proposed roles:

| Years | Used for | Line items | Cases (listed species) |
|---|---|---|---|
| 2000–2010 | Training. Hyperparameters tuned by time-ordered cross-validation inside these years (expanding window: for example, train to 2006 and validate on 2007–08; train to 2008 and validate on 2009–10). Peer norms are refitted on each fold's training part, so they never see the fold's validation years. Early stopping uses the fold's validation part. | 3,562,032 | 63,297 (22,179) |
| 2011 | Selection: comparing the tuned candidates (B5), building the stack (B6), choosing detectors (B7) and features (B8). | 370,113 | 7,294 (2,417) |
| 2012 | Fitting the stack's weights, the review thresholds and the confidence bands. | 391,609 | 6,898 (2,403) |
| 2013–2014 | Test, once (B10). | 783,365 | 12,399 (4,744) |

Counts are LEMIS animals with a class; "listed species" means a CITES listing in force in the
shipment year. Each range has enough cases for its job: 2,417 listed cases give bootstrap
intervals of roughly 1–2 points for the comparisons in 2011, and 6,898 cases are ample for a
stack of a handful of inputs in 2012.

Within each range, the metric that decides (B4) is computed on line items of species with a CITES
listing in force (notebook 04's choice), with all line items reported alongside. This also settles
the earlier question of which validation records to use.

For feasibility, the search runs on a sample of shipments (for example 20%, whole shipments kept
together); the chosen configurations are then refit on all of 2000–2010.

**Which model is deployed on CITES.** *Decided: refit on 2000–2014* (notebook 13, R3–R4: PR-AUC
falls from 0.570 to 0.512 as the gap grows from 1 to 7 years). The tested model is trained on 2000–2010, but CITES is scored
for 2015–2024, so more recent training years could matter. C4 measures how performance changes
as the gap between training and test years grows. Rule: if PR-AUC on listed line items falls by
more than its bootstrap interval between the shortest and the longest gap, the chosen
configuration is refit on all of 2000–2014 for deployment, and the report says that the deployed
model differs from the tested one in its training years only. Otherwise the tested model is
deployed unchanged.

**B2 Supervised candidates.** *Decided* (31 October 2026). RQ2. The data: about 5.1 million
training line items, 1.7% positives, mostly categorical features with many values (genus,
countries) plus a few numeric deviation features, and about 5.5 million CITES records to score.

How well a candidate copes with the class imbalance depends mainly on how it is trained (class
weights, resampling), not on the algorithm; B3 compares those options for every candidate, and
every candidate below accepts class weights.

| Candidate | In or out | Reason |
|---|---|---|
| Logistic regression | In | Interpretable linear baseline; trained by gradient-based optimisation, so it carries the optimiser comparison (B4) (PSUPR Day 1b) |
| Naive Bayes (categorical) | In, as a baseline | Fast; its independence assumption is broken here (for example `purpose_share_in_genus` derives from purpose), and how much that costs is itself a result |
| Decision tree | In | Interpretable; its high variance is what motivates bagging and boosting (PRMLS Day 5) |
| Random forest | In | Bagging with random feature subsets: reduces variance (PRMLS Day 5) |
| AdaBoost | In | Boosting by reweighting (PRMLS Day 5). Each round gives more weight to the records it got wrong, which are mostly the rare cases, so it suits imbalance; but it also upweights noisy labels (paperwork refusals, undetected cases). The comparison with gradient boosting shows which effect wins |
| Gradient-boosted trees (XGBoost) | In | Boosting by gradient steps: reduces bias; handles categorical features natively; less sensitive to label noise than AdaBoost's exponential loss (PRMLS Day 5) |
| Linear SVM | Out | Another linear decision boundary, differing from logistic regression only in its loss; it would add little to the comparison |
| MLP | Out | Training cost on millions of rows, for a model that addresses no weakness identified so far. Without it there is no baseline for a domain-adversarial network, so S3.1 is not pursued |
| Kernel SVM | Out | Training cost grows at least with the square of the number of rows; infeasible at 5 million |
| k-nearest neighbours | Out | Scoring 5.5 million records against 5 million stored ones is impractical, and distances over one-hot categories stop separating records (curse of dimensionality, PSUPR Day 3) |

**B3 Preparing the data for each candidate.** *Encodings and scaling decided (31 October 2026);
imbalance handling to test.* RQ2.
- **Categories:** native for XGBoost; one-hot over the most frequent levels for logistic
  regression; categorical counts for naive Bayes, with numeric features binned; integer codes or
  one-hot for the decision tree, random forest and AdaBoost.
- **Scaling:** z-score standardisation for logistic regression, fitted on the training part only
  and reused unchanged on later years and on CITES (PSUPR Day 1b). Not min-max: the numeric
  features are heavy-tailed, and min-max is set by the extremes. The tree models and naive Bayes
  need no scaling.
- **Class imbalance** (*to test*, per candidate, in the cross-validation). PSUPR Day 1b lists the
  options; the one with the best fold metric is kept:
  - class weights in the loss;
  - random undersampling of cleared line items;
  - SMOTE-NC (SMOTE for mixed categorical and numeric data);
  - for the ensembles, their imbalance-specific versions: balanced random forest (each tree trained
    on a balanced bootstrap sample) and RUSBoost (AdaBoost with undersampling in each round).

  SMOTE-NC and the two ensemble versions need the `imbalanced-learn` package.

**B4 Tuning, optimisation and the metric that decides.** *Decided* (31 October 2026). RQ2, RQ3.
- **Search:** five designed configurations per candidate, scored on the cross-validation folds of
  B1 (changed on 31 October 2026 from a random search of 20, to keep the search to about an hour):
  - configurations 1–4 compare the class-imbalance options of B3 at standard settings (for
    XGBoost, the current pipeline's settings);
  - configuration 5 changes the candidate's main setting (tree depth, number of trees, smoothing),
    using the imbalance option that did best among 1–4; for logistic regression it swaps the
    optimiser instead (below).

  The settings are in `notebooks/model_selection.py` (`BASE`, `IMBALANCE`, `VARIANT`). Five
  configurations explore less than twenty, so a candidate could lose for want of tuning; the
  report says so.
- **Optimisation, per model:**
  - logistic regression: full-batch quasi-Newton (L-BFGS) in configurations 1–4, stochastic
    gradient descent (SGD on the log loss, with a learning-rate schedule) in configuration 5, with
    the same class-imbalance handling, so the two optimisers are compared like for like; L2
    regularisation in both.
    This is the one candidate where the optimiser is a real choice; adaptive optimisers such as
    Adam matter for neural networks, which are not among the candidates (PRMLS Day 1b);
  - decision tree and random forest: greedy splits, no gradient;
  - AdaBoost: stagewise reweighting of the training records, no gradient;
  - XGBoost: second-order gradient boosting, early stopping on the fold's validation part.
- **The metric that decides: PR-AUC on listed line items,** with recall at a 1% and a 5% budget
  reported alongside for RQ3. The same metric scores the folds during tuning.
  - Accuracy is not used: clearing everything scores 98.3% (report §3.2).
  - ROC-AUC flatters a model when positives are rare.
  - F1 needs a threshold, and the CITES ranking has none.
  - Recall at 1% was the first proposal, but among listed line items cases are 2.4% of records,
    so a 1% budget has fewer slots than there are cases. Recall at 1% cannot exceed about 0.42 in
    2011 (0.49 in the test years); notebook 04's candidates reached 0.413–0.421, about 85% of that
    ceiling, so strong models are squeezed together. PR-AUC covers the whole ranking, does not
    depend on a chosen budget, is the standard measure when positives are rare, and is the
    report's existing headline metric.
- **Consequence for notebook 04:** it chose the training population on recall at 1%. When it is
  rerun on the rebuilt data (Part C), it is also judged on PR-AUC, and the report says that the
  metric changed and why.

**B5 Choosing among the supervised candidates.** *Decided: XGBoost* (31 October 2026, notebook
10). RQ2. On listed 2011 line items random forest (PR-AUC 0.570) and XGBoost (0.565) were tied
(difference −0.005, interval −0.012 to +0.002); all other candidates were significantly worse.
The rule below chose random forest. After seeing the tie, the choice was made on deployment cost
instead: random forest would add roughly 340 MB to the hosted app and need an extra package for
its reasons, while XGBoost is 1 MB with exact contributions built in. The chosen configuration
uses no class-imbalance handling, depth 6, learning rate 0.05 and minimum child weight 20.
Notebook 10 also found that class weights lowered PR-AUC for every strong candidate (T3), and
that the strong candidates' errors are highly correlated (T5). The original rule, for the
record:
- Each tuned candidate is scored on 2011 (B1).
- Draft rule: the best on the B4 metric is chosen; a candidate whose paired-bootstrap interval
  against the best includes zero counts as tied; ties go to the simpler model, in the order
  logistic regression, naive Bayes, decision tree, random forest, AdaBoost, XGBoost.
- The comparison table of all candidates is reported (RQ2): it shows what pattern recognition adds
  over simpler methods.

**B6 Building the stack.** *Decided* (31 October 2026, notebook 11): XGBoost and a random forest
(100 trees, depth 16), combined by the average of their percentile ranks. RQ4. Forward selection
added the random forest (+0.0098 PR-AUC with the 200-tree forest of B5) and then stopped: no
detector and no other model improved the stack. The meta-learner and the rank average tied, so
the average is kept. The forest was then reduced to 100 trees of depth 16 for the hosted app
(27 MB instead of 96 MB); the ensemble still gains +0.006 (interval +0.003 to +0.010). The plan,
for the record:
- **One stacking layer.** Its pool holds the tuned supervised candidates and the detectors of B7.
  This replaces the current fixed stack of one supervised model and three detectors.
- **Error correlation first.** For the strongest candidates, report how correlated their errors
  are on 2011 (for example, the overlap of their top-1% sets and the correlation of their scores on
  enforcement cases). Ensembles gain most from members whose errors are weakly correlated
  (PRMLS Day 5).
- **Forward selection.** Start from the B5 choice. Add the pool member that most improves the B4
  metric on 2011. Stop when the best addition's gain lies inside its bootstrap interval. This is
  the answer to "how many models": as many as keep helping, usually few. During selection the
  members are combined by the logistic meta-learner over the logit of each member's percentile
  rank, cross-fitted on 2011 by halves of the shipments; the final weights are fitted on 2012.
  (Changed on 31 October 2026, before any stack result: the plan was to average ranks during
  selection, but an unweighted average gives a weak detector XGBoost's weight, so no detector
  could ever enter. Averaging is still compared at the end, below.)
- **How the members are combined:** a logistic-regression meta-learner (stacking) against simple
  rank averaging (unweighted voting, PRMLS Day 5). Averaging is kept if it ties. The logistic
  meta-learner gives exact per-member contributions, which S5.2 can use.

**B7 Unsupervised candidates.** *Decided* (31 October 2026, notebook 11): none is part of the
score; the peer-deviation score is shown as context. RQ4. Alone, every detector ranks enforcement
cases above random (peer deviation 0.151, Isolation Forest 0.111, DBSCAN 0.094, Gaussian mixture
0.086, K-Means 0.038; random 0.024), but none improves the supervised ensemble. The plan, for the
record:

| Candidate | In or out | Reason |
|---|---|---|
| Peer-deviation score | In | Transparent statistical baseline, readable as reasons |
| Isolation Forest | In | Scales to millions of records; isolates rare combinations |
| K-Means distance | In | Clustering-based (PSUPR Day 4); k chosen by silhouette and stability. Currently adds nothing (weight 0.01, §7) |
| Gaussian mixture | In | Soft clustering; low likelihood marks an anomaly; components chosen by BIC |
| DBSCAN | In if it scales | Density-based, with noise points as anomalies (PSUPR Day 4). Fitted on a sample; new records scored by distance to the nearest core point. Dropped with that reason if it cannot score 5.5 million records |
| Local outlier factor, one-class SVM | Out | Same scaling problem as k-nearest neighbours and kernel SVM |
| Autoencoder | Out | A neural network; dropped with the MLP (B2) |

- Inputs: the numeric deviation features, z-score standardised (they are few, so no dimensionality
  reduction is needed).
- Tuning without labels: silhouette and stability for K-Means, BIC for the Gaussian mixture,
  neighbourhood size for DBSCAN, on the training years.
- Judged on 2011 in two ways: how well each ranks enforcement cases on its own (expected weak;
  unusual is not illegal), and whether it enters the stack in B6.
- If no detector enters the stack, that is reported as the RQ4 result. The hybrid is then the
  supervised ensemble alone, and RQ4's answer is no.

**B8 Feature selection.** *Decided* (31 October 2026, notebook 12): the 21 shared features; the
consignment feature of S2.2 is dropped. RQ2, RQ5. Removing any other group significantly lowers
PR-AUC on listed 2011 line items (most: countries −0.023, quantity −0.012), while the consignment
feature adds nothing to the ensemble (+0.0003). Almost every feature shifts strongly between LEMIS
and CITES (PSI above 0.25; appendix 9.4), but no group can be dropped for transfer: C6 also
requires a gain on the listed held-out set, and every remaining group helps there. The plan, for
the record:
- Features are tested in groups, not one by one (route features, rarity features, appendix,
  genus, countries, quantity deviation, consignment features if S2.2 adopts them).
- Importance (embedded: gain for tree models, coefficient size for linear ones) and mutual
  information with the label (filter) rank the groups (PSUPR Day 3).
- Draft rule: backward elimination on 2011. A group is dropped if removing it lowers the B4 metric
  by less than its bootstrap interval.
- WildGuard adds a second criterion: how far a feature shifts between LEMIS and CITES (population
  stability index against rule D's records). A group that is predictive but shifts strongly is
  tested in C6: it is dropped if the model transfers better without it.

**B9 Methods considered and not used.** *Decided* (31 October 2026).
- **Dimensionality reduction (PCA, SVD).** Not inside the model. Tree models do not need it; PCA
  on one-hot categories is not meaningful; and mixed components would break the plain-language
  reasons (report §4.6). Used once, to show the LEMIS–CITES shift as a projection of the numeric
  features (C4).
- **Association rules (Apriori, Eclat).** Not used. Combination rarity already measures how rare a
  product × purpose × source combination is, which is the support of a three-item itemset, so a
  rule miner would add no signal the features lack.

**B10 The final fit and the one test.** *Settled once B1–B8 are decided.*
- Refit the chosen members on all of 2000–2010; fit the stack's weights, thresholds and bands on
  2012.
- Check transfer (C6). Score the test years 2013–2014 once. No choice is revisited after that.

**The flow diagram for the report.** Drawn from the decisions in section "Decisions", not before:
research question → what the system must do → candidates → test and rule → choice, for each of
B2, B6, B7 and B8, and for the adaptation steps of C6.

---

## Part C. The evaluation

Settled:
- Split by shipment year with the roles of B1; line items of a shipment kept together; results per
  line item and per shipment (ALIGNING §1.1).
- The listing-rule baseline uses the appendix in force; it currently uses the latest appendix.
- Domain classifiers use raw fields only (fingerprinting, §6.6 B).
- Decisions D and B were made with the current features: rerun notebooks 04 and 05 on the rebuilt
  features as a check (their rules are already written). Notebook 04 is also judged on PR-AUC
  (B4). Before the rerun, `notebooks/experiment_setup.py` should read the appendix, `animal`,
  `sp_listed` and `rule_d` from the rebuilt aligned files instead of deriving them: it still uses
  the old genus fallback (notebook 06, G6) and maps CITES codes back to LEMIS ones.

**C1 Which held-out sets are reported, and which answers RQ2.** *Decided* (31 October 2026). RQ2.
- Why open: the headline PR-AUC (0.599) is on all LEMIS test records, where personal imports are
  49.6% of cases (L7); in CITES they are 3% of records (C6). On commercial records alone it is
  0.276 (§6.5).
- Decision: report four sets, all 2013–2014: all line items; species with a CITES listing in force
  (notebook 04's evaluation set); commercial; listed and commercial. RQ2 is answered on the listed
  set, with the others alongside.

**C2 The unit inspection effort is measured in.** *Decided* (31 October 2026). RQ3.
- Why open: the report's headline effort figures ("5% of shipments" catches 70%; 80% of cases
  needs "13% of shipments") are line-item metrics (0.700 and 0.130 in
  `outputs/lemis_evaluation.json`). Measured on shipments, the same model gives 73.0% and 8.9%
  (`outputs/lemis_supplementary.json`).
- Decision: measure effort in the unit a reviewer opens, shipments in LEMIS, consistent with S6.1.
  Line items are reported alongside.

**C3 What dropping the LEMIS-only fields costs.** *Measurement decided (31 October 2026); measured
in the rebuild.* RQ1.
- Why open: the report says restricting the model to shared fields "costs only 4% of PR-AUC" (full
  vs transfer model). But declared value is missing for all of 2014 and 44% of 2013 (L6), the two
  test years, so value can barely contribute there. The comparison understates what alignment
  gives up.
- Decision: measured on test records where value was recorded (the recorded part of 2013; 2014
  has none). The full and transfer models are compared on the same records, so value has the
  chance to contribute.
- This answers the second half of RQ1 ("what cannot be aligned") with a number.

**C4 How each part of the shift is measured.** *Decided* (31 October 2026). RQ5. This sets the structure of report
§6.6. It is the main answer to the point that the evidence for transfer is indirect: three of the
four parts can be measured with labels, and the fourth is stated as a limitation.

| Shift | Measured by | Status |
|---|---|---|
| Time | Within LEMIS: five-year training windows scored on the same year, 2011 (notebook 13): PR-AUC 0.570 at a 1-year gap, 0.512 at 7 years | Measured |
| Trade mix | Listed and commercial held-out sets (C1); the regional holdout (§6.6 A) | Partly done |
| Recording | Domain classifier on US-reported records (§6.6 B, AUC 0.771). The US-overlap labelled set (C5) proved unusable: most enforcement cases never reach the CITES record | Partly measurable |
| Jurisdiction | Not measurable: no labels outside the US. Indirect only (C7); stated as a limitation | Limitation |

A PCA projection of LEMIS and CITES (B9) illustrates the shift.

**C5 A labelled set in CITES's own format: the US overlap.** *Decided: not usable* (31 October
2026, notebook 07). RQ5. On the primary key, 72.0% of cleared line items have a matching
US-reported CITES record but only 39.1% of enforcement cases (seized 31%, abandoned 22%): the CITES
report records trade that took place, and most stopped shipments never entered. The gap (32.9
points) fails the rule, so S2.1, S5.1 and C6 are judged without this set. The finding itself
belongs in the report: a flag means a CITES record resembles trade that was stopped in the US, not
that the record was stopped (§1.4, §8). The original question, for the record:
- Why open: the US compiles its CITES report from the same declarations LEMIS holds (X11). 76% of
  2013–2014 LEMIS line items of listed species have a US-reported CITES record of the same year,
  species and exporter (M4). Attaching the LEMIS outcomes to those records would give the only
  labelled CITES records, for the one country where outcomes exist.
- Fix before looking:
  - the matching key;
  - how a CITES record matched to several line items gets its label (any case, or the case
    share);
  - that the set is evaluation only (2013–2014 are test years).
- First measurement: is an enforcement case as likely to have a CITES record as a cleared line
  item? Refused or seized shipments may never appear in the US report. Draft rule: the set is
  usable if case and cleared coverage differ by less than 10 points; otherwise the gap is reported
  as a finding about which labels can reach CITES at all.

**C6 Adaptation steps and the common rule.** *Decided* (31 October 2026, notebooks 12–13): no
adaptation step is adopted beyond those already decided, and the chosen ensemble passes the
transfer check: on no held-out region is it significantly worse than XGBoost alone. RQ5.
- Already decided: training population (notebook 04); importance weighting (rejected, §6.6 C);
  peer norms from the records being scored (rejected, S2.1, notebook 08).
- Decided without a further test: dropping strongly shifting feature groups (B8). Every
  remaining group significantly helps on the listed held-out set (notebook 12, F2), so none can
  pass this rule.
- To test: a small
  labelled target sample (simulated, §6.6 A).
- Considered and not pursued: a domain-adversarial network (S3.1).
- Common rule: a step is adopted only if it helps on the regional holdout **and** on the listed
  held-out set, and on the US-overlap set if usable. Importance weighting was judged the same way.
- The configuration chosen in Part B must also pass this before it is deployed (B10).

**C7 Pass criteria for the CITES checks, written before the rebuild.** *Decided* (31 October
2026). RQ6.
- Why: the current checks were designed alongside their results. Fixing what counts as a pass
  before the rebuild means the result cannot be judged after it is seen. A pass means the ranking
  is consistent with independent evidence, not that it is accurate: CITES has no labels.
- **Check 1, UNODC consistency.**
  - Groups: the 14 animal groups of UNODC's *World Wildlife Crime Report 2024* (Fig. 2.1),
    seizure shares renormalised to 100%.
  - Legal-trade share of each group: exporter-reported CITES records 2015–2021, the report's
    window (ALIGNING §1.2). Seizure-to-trade ratio = seizure share ÷ legal-trade share.
  - WildGuard's value per group: the share of its ranked records 2015–2021 in the priority tier.
  - **Pass:** Spearman ρ between the two is positive with a one-sided permutation p-value below
    0.05 (with 14 groups, roughly ρ ≥ 0.46), **and** higher than the ρ of the listing rule.
  - Groups with fewer than 500 records in the window are flagged in the table, because their
    shares are unstable (pangolins had 241).
- **Check 2, the confiscated-specimen proxy.**
  - Retrain the chosen supervised configuration without any source-derived feature (source, the
    wild-source flag, source share in genus).
  - Score the CITES records 2015–2024 that rule D selects, plus the source `I` records the ranking
    sets aside, and measure how well the score separates source `I` from the rest.
  - **Pass:** the lower end of the 95% bootstrap interval of ROC-AUC is above 0.5. The ROC-AUC and
    the share of source `I` records in the top 5% are reported either way.
- **Check 3, review of the top-ranked records.**
  - The 50 highest-ranked priority records 2015–2024, one per taxon (each taxon's highest).
  - Each is put into one of three categories defined now:
    - **(a) recognised risk pattern:** worked or high-value parts of a listed taxon (ivory, horn,
      bone, teeth, carvings, medicine or extracts); personal-purpose specimens of Appendix I taxa;
      pre-Convention or unknown source or origin claimed for products of a listed taxon;
    - **(b) a US legal regime carried abroad:** taxa refused in LEMIS mainly because of US law (for
      example marine mammals under the Marine Mammal Protection Act), in trade that does not
      involve the US;
    - **(c) no apparent reason.**
  - Both team members classify the 50 independently, and their agreement is reported.
    *Changed on 31 October 2026, before the classification was seen:* the categories are
    applied by a stated rule (`validate_cites.classify`) instead of two reviewers, so no
    agreement is reported. Records matching none of (a) and (b) fall in (c).
  - **Pass:** at least half in (a) and at most a fifth in (c). Category (b) is reported as the
    jurisdiction effect (C4, S6.3), neither a pass nor a fail.
- **Reported without a pass criterion:** the share of priority flags in marine-mammal taxa where
  the importer is not the US, against their share of ranked records (S6.3).

---

## Order of work

1. **Data rebuild (pipeline stages 0–2):** implement the settled items, decide S1.1, run the
   ALIGNING §2.4 checks. Part B's candidates are compared on this data. *Done on branch `rework`,
   31 October 2026: `prepare_data.py`, `check_alignment.py` (all checks pass), notebook 06.*
2. **C5:** build and test the US-overlap set. *Done, 31 October 2026: not usable (notebook 07).*
3. **Choices and plans:** all decided on 31 October 2026 (B1–B4, B9, C1–C4, C7; see Decisions).
4. **S2.1** (whose peer norms), then **S2.2** (consignment features). *Both done: LEMIS norms (notebook 08); `cons_sources` adopted (notebook 09).*
5. **Part B experiments:** B3 imbalance handling and B4 tuning → B5 supervised choice → B7
   detectors → B6 stack → B8 features.
6. **Transfer and review:** C6, S6.1; then the choices S5.2, S6.2, S6.3. (S5.1 is moot: B6 chose
   a rank average with no weights.) *Done, 31 October 2026 (notebooks 13–14 and your choices).*
   **Every design question is now decided.**
7. **B10:** final fit, transfer check, the one test. Rerun notebooks 04 and 05 on the rebuilt
   features. *The one test and the deployment refit ran on 31 October 2026 (pipeline, Part D);
   notebooks 04 and 05 still to rerun.*
8. **Report:** the flow diagram and §4 from Parts A and B and the decisions; §6 from Part C.

---

## Part D. Building it

The decisions above, turned into code changes on branch `rework`, in order. Each step is run and
checked before the next.

| Step | File | Change |
|---|---|---|
| D1 | `wildguard/config.py` | Year roles of B1: training 2000–2010, selection 2011, held back 2012, test 2013–2014, deployment 2000–2014 |
| D2 | `wildguard/models.py` | The shared representation and encodings of `notebooks/model_selection.py` move here; the chosen ensemble (B5–B6): XGBoost (configuration 5, 357 rounds, no class weight) and a random forest (100 trees, depth 16, minimum leaf 5); scores combined by the average of percentile ranks within the scored population. The peer-deviation score stays for context; Isolation Forest, K-Means and the meta-learner leave the deployed path |
| D3 | `wildguard/train.py` | Train on animals with a class (2000–2010); the one test (B10) on 2013–2014: the ensemble, its two members, logistic regression, the listing rule (appendix in force) and random ranking, on the four held-out sets of C1, per line item and per shipment (C2), with bootstrap intervals; C3 on the 2013 records with a recorded value; then refit the ensemble on 2000–2014 for deployment (B1) |
| D4 | `wildguard/cites_features.py` | CITES records described with the deployment model's LEMIS norms (S2.1; norms from 2000–2014) |
| D5 | `wildguard/score_cites.py` | Rule D records without source I or purpose L; percentile ranks within each year, averaged; consignment = export permit (a record without one stands alone), scored by its highest record (S6.1); tiers per year on consignments (priority top 1%, review next 4%); reasons from XGBoost's contributions with the peer-deviation signal (S5.2); a US-law warning for marine-mammal products with a non-US importer (S6.3); aggregates from exporter-reported records |
| D6 | `wildguard/validate_cites.py` | The three checks of C7 with their pass criteria; the top 50 records exported for the two-person review |
| D7 | `wildguard/domain_shift.py` | The shift measurements of C4: regional transfer of the ensemble, ageing, the domain classifier on raw fields against rule D's records, PSI per feature, a PCA projection (B9) |
| D8 | `wildguard/analysis.py`, `explain.py` | Calibration and worked examples on the new ensemble; reason texts for `genus_listed`, the US-law warning and the peer context |
| D9 | `app/app.py` | Only what the new outputs require, so the app runs; the interface itself changes after your instructions |
| D10 | `run_pipeline.sh`, notebooks 04–05 | The step order; rerun notebooks 04 and 05 on the rebuilt data as the check of Part C |

## Decisions

| Question | Decision | Date | Evidence |
|---|---|---|---|
| Mirror records (ALIGNING §2.2) | Rule D | 31 October 2026 | notebook 05, M1–M9 |
| Training population (ALIGNING §2.2) | Animals with a class | 31 October 2026 | notebook 04, P1–P5 |
| B1 Year roles | 2000–2010 training with time-ordered cross-validation; 2011 selection; 2012 stack weights; 2013–2014 test once. Refit on 2000–2014 for deployment only if C4 shows performance falling with the gap | 31 October 2026 | Case counts per range (B1 table) |
| B2 Supervised candidates | Logistic regression, naive Bayes, decision tree, random forest, AdaBoost, XGBoost. Out: linear SVM, MLP, kernel SVM, k-nearest neighbours | 31 October 2026 | Reasons in the B2 table |
| B4 Deciding metric | PR-AUC on listed line items; recall at 1% and 5% reported | 31 October 2026 | Recall at 1% is capped near 0.42 by the case rate |
| B4 Search size | Five designed configurations per candidate instead of a random search of 20 | 31 October 2026 | To keep the search to about an hour |
| S3.1 DANN | Not pursued; future work | 31 October 2026 | No MLP baseline (B2) |
| B3 Encodings and scaling | As listed in B3; imbalance handling still to test | 31 October 2026 | — |
| B9 PCA, association rules | PCA once, to show the shift (C4); no association rules | 31 October 2026 | Reasons in B9 |
| C1 Held-out sets | All, listed, commercial, listed and commercial; RQ2 answered on listed | 31 October 2026 | — |
| C2 Effort unit | Shipments, with line items alongside | 31 October 2026 | Headline figures were line-item metrics |
| C3 Cost of LEMIS-only fields | Measured on test records where value was recorded | 31 October 2026 | Value missing for 2014 and 44% of 2013 |
| C4 Shift measurement | Time, trade mix, recording measured; jurisdiction a limitation | 31 October 2026 | C4 table |
| C7 CITES check criteria | Pass criteria for checks 1–3 as written in C7 | 31 October 2026 | — |
| S1.1 Product groups | One leather group; live rock with raw coral; plant codes to `other`; CITES fibres to `hair` | 31 October 2026 | notebook 06, G1–G5 |
| Appendix genus rule | Genus decides only when no species is named; otherwise `genus_listed` | 31 October 2026 | notebook 06, G6 |
| Leech class | LEMIS `Clitellata` leech genera map to CITES `Hirudinoidea` | 31 October 2026 | ALIGNING §2.4 check 3 |
| C5 US-overlap set | Not usable as labelled CITES data | 31 October 2026 | notebook 07, U1–U4: cases covered 39.1% vs cleared 72.0% |
| S2.1 Peer norms on CITES | LEMIS training-year norms | 31 October 2026 | notebook 08, N1–N3: target norms significantly worse on 2 of 6 held-out populations |
| S2.2 Consignment features | `cons_sources` (distinct sources in the shipment or permit) | 31 October 2026 | notebook 09, K1–K4: +0.002 PR-AUC, interval just above zero |
| B5 Supervised model | XGBoost, configuration 5 (no imbalance handling, depth 6, learning rate 0.05) | 31 October 2026 | notebook 10, T1–T6: tied with random forest (0.565 vs 0.570); tie broken on deployment cost after the result was seen, not by the written rule |
| B6 Stack | XGBoost + random forest (100 trees, depth 16), average of percentile ranks | 31 October 2026 | notebook 11, H3–H7: forest adds +0.006 at this size (27 MB); size chosen for the hosted app |
| B7 Detectors | None in the score; peer-deviation score shown as context | 31 October 2026 | notebook 11, H2–H3: no detector improves the ensemble |
| S5.1 Stack weights on CITES | Moot: no fitted weights | 31 October 2026 | B6 |
| B8 Features | The 21 shared features; consignment feature dropped | 31 October 2026 | notebook 12, F1–F5 |
| C6 Transfer check | Ensemble passes; no further adaptation step | 31 October 2026 | notebook 13, R1–R2 |
| B1 Deployed model | Refit on 2000–2014 after the one test | 31 October 2026 | notebook 13, R3–R4: performance falls with the gap |
| S6.1 Review unit | Permits, scored by their highest record | 31 October 2026 | notebook 14, W1–W4 |
| S5.2 Reasons | XGBoost's contributions, with the peer-deviation signal beside them | 31 October 2026 | Choice |
| S6.2 Flag card | Context only: the consignment's records and both countries' reports | 31 October 2026 | Choice |
| S6.3 US-specific taxa | Warning on the flag card | 31 October 2026 | Choice |
| C7 Check 3 method | The categories applied by a stated rule instead of two human reviewers | 31 October 2026 | Team time; decided before the classification was seen |
| C7 Check 3 result | Fails: 62% (a), 8% (b), 30% (c), against at most 20% (c). All 15 (c) records are personal-purpose imports of wild Appendix II specimens (11 into the US, 4 into New Zealand), a pattern US enforcement acts on (personal imports refused 25.5% in LEMIS) but not one of C7's recognised patterns. Reported as a fail, not re-ruled | 31 October 2026 | `outputs/cites_top50_review.csv`, `cites_validation.json` |
| B10 The one test | Ensemble kept as decided: on 2013–2014 it scored PR-AUC 0.540 on listed line items, 0.006 below XGBoost alone (interval −0.009 to −0.003), with equal recall at the review budgets. Not revisited, so the test stays untouched; the report states it | 31 October 2026 | `outputs/lemis_evaluation.json` |
