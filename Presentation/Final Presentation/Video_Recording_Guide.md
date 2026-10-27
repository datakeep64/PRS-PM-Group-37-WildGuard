# Final video presentation — recording guide

**Target length:** 12–14 minutes (requirement: 10–15). **Format:** `.mp4`.
**Content required:** describe the project fully, the methods, and the metrics used to evaluate
its success; a demo is optional but included here.

## Run sheet

| # | Slide | Time | Speaker |
|---|---|---|---|
| 1 | Title | 0:00–0:40 | |
| 2 | The problem | 0:40–1:40 | |
| 3 | How we got here | 1:40–2:30 | |
| 4 | The idea: LEMIS labels, CITES scale | 2:30–3:30 | |
| 5 | Schema alignment | 3:30–4:30 | |
| 6 | System design | 4:30–5:40 | |
| 7 | Techniques mapped to the module | 5:40–6:30 | |
| 8 | Does it work? | 6:30–7:40 | |
| 9 | What an inspection budget buys | 7:40–8:40 | |
| 10 | Hybrid vs parts | 8:40–9:30 | |
| 11 | Does it transfer? | 9:30–10:30 | |
| 12 | What it surfaces in CITES | 10:30–11:30 | |
| 13 | Demo (live app) | 11:30–13:00 | |
| 14 | Limitations | 13:00–13:40 | |
| 15 | Conclusion | 13:40–14:20 | |

Fill in the speaker column to split the narration between the two of you. The narration for
every slide is in the speaker notes of `WildGuard_Final_Presentation.pptx` and is reproduced
below.

## Recording options

* **PowerPoint:** *Slide Show → Record → From Beginning*. Narrate over the slides, then
  *File → Export → Create a Video* (1080p). For the demo, record the app separately (below) and
  insert the clip on slide 13 with *Insert → Video*.
* **Zoom:** start a meeting with yourself, share the screen, *Record → Record on this computer*.
  Zoom writes an `.mp4` when the meeting ends. Two presenters can join the same meeting.
* **macOS:** press ⌘⇧5, choose *Record Entire Screen* and enable the microphone under *Options*.

## Demo segment (about 90 seconds)

Start the app before recording:

```bash
source .venv/bin/activate
streamlit run "SYSTEM CODE/app/app.py"
```

1. **Review queue** (`http://localhost:8501/?page=queue`). Show the ranked list. Filter the class
   to *Reptilia* or search a taxon such as *Psittacus*. Click a row and walk through its flag
   card: the record, the *Why flagged* reasons, and the four signals.
2. **Check a shipment** (`?page=check&example=tortoise`). The Indian star tortoise case opens
   already scored: two live wild tortoises, declared as personal imports from Malaysia at $400
   each. Point out the reasons (purpose, quantity against peers, unit value), then mention it was
   actually seized. Note on camera that this case is from a training year, so it is an
   illustration, not a test result.
3. **Does it work?** (`?page=evaluation`). Drag the budget slider from 1% to 10% and read off
   *cases caught* against random. Switch to *Commercial purpose only* to show the harder subset.
4. **Risk patterns** (`?page=patterns`). Show *Product form*, then scroll to the UNODC
   consistency chart.

## Narration (from the speaker notes)

### Slide 1 · Title  
*≈0.7 min at 140 words/min*

Hello, we are Group 37 and this is WildGuard. In one sentence: WildGuard takes what US wildlife inspectors actually decided about 5.5 million import declarations, learns the patterns that preceded enforcement action, and uses them to rank the much larger, unlabelled global CITES trade database so that human reviewers know where to look first. Over the next twelve minutes we will cover the problem, how our direction changed after the first presentation, the system we built, how well it works on held-out data, how it behaves when applied to CITES, a short demo, and its limitations.

### Slide 2 · The problem: more trade than anyone can inspect  
*≈0.9 min at 140 words/min*

Why is this a pattern-recognition problem? First, scale: the CITES Trade Database we downloaded holds over 29 million trade records since 1975, and about 800 thousand new animal-trade records arrive every year. No team can review them one by one. Second, the risk signal is rarely in a single field. It sits in combinations: species, product form, quantity, purpose, source and route, relative to what is normal for that species. That is exactly what pattern recognition is good at. Third, the need is officially recognised: the CITES Secretariat and the World Customs Organization developed guidance on risk assessment and inspections for CITES trade controls, taken forward at CoP20 in December 2025, but risk targeting is still done largely by hand, officer by officer.

### Slide 3 · How we got here: from image fusion to label transfer  
*≈0.7 min at 140 words/min*

At the first presentation we explained why we dropped our original proposal, which fused CITES records with pangolin images. Briefly: the image stream covered one species while the records covered all of them, so missing images would have distorted scores; it would mostly have re-discovered what is already known; and the image data was never accessible - the dataset request went unanswered and synthetic augmentation would have created a synthetic-to-real gap. The new direction keeps one modality, structured trade records, and uses two large, real, freely available datasets that complement each other.

### Slide 4 · The idea: LEMIS has the labels, CITES has the scale  
*≈0.7 min at 140 words/min*

LEMIS is the US Fish and Wildlife Service's record of every wildlife import declaration: 5.5 million line items from 2000 to 2014, each with what inspectors actually did - cleared, refused, seized, abandoned or re-exported. That gives us labels, but only for the United States. CITES covers every country but has no enforcement labels; the illegal-trade database is access-restricted. The two complement each other's gap, and because the US builds its CITES reports from the same kind of declarations, their codes line up. WildGuard learns on LEMIS and applies what it learned to CITES.

### Slide 5 · Making two datasets speak the same language  
*≈0.8 min at 140 words/min*

Schema alignment is our knowledge-modelling step. LEMIS writes product forms as three-letter codes, CITES as English terms; units differ; LEMIS has no CITES appendix. We built a crosswalk into 31 shared product groups and six unit groups, converted quantities, and looked up each LEMIS taxon's CITES appendix. Source and purpose codes are identical letter codes in both. To check the alignment we trained a classifier to tell LEMIS records from the records the US itself reported to CITES for the same years. Its AUC was 0.771 for CITES-listed taxa, against 0.967 for global CITES trade - so the mapping is consistent, and the remaining difference is real distribution shift.

### Slide 6 · System design  
*≈0.9 min at 140 words/min*

Here is the whole system. The top lane is LEMIS, where everything is trained and evaluated with real labels. Records are aligned, then described by relative features - how far a record sits from the norm for its own peer group of genus, product and unit, and how rare its route, purpose and source are. An XGBoost model learns from outcomes; three unsupervised detectors - a peer-deviation score, an Isolation Forest and K-Means distance - score anomalies without labels; a stacked meta-learner combines all four. The bottom lane is CITES. The same mapping is applied, peer norms are re-learned from CITES itself, the trained model and combiner weights are transferred, the label-free detectors are re-fitted on CITES, and every record is ranked into review tiers with plain-language reasons.

### Slide 7 · Pattern-recognition techniques, mapped to the module  
*≈0.6 min at 140 words/min*

The module asks for at least three of four aspects. We fully cover three of them, and add a sense-making layer on top. Supervised and unsupervised learning: XGBoost, Random Forest and Logistic Regression on outcomes, plus three label-free detectors. Machine-learning techniques: gradient boosting, bagging, clustering with silhouette-based model selection, and isolation-based outlier detection. Hybrid and ensemble: the tree ensembles themselves, and a stacked meta-learner that combines supervised and unsupervised evidence. Sense-making: turning scores into decisions a person can act on - TreeSHAP explanations on every flag, confidence-aware triage bands, and risk-pattern analytics.

### Slide 8 · Does it work? Held-out LEMIS shipments, 2013-2014  
*≈0.9 min at 140 words/min*

Because only LEMIS has ground truth, the rigorous test happens there. We trained on 2000 to 2010, tuned on 2011 and 2012, and scored 2013 and 2014, which the system had never seen. Accuracy would be meaningless here: only 1.6 percent of test line items were enforcement cases, so clearing everything scores 98 percent. Instead we ask an inspector's question. If you can only inspect the top 5 percent, WildGuard catches 70% of the real enforcement cases, versus 5 percent at random. Among its top 1 percent of flags, 79% were real cases. To catch 80 percent of cases you inspect 13% of shipments instead of 80 percent. And this is the transfer model - it uses only fields CITES also has.

### Slide 9 · What an inspection budget buys  
*≈0.7 min at 140 words/min*

This curve is the usability argument. The x-axis is how much of the flow you inspect, highest score first; the y-axis is the share of real enforcement cases you catch. WildGuard in blue climbs steeply; the grey diagonal is random inspection; the green line is the obvious manual rule - inspect the most endangered, wild-caught species first - which barely beats random. On the right are confidence-aware bands with thresholds fixed on the validation years: 56% of shipments could be auto-cleared while missing only 7% of cases, and a priority band of 2% of shipments holds 60% of all cases.

### Slide 10 · Hybrid beats either part alone - but unusual is not illegal  
*≈0.6 min at 140 words/min*

Does combining supervised and unsupervised evidence help? On the same held-out years, the unsupervised detectors alone reach a PR-AUC of 0.050 - about three times random, but far from useful on their own, because a lot of perfectly legal trade is unusual. The supervised model alone reaches 0.588, and the stacked hybrid 0.599. The gain is modest but consistent, and it appears in both the transfer and full variants. The meta-learner gives the Isolation Forest the largest anomaly weight. XGBoost also clearly beat Random Forest and Logistic Regression.

### Slide 11 · Does it transfer? A simulated shift we can actually score  
*≈1.0 min at 140 words/min*

CITES has no labels, so we cannot score transfer to CITES directly. Instead we simulated it inside LEMIS: we trained without any shipments exported from Asia, then tested on held-out Asia shipments. In-domain the model reaches 0.592. Naive transfer to the unseen region drops to 0.539. Re-learning the peer norms from unlabelled target data - exactly what we do for CITES - gives 0.528. Importance weighting gives 0.546. Adding just 1 percent of target labels gives 0.542, and a model trained on all target labels reaches 0.616. Importance weighting looked best in this simulation, but on held-out LEMIS records of CITES-listed species - the records most like CITES - it lowered PR-AUC from 0.630 to 0.600, so the deployed model is unweighted. The lesson: a few real target labels beat any label-free adaptation trick.

### Slide 12 · What WildGuard surfaces in CITES, 2015-2024  
*≈1.0 min at 140 words/min*

We scored 8.25 million CITES animal-trade records. Records already marked as confiscated were set aside - flagging what has already been seized adds nothing - and the rest were ranked each year into tiers: priority is the top 1 percent, review the next 4 percent. On the left is where the priority tier concentrates by product form: horn, shells, ivory, teeth and bone. Without labels we make no accuracy claim here; instead we ran two independent checks. First, against UNODC's World Wildlife Crime Report 2024: species groups that are over-represented in global seizures relative to their legal trade should be flagged more often - the rank correlation is 0.864 with p of 0.0001. Second, CITES marks confiscated specimens with source code I. A version of the model that never sees the source field still ranks those records higher than the rest, with an AUC of 0.874.

### Slide 13 · Demo: the WildGuard review tool  
*≈0.9 min at 140 words/min*

Now a short demo of the MVP, a Streamlit app. The review queue lists CITES records ranked for review; selecting one opens a flag card with the record, the reasons it was flagged, and the four signals behind the score. The 'Check a shipment' page scores any record - here the real Indian star tortoise case from LEMIS: two live wild tortoises declared as a personal import from Malaysia at 400 dollars each, where 99 percent of comparable imports were commercial lots of around ten animals at a median of 35 dollars each. It was seized. The 'Does it work?' page lets you move an inspection budget and see how many real cases you would catch.  [Switch to the live app for the demo segment.]

### Slide 14 · Limitations, stated plainly  
*≈1.1 min at 140 words/min*

Some honest limitations. First, a flag is not evidence of crime: our labels are enforcement outcomes, which include paperwork refusals, and absolute probabilities do not carry over to CITES, so there we rank by budget. Second, LEMIS ends in 2014 and reflects US enforcement priorities. Third, half of LEMIS cases are personal imports, while CITES is mostly commercial; on commercial shipments alone the model is weaker. Fourth, country features can encode where enforcement has looked before, so flags must stay a prompt for human review, never an automatic decision. Some labels reflect US law rather than CITES rules: marine-mammal products are refused ten times more often in LEMIS, consistent with the US Marine Mammal Protection Act, and the model carries that into trade between other countries where it may be lawful. Finally, the full model may slightly overstate performance because a zero declared value is more common in refused shipments; the deployed transfer model does not use value at all.

### Slide 15 · Conclusion  
*≈0.9 min at 140 words/min*

To conclude. Enforcement knowledge does transfer: using only fields CITES also has, inspecting five percent of held-out shipments catches 70% of real enforcement cases. Combining supervised and unsupervised evidence helps a little, and every learned model beats the obvious manual rule by a wide margin. The enabling step was unglamorous - aligning two public datasets so they speak one language. On CITES, where no labels exist, the ranking is directionally consistent with independent seizure statistics, which is the honest claim we can make: WildGuard is a triage aid for human reviewers, not a verdict. Next steps would be a small labelled sample from a CITES authority, newer LEMIS data, reporting mismatches as an extra signal, and a feedback loop from analysts. Thank you.

