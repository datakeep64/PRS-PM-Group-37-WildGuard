# Final video presentation — recording guide

**Target length:** about 13 minutes (requirement: 10–15). **Format:** `.mp4`.
**Content required:** the whole project, the methods, and the metrics used to judge its success; a
demo of the app is included (slides 13–14).

## The slides

The deck is the Claude artifact **WildGuard Final Presentation**
(https://claude.ai/artifact/BShBnjC8m2jY4b8oDkQhtM; private until shared from its Share menu). It has
16 slides, and each slide's narration is in its speaker notes (reproduced below).
Present it from the artifact page, or download it from the page's Share › Export menu as `.pptx`
or `.pdf` and save the files here as `WildGuard_Final_Presentation.pptx` and
`WildGuard_Final_Presentation.pdf`, replacing the 27 October versions.

## Run sheet

| # | Slide | Time | Speaker |
|---|---|---|---|
| 1 | WildGuard | 0:00–0:34 | |
| 2 | More trade than anyone can inspect | 0:34–1:16 | |
| 3 | LEMIS has the outcomes; CITES has the scale | 1:16–1:56 | |
| 4 | Two datasets, one shared schema | 1:56–2:40 | |
| 5 | Learn where outcomes exist, rank where they do not | 2:40–3:26 | |
| 6 | Nothing is tuned on the years it is judged on | 3:26–4:10 | |
| 7 | Every model choice made by comparison | 4:10–4:55 | |
| 8 | The one test: US imports 2013–2014, never seen | 4:55–5:39 | |
| 9 | What an inspection budget buys | 5:39–6:17 | |
| 10 | What helped, and what did not | 6:17–7:06 | |
| 11 | From CITES records to a review queue | 7:06–7:53 | |
| 12 | Checked against outside evidence: one pass, two fails | 7:53–8:47 | |
| 13 | The review queue | 8:47–10:06 | |
| 14 | Check any record, see the evidence | 10:06–11:15 | |
| 15 | Limits, stated plainly | 11:15–12:02 | |
| 16 | What WildGuard shows | 12:02–12:52 | |

Timings assume about 140 words a minute, plus half a minute of screen time on each demo slide.
Fill in the speaker column to split the narration between the two of you.

## Recording options

* **PowerPoint** (after downloading the `.pptx`): *Slide Show → Record → From Beginning*. Narrate
  over the slides, then *File → Export → Create a Video* (1080p). For the demo, record the app
  separately (below) and insert the clip on slides 13–14 with *Insert → Video*.
* **Zoom:** start a meeting with yourself, share the screen, *Record → Record on this computer*.
  Zoom writes an `.mp4` when the meeting ends. Two presenters can join the same meeting.
* **macOS:** press ⌘⇧5, choose *Record Entire Screen* and enable the microphone under *Options*.

## Demo segment (slides 13–14, about 2½ minutes)

Start the app before recording (or use the hosted app once the latest version is deployed):

```bash
source .venv/bin/activate
streamlit run "SYSTEM CODE/app/app.py"
```

1. **Review queue** (`http://localhost:8501/?page=queue`). The list opens on 2024's priority permits.
   Open *More filters* and tick *Only permits reported by both countries*: the first permit is
   sperm-whale carvings from Greenland to Denmark. Walk down its flag card: tier and rank in the
   year, *Why flagged*, *How unusual for its peers*, the US-law warning, and *On this permit*, where
   the two countries' reports disagree (carvings against teeth). Untick the filter and type
   *Crocodylus siamensis* in *Search taxon* to show the Thailand → Japan leather permits: the pattern
   that fills half the priority tier.
2. **Check a record** (`?page=check&example=saiga`). The saiga-antelope medicine record opens already
   ranked: priority, top 0.01% of 2019 permits, with its reasons. Optionally show
   `?page=check&example=tortoise`: the US-declaration form with value, port and month (a
   training-year case, so an illustration, not a test result).
3. **Does it work?** (`?page=evaluation`). Drag the budget slider from 1% to 10% and read off the cases
   found against random inspection and the listing rule. Switch *Shipments* to *Commercial imports*
   to show the harder subset, and mention the note that XGBoost alone scored slightly higher.
4. Optional, **What it flags** (`?page=patterns`): the three checks, one pass and two fails, as they
   are.

## Narration (from the speaker notes)

### Slide 1 · WildGuard  
*≈0.6 min*

Hello, we are Group 37, and this is WildGuard. WildGuard learns from what US wildlife inspectors actually did with millions of import declarations, and uses those patterns to rank the much larger, unlabelled global CITES trade database, so that reviewers know where to look first. We will cover the problem, the idea, how we aligned the data and chose every model, the results of one held-out test, what the system finds in CITES, a short demo, and its limits.

### Slide 2 · More trade than anyone can inspect  
*≈0.7 min*

Why is this a pattern-recognition problem? First, scale: the CITES Trade Database holds over 29 million records, and about 800 thousand new animal-trade records arrive every year, so inspections must be targeted by risk, as the CITES and World Customs Organization guidance asks. Second, the signal is rare: only 1.7 percent of US import line items ever drew enforcement action. Third, it is not in one field. Two live tortoises look ordinary until you compare them with their peers: the purpose, the price and the route make them stand out. Finding such combinations is what pattern recognition is for.

### Slide 3 · LEMIS has the outcomes; CITES has the scale  
*≈0.7 min*

Two public datasets fill each other’s gaps. LEMIS is the US Fish and Wildlife Service record of every wildlife import from 2000 to 2014, and for each line item it says what inspectors did: cleared it, or refused, seized, abandoned or re-exported it. Those are labels, for one country. The CITES Trade Database covers every Party but has no enforcement outcome. WildGuard learns the patterns that preceded enforcement in LEMIS and ranks CITES records with them. Six research questions guide the work: alignment, prediction, usefulness, combining evidence, transfer, and plausibility on CITES.

### Slide 4 · Two datasets, one shared schema  
*≈0.7 min*

First the data had to describe the same trade the same way: our knowledge-modelling step. Some codes look identical but are not: pre-Convention is source P in LEMIS and O in CITES; reintroduction is purpose Y in one and N in the other. Products are grouped by what the object is, for example one leather group for small goods, large goods and shoes. LEMIS has no CITES appendix, so for each shipment we look up the listing in force that year. Fields only LEMIS has, declared value, port and month, stay out; they are worth 0.035 PR-AUC, which is what alignment gives up.

### Slide 5 · Learn where outcomes exist, rank where they do not  
*≈0.8 min*

Here is the system in two lanes. In the LEMIS lane, where labels exist, we select animal line items, align them, describe each record against its peers, and train two models, XGBoost and a random forest. They are tested once, on 2013 and 2014, then refit on all fifteen years. Three things cross to the CITES lane: the crosswalk, the peer norms and the deployed models. Each CITES trade is scored once, records are grouped into export permits, and each year’s permits are ranked into a priority tier, the top one percent, and a review tier, the next four. The CITES lane is checked against outside evidence.

### Slide 6 · Nothing is tuned on the years it is judged on  
*≈0.7 min*

Only LEMIS has outcomes, so the evaluation lives there, and we controlled which years each step may see. Training uses 2000 to 2010, tuned by time-ordered cross-validation inside those years. Every design choice was made on 2011; 2012 was held back; 2013 to 2014 are the one test, scored once. The deciding metric is PR-AUC on CITES-listed species, the part of LEMIS most like CITES; with under two percent positives, accuracy would reward clearing everything. Effort is counted in shipments, what an inspector opens, with 95 percent intervals from a bootstrap of whole shipments, against random inspection and a “most endangered first” rule.

### Slide 7 · Every model choice made by comparison  
*≈0.8 min*

Every piece was chosen by comparison, with its decision rule written before its result. Six supervised candidates, from logistic regression to XGBoost, were compared in five configurations each. The random forest and XGBoost tied on 2011; our rule sent ties to the simpler model, but we broke this one on deployment cost, because the forest was far larger for the hosted app, and we report that openly. Forward selection then added a random forest to XGBoost and stopped: none of five unsupervised detectors helped. Features, transfer, training years and the review unit were decided the same way, and only then did we run the one test.

### Slide 8 · The one test: US imports 2013–2014, never seen  
*≈0.7 min*

Here is the one test, on 2013–2014 US imports of CITES-listed species, never seen before. WildGuard reaches a PR-AUC of 0.540 per line item, 27 times random, and 0.643 per shipment. The obvious manual rule, most endangered species first, scores 0.026: listing status alone is not the signal; combinations are. Every learned model beats both baselines widely. One result went against our design: XGBoost alone scored 0.006 higher than the ensemble, a small but real difference, after the ensemble had led on 2011. We kept the design and report it, because changing the model after the test would make the test meaningless.

### Slide 9 · What an inspection budget buys  
*≈0.6 min*

This curve is the usability argument. Along the bottom, the share of shipments inspected, highest score first; up the side, the share of shipments with an enforcement case found. With a five percent budget, WildGuard finds 63 percent of the cases, against five for random inspection and six for the listing rule. Finding 80 percent takes 21 percent of shipments instead of 80. And 99 of every 100 shipments in the top one percent really had a case. Commercial trade is harder; we return to that among the limits.

### Slide 10 · What helped, and what did not  
*≈0.8 min*

What helped? We tested five unsupervised detectors, including Isolation Forest, K-Means and DBSCAN. Each ranks cases above random, but none improved the supervised models, because much legal trade is simply unusual: the answer to RQ4 is no for unsupervised evidence. Combining two supervised models gained a little on 2011 and lost a little on the test. Class weights, the usual fix for imbalance, made the ranking worse. On transfer: trained without a world region, the ensemble still worked there, for all five regions. But it ages with the gap to the scored year, so the deployed model is refit on all fifteen years. And differences in law cannot be measured without labels outside the US.

### Slide 11 · From CITES records to a review queue  
*≈0.8 min*

Now CITES. Of 8.2 million animal records from 2015 to 2024, we count each trade once, and set aside records already marked as confiscated, since flagging what is already seized adds nothing. That leaves 5.5 million records on 3.3 million export permits. A reviewer opens a permit, not a line, so a permit takes its highest record’s score, and permits are ranked within each year, because recent reports arrive late. Below is a real 2024 permit, crocodile leather from France to China, in the priority tier. One lesson: anonymised permit IDs also occur on unrelated trades, so a permit is identified by its ID, exporter, importer and year.

### Slide 12 · Checked against outside evidence: one pass, two fails  
*≈0.9 min*

CITES has no outcomes, so we fixed three checks and their pass marks in advance, and report them as they came out. The UNODC check fails: groups over-represented in seizures are not clearly flagged more often. It had passed earlier, but only through three pangolin records that a permit-ID collision put in the priority tier; once we fixed the grouping, it failed. The confiscated-specimen check passes: without any source information, the model still ranks confiscated specimens far above the rest. The top-fifty check fails: thirty percent show no recognised pattern. And the biggest finding: half of the priority tier is Siamese crocodile leather, legal under CITES but refused by the US 95 percent of the time. WildGuard ranks trade by how US enforcement would treat it.

### Slide 13 · The review queue  
*≈1.3 min*

[Switch to the live app, Review queue.] The tool is for a CITES or customs reviewer with limited time. The queue opens on this year’s priority permits. Selecting one opens its flag card: first its tier and rank in the year, never a probability, because on CITES the score is a ranking. Then why it was flagged, in plain language, and how unusual it is for its peers. Here, sperm-whale carvings from Greenland to Denmark: both countries reported the permit, and their reports disagree, carvings against teeth. And a warning: the pattern may reflect US law, because the Marine Mammal Protection Act bans such imports and this trade does not involve the US. The reviewer decides; WildGuard says where to look first.

### Slide 14 · Check any record, see the evidence  
*≈1.1 min*

[Live app, Check a record.] A reviewer can also check one record. Saiga-antelope medicine from China to Japan ranks in the top 0.01 percent of 2019 permits, with its reasons: an unknown origin, a medicine product, an unusual source. A US declaration can add value, port and month. [Switch to Does it work?] For anyone deciding whether to trust the tool, this page shows the one test: drag the budget and read off how many real cases a team would find, against random and the listing rule. Every page puts the answer first and keeps the technical evidence one click below.

### Slide 15 · Limits, stated plainly  
*≈0.8 min*

Some limits, stated plainly. A flag is a prompt, not proof: our labels are what US inspectors did, and most stopped shipments never reach CITES, so a flag means a record resembles trade that was stopped in the US. US law travels with the labels, as Siamese crocodile and marine mammals show, and only marine mammals carry a warning. Commercial trade is much harder, and CITES is mostly commercial. The labels end in 2014, and the ranking ages. We make no accuracy claim on CITES; two of three checks fail. And permit signals can be shown to a reviewer but not learned. So every flag goes to a person.

### Slide 16 · What WildGuard shows  
*≈0.8 min*

To conclude. Enforcement knowledge partly transfers: using only fields both datasets share, inspecting five percent of held-out US shipments finds 63 percent of the cases, where the manual rule finds six. The enabling work was unglamorous: aligning two public datasets and choosing every model by a comparison whose rule was written first. On CITES the ranking shows its origin: it finds confiscated specimens it was never told about, but not the UNODC seizure pattern, and much of what it flags is trade US law prohibits. So WildGuard is a triage aid, not a verdict. Next would be a labelled sample from a CITES authority, warnings for every US prohibition, and a rebuild with each CITES release. Thank you.
