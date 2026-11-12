# Thesis Review Notes: "POWER OUTAGE THESIS - II (5).pdf"

Page numbers are the **printed page numbers** at the bottom of each page. Front-matter items have no
page numbers. Every Chapter 5 number was checked against `reports/` and the dataset. Numbers not
listed below were correct.

---

## A. Major issues (fix these first)

### A1. Step 4 opening paragraph still has the old split, with the numbers deleted (p. 41)
> "Of the records, records, or about , were used for training. Another records, or , were used for
> validation, while the remaining or , were used for testing."

This paragraph is the old 83/9/7 random split with the numbers removed. It also says Scikit-learn was
used to "prepare the inputs" and to divide the data, which is no longer true. Replace the whole
paragraph with:

> The prepared dataset was split by date into a development period (31 August 2022 to 31 December 2025;
> 10,971 records, 358 outages) and a held-out test period (1 January to 5 October 2026; 2,502 records,
> 60 outages). Within the development period, four rolling half-year validation folds (2024–2025) were
> used to tune the model, calibrate its probabilities and select the warning threshold. The test period
> was used only once, to evaluate the final model. The model was developed in Python using Pandas,
> Scikit-learn and XGBoost. XGBoost was used to estimate the chance of an outage on each recloser-day.
> The development followed nine steps, described below.

### A2. Weather is matched by date *and recloser*, not by date only
The dataset gives each recloser its own weather values. On 98–100% of dates, the nine reclosers have
different rainfall, gust and temperature values, and each recloser is tied to one barangay code
(`psgc`). The "vs p90" reference values also differ by recloser (see A6). The thesis says otherwise in
several places:

- Step 3 (p. 38): "Daily weather forecasts were matched by date". Change to "matched by date and by
  the location of each recloser's barangay".
- Section 4.6 (p. 36): "using the forecast data for the Puerto Princesa City location". Change to "for
  the barangay location mapped to each recloser".
- The Step 3 code snippet merges on `"date"` only, with `validate="many_to_one"`. It should merge on
  `["date", "psgc"]` (or the recloser's location key), and the lag/rolling code should run separately
  for each location (`groupby`). The snippet file I generated has the same mistake. I can fix it.

Confirm which location key your weather download script actually used before changing the text.

### A3. Section 3.3 says imbalance is handled with `scale_pos_weight = count(0)/count(1)` (p. 26)
The final model uses `scale_pos_weight = 1`, meaning no weighting. Section 4.7 Step 4.6 and Table 5.4
say weighting was tested and rejected. Rewrite the last paragraph of 3.3, for example:

> XGBoost can address class imbalance through the scale_pos_weight hyperparameter, which gives outage
> records a larger weight, typically count(y=0)/count(y=1). In this study, no weighting, square-root
> weighting and full-ratio weighting were compared during tuning (Section 4.7). Imbalance was otherwise
> handled through imbalance-aware metrics (PR-AUC, recall, F1), probability calibration and the choice
> of warning threshold.

Also update recommendation 10 (p. 81): "...for comparison with the class-weighting options tested in this study."

### A4. The Brier benchmark of "< 0.25" is too easy to defend (p. 51)
A score of 0.25 is what a model gets by always predicting 50%. With a 2.4% outage rate, even a model
that always predicts 2.4% scores about 0.023, far below 0.25. A panelist can say the benchmark is met by
any model. The justification sentence is also incorrect: an uninformative model's Brier score is
p(1 − p), not "the base rate".

Change the benchmark to "Brier score lower than the climatological baseline, i.e. Brier skill score
> 0". Your results already meet it (BSS 0.094 on the test set), and Table 5.7 already reports it.
Then update:
- the Table 5.5 "Benchmark" column for Brier ("< baseline 0.0234");
- the abstract;
- Chapter 6, conclusion 2.

### A5. Step 4 sub-steps are numbered 4.1–4.9, the same as the Chapter 4 section numbers (pp. 41–48)
"4.8 Choosing the Warning Threshold" sits two pages before "4.8 Data Analysis Plan", and "4.9 Training
the Final Model" comes before "4.9 Work Plan". Cross-references such as "(4.6)", "(4.7)" and "(4.8)"
are ambiguous. Renumber the sub-steps as **4.7.4.1–4.7.4.9**, or as **Step 4.1–Step 4.9**, or as
**(a)–(i)**, and update the in-text references.

### A6. Table 4.2 p90 reference values do not match the data (p. 31)
The table says rainfall p90 is 15.96 mm and gust p90 is 30.2 km/h. In the dataset, the reference is
different for each recloser:

| Recloser | Rain p90 (mm) | Gust p90 (km/h) |
|---|---|---|
| BUNCAG | 17.91 | 43.26 |
| CHARIOT | 18.54 | 43.20 |
| FACTOR | 18.48 | 42.43 |
| KALIKASAN | 18.16 | 43.04 |
| LACSAMANA | 19.63 | 42.04 |
| MENDOZA | 16.37 | 41.67 |
| PENRO | 21.66 | 42.60 |
| PILTEL | 18.10 | 43.20 |
| SAN MANUEL | 21.80 | 42.50 |

Suggested wording: "Forecast precipitation divided by the recloser's own 90th-percentile reference
(16.4–21.8 mm)". For gusts, use "41.7–43.3 km/h". Also state which period the
p90 was fitted on (the name "fit_p90" suggests the training period).

### A7. The work plan includes "Prototype Development" (p. 53)
Section 1.4 says the study does not build a deployable application, and recommendation 11 says a
prototype is future work. Rename that Gantt phase to something like "Model interpretation (SHAP) and
ablation analysis" in both the Gantt chart and the text.

### A8. Broken sentences in Chapter 5 and the Discussion
- **p. 61, after Table 5.6:** "Of the received no warning (99.56%)." Words are missing, and 99.56% does
  not match any result. Replace with: "Of the 2,442 recloser-days without an outage, 2,385 (97.7%)
  received no warning."
- **p. 76, Discussion, first bullet:** a sentence was pasted into the middle of another: "...held-out
  test set with a 2.40% Among the models evaluated ... difference. outage rate." Move "outage rate." back
  after "2.40%" and make "Among the models..." its own sentence.
- **p. 75:** "...changed test ROC-AUC by 0.001 in either direction or less, which neither changes the
  model's ranking performance.." Change to "...by no more than 0.001 in either direction, so neither
  group changed the model's ranking performance."

### A9. Missing and leftover references (pp. 82–85)
**Cited in the text but missing from the reference list:**
- Pedregosa et al. (2011)
- Hunter (2007)
- Lundberg et al. (2020)
- Open-Meteo (2026)
- Fu et al. (2022)
- Sivhugwana & Rangana (2025)
- Ke et al. (2017), if you add LightGBM to 4.6

**In the reference list but no longer cited, so delete:**
- Google Research (2024), Google Colab
- Waskom (2021), Seaborn
- the empty PAGASA (2026) entry
- Pressman & Maxim (2020), which I could not find cited anywhere

**Mismatches:**
- Python Software Foundation: the text says 2024 (Python 3.13), but the list says 2023 (Version 3.11).
- The Philippine Information Agency (2026) entry has no title details or URL.

### A10. Seaborn is still mentioned in Step 5 (p. 49)
"Matplotlib and Seaborn were used to visualize the results". Section 4.6 now says only Matplotlib. Remove
"and Seaborn". Step 5 should also mention:
- average precision;
- the reliability diagram and Brier skill score;
- the day-block bootstrap confidence intervals;
- the baseline comparison.

---

## B. Tables, figures and numbering

| # | Where | Issue | Fix |
|---|---|---|---|
| B1 | p. 43 | Fold table caption is still a placeholder: "Table n.n Name" | e.g. "Table 4.8 Rolling Validation Folds" (renumber the tables after it) |
| B2 | p. 50 | Evaluation Metrics table is labeled "Table 4.5", but Table 4.5 is Outage History | Renumber, e.g. Table 4.9, and update "Table 4.5 shows..." |
| B3 | p. 32 | Table 4.3 is titled "Lagged weather and outage-related features" but lists only weather | "Lagged and Cumulative Weather Features" |
| B4 | p. 33 | `network_outage_unit_days_prev7d` has no description | "Days in the previous 7 days with a reported outage anywhere in the recloser's wider network unit" |
| B5 | pp. 30–34 | The Chapter 4 tables list 63 of the 66 inputs. Missing: `forecast_gust_x_veg5m`, `forecast_precipitation_x_veg5m` and `recloser` | Add a small table: "Weather–vegetation interaction and identity features" |
| B6 | p. 31 | `forecast_weather_code` is described as "(categorical)" | The model uses it as a number, so write "Numeric WMO weather code" (Section 5.2 already says this) |
| B7 | p. 30 | Table 4.1 shows the day-of-year angle as 2π(day − 1)/365 | The dataset divides by the number of days in that year: 2π(day − 1)/N, with N = 365 or 366 |
| B8 | pp. 60–69 | Figure numbers go 5.1, 5.2, 5.4, 5.5, 5.6 (no 5.3), and the baseline figure is "Figure 5.B1" | Renumber in order. "5.B1" was the report's appendix label. |
| B9 | p. 61 | The confusion matrix is "Table 5.6" and the baseline tables are 5.6a–d, so readers may think they belong together | Optional: number the baseline tables 5.7–5.10, then shift Tables 5.7–5.10c by four |
| B10 | pp. 63–65 | Tables 5.6b and 5.6c are too narrow, so words wrap mid-word ("LightGB M", "Regressi on", "ROC-AU C", "0.02 12") | Landscape page, smaller font (9–10 pt), or shorten the headers |
| B11 | p. 70 | Table 5.8, rank 6 (`forecast_humidity_max_pct`) says "Non-monotonic" | The SHAP data show a clear negative trend (Spearman −0.83). Use "Higher maximum humidity lowers predicted risk" to match the other rows, or keep the wording if you checked the dependence plot. |
| B12 | pp. ii–iii | The table of contents and the lists of figures/tables are outdated: no Chapters V–VI, no Table 4.7+ or 5.x, no Figures 5.x, and wrong page numbers (e.g. References "43" vs actual 82). Some titles differ: "List of time-based features" vs "List of Temporal features", "Equipment Integrity Proxy Features" vs "Infrastructure and Repair History Features", "Ethical Consideration" vs "Ethical Considerations", "REVIEW OF RELATED LITERATURE" vs "...LITERATURES AND SYSTEMS" | Regenerate them with Word's automatic table of contents and caption lists |

---

## C. Content consistency

| # | Where | Issue | Fix |
|---|---|---|---|
| C1 | 1.4 (p. 5), 4.4 (p. 31), 4.5 (p. 35) | Period stated as "September 2022 to 2026" or "September 2022 to September 2026" | "August 2022 to October 2026" (31 Aug 2022 – 5 Oct 2026) |
| C2 | Appendix B (p. 88) | Note says the purchased outage data covers 2021–2025, but the study uses Aug 2022 – Oct 2026 | Explain that 2026 records came from PALECO public advisories, or correct the years |
| C3 | Step 2 (p. 37) | The sentence explaining why only 9 of 21 reclosers were kept was removed. Nowhere else explains "9 reclosers". | Add it back: "Only reclosers that appeared in every year from 2022 to 2026 were kept, reducing 21 recloser names to 9." |
| C4 | Table 2.3, Liaqat row (p. 17) | "engineered temporal features such as hour, month, season" | The model is daily, so it has no hour feature. Use "day of week, month and day of year". |
| C5 | Operational definitions (p. 8) | "Warning Threshold ... chosen to maximize F1 on the validation period" | Add "among thresholds with validation recall between 52% and 68%" |
| C6 | Operational definitions (p. 8) | "Temporal Lag Features" mentions only outage-record lags | Add lagged weather (previous 1, 3 and 7 days) |
| C7 | Objective 4 (p. 4) | "...contribution to the model's overall accuracy" | Accuracy is never reported (Section 3.3 says it is inappropriate). Use "overall predictive performance". |
| C8 | Data Analysis Plan (pp. 50–51) | F1 is called the "Primary classification performance metric", recall is "the priority classification metric", and Brier is "the primary metric for both predictive and calibration quality" | Choose one wording, e.g. recall = priority, F1 = summary threshold metric, ROC-AUC = ranking, Brier = probability quality |
| C9 | Data Analysis Plan (p. 51) | The ROC-AUC ≥ 0.75 benchmark is attributed to Sharma et al. (2023) and Lee et al. (2025), and F1 ≥ 0.60 to Putra et al. (2025) | Check that those papers actually state these values. If not, write "adopted by the researchers as a minimum benchmark". |
| C10 | Baseline paragraph (p. 52) | Logistic Regression description is missing "standardized inputs" | Add it, so it matches Section 5.4.1 |
| C11 | 4.6 Libraries (p. 36) | LightGBM is not listed, but it is a baseline | Add a bullet: LightGBM (4.6) (Ke et al., 2017) |
| C12 | Recommendation 11 (p. 81) | "displays outage probability by barangay" | The model predicts per recloser: "by recloser and the barangays it serves" |
| C13 | 4.2 Research Design (p. 28) | "anticipating outages one day ahead" | Fine, since Chapter 5 states the Historical Forecast API limitation. Add the missing period at the end of the sentence. |
| C14 | 3.4 SHAP (p. 25), Section 4.5 (p. 35) | Future tense ("SHAP will be used", "instruments to be used") | Past tense |

---

## D. Front matter and names

| # | Where | Issue |
|---|---|---|
| D1 | Title page vs approval sheet | "Ombion, Kert Ace" vs "Kert-Ace Ombion". Use the same spelling everywhere. |
| D2 | Title page | "Department OF Computer Science" should be "Department of Computer Science" |
| D3 | Certification of Originality and Responsible AI Use | Still template placeholders. If AI tools were used for code, analysis or editing, list each tool, the sections it affected and how you verified the output. |
| D4 | Ethics, Privacy and Data Compliance | All rows are placeholders |
| D5 | Acknowledgments and approval sheet | Placeholders |
| D6 | Appendices C and D | Both are titled "Certification of Readiness for Proposal Defense". D is probably for the final defense. |

---

## E. Small wording and typo fixes

| Page | Current | Fix |
|---|---|---|
| p. 1 | "th e loss of power" | "the loss of power" |
| p. 1 | "(Franciso, 2022)" | "(Francisco, 2022)" |
| p. 1 | "resulting to frequent power outages" | "resulting in" |
| p. 2 | "outages does not only occur ... may be influence by" | "outages do not only occur ... may be influenced by" |
| p. 3 | "develop a XGBoost-based" | "an XGBoost-based" |
| p. 3 | "Brier Score ?" | "Brier score?" (no space) |
| p. 5 | "containing informations" | "information" |
| p. 9 | "Logistics Regression" | "Logistic Regression" |
| p. 11 | "the reviewed literatures examined" | "the reviewed literature" |
| p. 13 | "(CNN and RNN))" | one closing parenthesis |
| p. 15 | "interruptions outage in selected reclosers in Puerto Princesa City ,rather" | "interruptions in selected reclosers in Puerto Princesa City, rather" |
| p. 17 | "not available in small distribution systems.." | one period |
| p. 19 | Two sentences say the same thing ("On the other hand, boosting builds trees sequentially..." / "Conversely, Boosting algorithms...") | keep one |
| p. 23 | "estimates a 85% likelihood" | "an 85% likelihood" |
| p. 24 | "The model sequentially builds decision trees..." appears twice | delete the repeat |
| p. 25 | SHAP formula has an extra "=": "Σ = ϕi" | f(x) = ϕ0 + Σ ϕi |
| p. 33 | "summarized for and summarized per recloser through its mapped barangays.." | "summarized per recloser through its mapped barangays." |
| p. 33 | "These includes rolling counts" | "These include" |
| p. 35 | "Specifically, PALECO data request and service advisories." (fragment) | "**PALECO data request and service advisories.** A formal data request..." as a run-in heading |
| p. 37 | Duplicate sentences "The weather variables include..." / "The weather variables includes..." | keep one |
| p. 37 | "the datasets was loaded" | "the datasets were loaded" |
| p. 37 | "utility equipments" | "utility equipment" |
| p. 50 | F1 formula is labeled "F2" and is missing the factor 2 | F1 = 2 × (Precision × Recall) / (Precision + Recall) |
| p. 51 | Brier formula "BS = 1/n = Σ" | BS = (1/n) Σ (p̂ₜ − yₜ)² |
| p. 51 | "...(Putra et al., 2025)" | sentence needs a final period |
| p. 54 | Chapter title "RESULT AND DISCUSSION" | "RESULTS AND DISCUSSION" |
| p. 54 | "...an outage prevalence of 3.10%" | add a final period |
| p. 65 | First bullet under Table 5.6c ("XGBoost issued about 2 alerts...") lost its bullet | restore the bullet |
| p. 73 | "Figure 5.6 and Table 5.9 shows" | "show" |
| p. 59 | "Table 4.5 shows the statistical methods ... that was used" | "that were used" |

---

## G. Metrics and methods: are they introduced before they are used?

Rule of thumb: anything reported in Chapter 5 should first appear in the objectives or scope, then be
defined in Chapter 1.6 or the Chapter 4 Data Analysis Plan, then be described in the Chapter 4
procedure.

✓ = present, — = missing, (✓) = only mentioned in passing

| Item | SOP Q2 / Obj. 2 (pp. 3–4) | Scope 1.4 (p. 4) | Definitions 1.6 (pp. 7–8) | Conceptual framework output (p. 28) | Step 5 (p. 49) | Data Analysis Plan (pp. 49–52) | Used in Ch. 5 |
|---|---|---|---|---|---|---|---|
| Precision, recall, F1 | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| AUC-ROC | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Brier score | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Reliability diagram / calibration | ✓ | ✓ | ✓ | — | ✓ | ✓ | ✓ |
| **Average precision (PR-AUC)** | — | — | ✓ | — | — | — | ✓ (Tables 5.5, 5.6b, 5.10b; abstract) |
| **Specificity** | — | — | — | — | — | — | ✓ (Tables 5.5, 5.6, 5.6c) |
| **Brier skill score** | — | — | — | — | — | — | ✓ (Tables 5.6b, 5.7; Ch. 6) |
| **Bootstrap 95% CIs** | — | — | — | — | — | — | ✓ (described for the first time on p. 59) |
| **Ablation (feature-set removal)** | (✓) SOP Q4 / Obj. 4 | — | — | — | — | — | ✓ (Section 5.7) |
| Baseline comparison | — | — | — | — | — | ✓ | ✓ |
| Warning threshold | — | — | ✓ | ✓ | — | — | ✓ |
| Platt calibration | — | — | — | ✓ | — | — | ✓ |
| Confusion matrix (TP/FP/FN/TN) | — | — | — | — | — | (✓) formulas only | ✓ |
| Lift over random | — | — | — | — | — | — | ✓ (Table 5.6b only, defined in its note) |
| Top-k% risk capture | — | — | — | — | — | — | ✓ (Table 5.6d, abstract, Ch. 6) |

### What to add (the minimum set)

1. **Average precision (PR-AUC)** is the metric that most needs adding. It was used to tune the model
   (Step 4.6) and is reported throughout Chapter 5, but it is missing from the research question,
   objective, scope, Step 5 and Data Analysis Plan.
   - **SOP Q2 (p. 3) and Objective 2 (p. 4):** "...AUC-ROC, average precision (PR-AUC), F1-score,
     precision, recall, calibration analysis, and Brier Score".
   - **Scope 1.4 (p. 4):** add "average precision" to the metric list.
   - **Data Analysis Plan table (p. 50):** add a row: *Average Precision (PR-AUC): area under the
     precision–recall curve; summarizes how well outage days are ranked above non-outage days when
     outages are rare; used for hyperparameter selection and compared with the outage rate
     (random = prevalence).*
   - **Step 5 (p. 49):** add it to the metric list.
   - **Choose one name.** Chapter 4 says "PR-AUC" (pp. 36, 45–46) and Chapter 5 says "average
     precision" (most places), while Table 5.10b and 5.10c headings say "PR-AUC". Define it once as
     "average precision (PR-AUC)" and use one form afterwards.
2. **Specificity:** add to the Data Analysis Plan as TN / (TN + FP), "share of non-outage recloser-days
   that correctly received no warning", and to the 1.6 definitions.
3. **Brier skill score:** add BSS = 1 − Brier / Brier_baseline, where the baseline always predicts the
   observed outage rate, to the Data Analysis Plan, next to the Brier score. This also fixes the "< 0.25"
   benchmark problem (A4).
4. **Bootstrap confidence intervals:** add one paragraph to the Data Analysis Plan: "95% confidence
   intervals for test metrics were estimated with a day-block bootstrap (1,000 resamples of whole
   days), because outages on the same day are correlated across reclosers. Differences between models
   or feature sets whose interval excludes zero were treated as clear."
5. **Ablation procedure (Objective 4).** Chapter 4 never explains how Objective 4 is answered; the word
   "ablation" first appears in Chapter 5. Add a paragraph to the Data Analysis Plan:
   > Feature contribution analysis (ablation). To assess the contribution of engineered features, the
   > model was retrained with selected feature groups removed: (a) lagged meteorological features
   > (15 inputs), (b) cumulative outage-history features (8 inputs), (c) both, plus supplementary sets
   > without vegetation, without infrastructure and repair history, weather only, and non-weather only.
   > Each set used the same hyperparameters, chronological splits and calibration method, and its own
   > threshold was selected with the same rule. Changes in test ROC-AUC and average precision were
   > reported with day-block bootstrap 95% confidence intervals.

   Say in the text that "lagged meteorological features" means the 12 lagged forecast-weather inputs
   plus the 3 wet-weather history inputs (66 − 15 = 51 in Table 5.10). Otherwise a reader comparing
   with Table 5.3 (12 lagged) will not see where 15 comes from.
6. **Confusion matrix:** one sentence in the Data Analysis Plan defining TP, FP, FN and TN for a
   recloser-day.
7. **Conceptual framework output (p. 28):** add average precision, the reliability diagram, ablation
   results and the baseline comparison.
8. **Step 5 (p. 49):** rewrite it as a summary of the Data Analysis Plan:
   - all metrics above;
   - bootstrap CIs;
   - SHAP on the 2,502 test recloser-days, averaged over the 3 seed models, with group totals;
   - ablation;
   - baselines;
   - Matplotlib only.

### Other cross-chapter mismatches

| Item | Where | Mismatch | Fix |
|---|---|---|---|
| Metric name | throughout | Both "AUC-ROC" and "ROC-AUC" are used, sometimes in the same paragraph (e.g. pp. 74–75) | Choose one; the objectives use AUC-ROC |
| Objective 4 wording | SOP Q4 (p. 3), Obj. 4 (p. 4), Section 5.7 title (p. 74) | "lagged features and cumulative outage frequency" vs "lagged meteorological data and rolling outage frequency" vs "Lagged and Cumulative Outage Features" (the title suggests the outage features are the lagged ones) | Use one phrase everywhere, e.g. "lagged meteorological features and cumulative outage-history features" |
| Objective 4 measure | Obj. 4 (p. 4) | "...contribution to the model's overall accuracy" | "overall predictive performance" (accuracy is never used) |
| Feature groups in SOP Q3 | p. 3 | Lists "meteorological, temporal, infrastructure, and vegetation-related", leaving out **outage history**, which SHAP and Chapter 5 report | Add "outage-history" |
| Feature groups in 1.4 / 4.1 / 4.2 | pp. 4, 27–28 | Some list only weather, history and temporal; the 4.2 design sentence omits infrastructure and temporal | Use the same five data sources everywhere: outage history, weather, vegetation, infrastructure/repair history, calendar |
| Threshold rule | 1.6 (p. 8) vs Step 4.8 (p. 48) | "maximize F1" vs "max F1 among recall 52–68%" | Use the full rule in both |
| Recall vs F1 priority | Data Analysis Plan (pp. 50–51) | Recall is "the priority metric", F1 is "the primary classification metric", Brier is "the primary metric" | See C8 |
| Calibration theory | Chapter 3 | Chapter 3 covers XGBoost, SHAP and class imbalance, but not probability calibration, Platt scaling or threshold selection, which Chapters 4–5 rely on | Optional short Section 3.4 "Probability Calibration and Decision Thresholds" |
| Ensemble of 3 seeds | Ch. 3, 4.1 vs Ch. 4.7 / 5 / abstract | Chapter 5 and the abstract say "an ensemble of three XGBoost models"; the framework and Chapter 3 describe a single model | Mention the 3-seed averaging once in 4.1 or 4.2 |
| Title | Title page | Lists "Historical, Meteorological, Vegetation, and Infrastructure Reliability Data" but not calendar/temporal features | Fine as is; mention only if the panel asks |

---

## F. Checked and correct

These match the saved results exactly:
- abstract numbers;
- Tables 5.1, 5.2, 5.2b (all fold counts add up);
- Tables 5.4, 5.5, 5.6, 5.6a–d, 5.7, 5.8, 5.9, 5.10, 5.10b, 5.10c;
- the reliability-bin figures (92% in the lowest bin; 17.2% vs 50.0%; 66.5% vs 35.7%);
- SHAP shares (21.6%, 75.4%, 8.6%);
- feature-level values (7.2 mm / 5.7% / +0.21, and so on);
- routine vs weather-stress results;
- the top-5% risk ranking (30 of 60);
- Chapter 6 numbers.

The Step 4.8 threshold search range (0.01–0.60) and the 52–68% recall band also match `train.py`.
