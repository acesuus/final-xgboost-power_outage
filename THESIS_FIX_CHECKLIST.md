# Thesis Fix Checklist

Tick an item by changing `[ ]` to `[x]`. IDs match `THESIS_REVIEW_NOTES.md`, which has the details and
suggested wording. Pages are the printed page numbers in "POWER OUTAGE THESIS - II (5).pdf".

**Progress:** 0 / 115 done

---

## A. Major issues

- [ ] **A1** p. 41: Replace the broken Step 4 opening paragraph ("Of the records, records, or about ,") with the time-split paragraph
- [ ] **A2a** p. 38: Step 3 says weather was "matched by date"; change to date + recloser location
- [ ] **A2b** p. 36: Section 4.6 says "Puerto Princesa City location"; change to each recloser's barangay location
- [ ] **A2c** Step 3 code: merge weather on date + location, and compute lags per location
- [ ] **A2d** Confirm which location key the weather download script used
- [ ] **A3a** p. 26: Rewrite Section 3.3: weighting was tested, and the final model uses no weighting
- [ ] **A3b** p. 81: Recommendation 10: mention the weighting options tested
- [ ] **A4a** p. 51: Brier benchmark: replace "< 0.25" with "beats the base-rate baseline (BSS > 0)"
- [ ] **A4b** p. 51: Fix the justification sentence (uninformative Brier = p(1 − p))
- [ ] **A4c** p. 59: Table 5.5: update the Brier benchmark column
- [ ] **A4d** Abstract and Chapter 6, conclusion 2: update the Brier benchmark wording
- [ ] **A5a** pp. 41–48: Renumber Step 4 sub-steps 4.1–4.9 so they don't clash with the section numbers
- [ ] **A5b** pp. 41–48: Update in-text references "(4.6)", "(4.7)", "(4.8)"
- [ ] **A6a** p. 31: Table 4.2: replace the p90 values (15.96 mm / 30.2 km/h) with the per-recloser ranges (16.4–21.8 mm / 41.7–43.3 km/h)
- [ ] **A6b** p. 31: State which period the p90 reference was fitted on
- [ ] **A7** p. 53 and Gantt chart: Rename the "Prototype Development" phase
- [ ] **A8a** p. 61: Fix "Of the received no warning (99.56%)" → 2,385 of 2,442 (97.7%)
- [ ] **A8b** p. 76: Discussion, first bullet: move "outage rate." back after "2.40%"
- [ ] **A8c** p. 75: Fix the "which neither changes ... performance.." sentence
- [ ] **A9a** References: add Pedregosa et al. (2011)
- [ ] **A9b** References: add Hunter (2007)
- [ ] **A9c** References: add Lundberg et al. (2020)
- [ ] **A9d** References: add Open-Meteo (2026)
- [ ] **A9e** References: add Fu et al. (2022)
- [ ] **A9f** References: add Sivhugwana & Rangana (2025)
- [ ] **A9g** References: add Ke et al. (2017), if LightGBM is listed in 4.6
- [ ] **A9h** References: delete Google Research (2024), Colab
- [ ] **A9i** References: delete Waskom (2021), Seaborn
- [ ] **A9j** References: delete the empty PAGASA (2026) entry
- [ ] **A9k** References: delete Pressman & Maxim (2020), or cite it
- [ ] **A9l** References: Python Software Foundation 2023 / v3.11 → 2024 / 3.13
- [ ] **A9m** References: complete the Philippine Information Agency (2026) entry
- [ ] **A10** p. 49: Step 5: remove "Seaborn"

## B. Tables, figures and numbering

- [ ] **B1** p. 43: Replace the "Table n.n Name" caption on the rolling-folds table
- [ ] **B2** p. 50: Renumber the Evaluation Metrics table (it is labeled 4.5, which is already used)
- [ ] **B3** p. 32: Rename Table 4.3 to "Lagged and Cumulative Weather Features"
- [ ] **B4** p. 33: Table 4.5: add a description for `network_outage_unit_days_prev7d`
- [ ] **B5** pp. 30–34: Add a table for `forecast_gust_x_veg5m`, `forecast_precipitation_x_veg5m` and `recloser`
- [ ] **B6** p. 31: `forecast_weather_code`: "(categorical)" → "numeric WMO weather code"
- [ ] **B7** p. 30: Table 4.1: day-of-year angle 2π(day − 1)/365 → 2π(day − 1)/N (N = 365 or 366)
- [ ] **B8** pp. 60–69: Renumber the figures (5.3 is missing; "Figure 5.B1")
- [ ] **B9** p. 61: (Optional) Renumber the baseline tables 5.6a–d so they don't share "5.6" with the confusion matrix
- [ ] **B10** pp. 63–65: Widen Tables 5.6b and 5.6c (landscape or smaller font) so words don't wrap
- [ ] **B11** p. 70: Table 5.8, rank 6: direction wording for `forecast_humidity_max_pct`
- [ ] **B12** pp. ii–iii: Regenerate the table of contents and the lists of figures and tables (Chapters V–VI, new tables and figures, page numbers, titles)

## C. Content consistency

- [ ] **C1** pp. 4, 31, 35: Study period: "September 2022 to (September) 2026" → "August 2022 to October 2026"
- [ ] **C2** p. 88: Appendix B: purchased data years (2021–2025) vs study period
- [ ] **C3** p. 37: Step 2: restore the sentence on why 21 reclosers were reduced to 9
- [ ] **C4** p. 17: Table 2.3, Liaqat row: remove "hour" from the temporal features
- [ ] **C5** p. 8: Definition of Warning Threshold: add the 52–68% recall band
- [ ] **C6** p. 8: Definition of Temporal Lag Features: add lagged weather
- [ ] **C7** p. 4: Objective 4: "overall accuracy" → "overall predictive performance"
- [ ] **C8** pp. 50–51: Use one wording for the priority/primary metric (recall vs F1 vs Brier)
- [ ] **C9** p. 51: Check that the cited papers support AUC ≥ 0.75 and F1 ≥ 0.60
- [ ] **C10** p. 52: Baseline paragraph: add "standardized inputs" for Logistic Regression
- [ ] **C11** p. 36: Section 4.6: add LightGBM to the libraries
- [ ] **C12** p. 81: Recommendation 11: "by barangay" → "by recloser"
- [ ] **C13** p. 28: Section 4.2: add the missing period after "one day ahead"
- [ ] **C14** pp. 25, 35: Change future tense to past tense ("will be used", "to be used")

## D. Front matter and names

- [ ] **D1** Use the same name spelling everywhere: "Kert Ace" vs "Kert-Ace"
- [ ] **D2** Title page: "Department OF" → "Department of"
- [ ] **D3** Fill in the AI-use certification table
- [ ] **D4** Fill in the Ethics, Privacy and Data Compliance statement
- [ ] **D5** Fill in the acknowledgments and approval sheet names
- [ ] **D6** Appendix D: fix its title (it duplicates Appendix C)

## E. Typos and formulas

- [ ] **E1** p. 1: "th e loss" → "the loss"
- [ ] **E2** p. 1: "Franciso" → "Francisco"
- [ ] **E3** p. 1: "resulting to" → "resulting in"
- [ ] **E4** p. 2: "outages does ... may be influence" → "do ... influenced"
- [ ] **E5** p. 3: "a XGBoost-based" → "an XGBoost-based"
- [ ] **E6** p. 3: "Brier Score ?" → "Brier score?"
- [ ] **E7** p. 5: "informations" → "information"
- [ ] **E8** p. 9: "Logistics Regression" → "Logistic Regression"
- [ ] **E9** p. 11: "reviewed literatures" → "reviewed literature"
- [ ] **E10** p. 13: "(CNN and RNN))" → one closing parenthesis
- [ ] **E11** p. 15: "interruptions outage ... City ,rather" → "interruptions ... City, rather"
- [ ] **E12** p. 17: "systems.." → one period
- [ ] **E13** p. 19: Remove the duplicated boosting sentence
- [ ] **E14** p. 23: "a 85%" → "an 85%"
- [ ] **E15** p. 24: Remove the duplicated "The model sequentially builds..." sentence
- [ ] **E16** p. 25: SHAP formula: remove the stray "="
- [ ] **E17** p. 33: "summarized for and summarized per ... barangays.." → fix
- [ ] **E18** p. 33: "These includes" → "These include"
- [ ] **E19** p. 35: Make "PALECO data request and service advisories." a run-in heading
- [ ] **E20** p. 37: Remove the duplicated "weather variables include(s)" sentence
- [ ] **E21** p. 37: "the datasets was" → "were"
- [ ] **E22** p. 37: "utility equipments" → "utility equipment"
- [ ] **E23** p. 50: F1 formula: label "F2" → "F1", and add "2 ×"
- [ ] **E24** p. 51: Brier formula: remove the stray "="
- [ ] **E25** p. 51: Add the missing final period after "(Putra et al., 2025)"
- [ ] **E26** p. 54: "RESULT AND DISCUSSION" → "RESULTS AND DISCUSSION"
- [ ] **E27** p. 54: Add the missing period after "3.10%"
- [ ] **E28** p. 65: Restore the bullet on the first XGBoost line under Table 5.6c
- [ ] **E29** p. 73: "Figure 5.6 and Table 5.9 shows" → "show"
- [ ] **E30** p. 59: "metrics that was used" → "were used"

## G. Metrics and methods introduced before use

- [ ] **G1a** p. 3: Research question 2: add average precision (PR-AUC)
- [ ] **G1b** p. 4: Objective 2: add average precision (PR-AUC)
- [ ] **G1c** p. 4: Scope 1.4: add average precision
- [ ] **G1d** p. 50: Data Analysis Plan: add an Average Precision row
- [ ] **G1e** p. 49: Step 5: add average precision
- [ ] **G1f** Throughout: use one name, "average precision (PR-AUC)", consistently
- [ ] **G2** Data Analysis Plan and 1.6: define specificity, TN / (TN + FP)
- [ ] **G3** Data Analysis Plan: define the Brier skill score
- [ ] **G4** Data Analysis Plan: add a paragraph on the day-block bootstrap 95% CIs
- [ ] **G5a** Data Analysis Plan: add a paragraph on the ablation procedure (Objective 4)
- [ ] **G5b** Explain that "lagged meteorological" = 12 lagged weather + 3 wet-weather history (15 inputs)
- [ ] **G6** Data Analysis Plan: one sentence defining TP, FP, FN, TN
- [ ] **G7** p. 28: Conceptual framework output: add AP, reliability diagram, ablation, baselines
- [ ] **G8** p. 49: Rewrite Step 5 to summarize all evaluation methods
- [ ] **G9** Throughout: use "AUC-ROC" or "ROC-AUC", not both
- [ ] **G10** pp. 3, 4, 74: Use the same wording for Objective 4 in RQ4, Obj. 4 and the 5.7 title
- [ ] **G11** p. 3: Research question 3: add outage-history features
- [ ] **G12** pp. 4, 27–28: List the same five data sources in 1.4, 4.1 and 4.2
- [ ] **G13** Mention the 3-seed ensemble in 4.1 or 4.2
- [ ] **G14** (Optional) Add Section 3.4, Probability Calibration and Decision Thresholds
