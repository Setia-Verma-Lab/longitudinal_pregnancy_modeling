# longitudinal_pregnancy_modeling

Source code for the manuscript: *Prototype Learning from Electronic Health Records for Interpretable Classification of Pregnancy Outcomes*.

Paths in notebooks/scripts use neutral placeholders (`catalog.omop`, `catalog.schema`, `/path/to/exports`); point these at your own catalog/volume before running.

## Structure

```
n3c_pregnancy_refactored/   # pregnancy episode phenotyping (HIPPS)
data_processing/            # cohort construction, MEDS conversion, tabular prep
modeling/                   # ProtoPregNet training, baselines, interpretability
  embedding_calculations/
  baselines/
  prototype_quality/
  appendix_analyses/
    landmarks/
```

### `n3c_pregnancy_refactored/`
Databricks refactorization of the [N3C HIPPS pregnancy-episode pipeline](https://github.com/jonessarae/n3c_pregnancy_cohort).

| File | Summary |
|---|---|
| `1_Map_external_concepts_to_standard_OMOP_concepts_databricks.py` | Maps external pregnancy concepts to standard OMOP concept IDs. |
| `2_Get_concept_prevalence_and_specificity_in_preg_women_databricks.py` | Estimates prevalence/specificity of pregnancy-related concepts. |
| `3_Define_pregnancy_episodes_with_HIP_algorithm_databricks.py` | Infers pregnancy episodes with the Hierarchy-based Inference of Pregnancy (HIP) algorithm. |
| `4_Identify_gestational_timing_specific_concepts_for_PPS_databricks.py` | Selects gestational-timing–specific concepts for PPS. |
| `5_Define_pregnancy_episodes_with_PPS_algorithm_databricks.py` | Infers episodes with the Pregnancy Progression Signature (PPS) algorithm. |
| `6_Merge_pregnancy_episodes_databricks.py` | Merges HIP and PPS episodes into a unified episode set. |
| `7_Infer_start_date_with_ESD_algorithm_and_add_metadata_databricks.py` | Estimates episode start dates (ESD) and attaches pregnancy metadata. |

### `data_processing/`

| File | Summary |
|---|---|
| `pregnancy_cohort_processing.ipynb` | Builds filtered pregnancy cohort and outcome labels used for modeling. |
| `omop_meds_etl_building.ipynb` | Converts OMOP exports into MEDS format for CLMBR embedding. |
| `tabular_data_prep.ipynb` | Builds the tabular sparse matrix and global feature set used by full-record and landmark baselines. |
| `table_one.ipynb` | Table 1 and covariate exploration. |

### `modeling/`

| File | Summary |
|---|---|
| `global_branch_hpo.ipynb` | Trains and tunes the global (whole-pregnancy) prototype branch. |
| `local_branch_hpo.ipynb` | Trains and tunes the local (trimester delta) prototype branch. |
| `local_branch_diagnostic.ipynb` | Compares local embedding variants (raw vs. delta) before final model training. |
| `fusion.ipynb` | Fuses frozen local + global prototype similarities into ProtoPregNet and evaluates held-out test performance. |

#### `modeling/embedding_calculations/`

| File | Summary |
|---|---|
| `clmbr_t_base_modeling.ipynb` | Fits / applies CLMBR-T to produce patient-level EHR embeddings. |
| `calculate_delta_embeddings.ipynb` | Builds delta embeddings (from rolling trimesters) for the local branch. |

#### `modeling/baselines/`

| File | Summary |
|---|---|
| `protopregnet_raw_embedding_baselines.ipynb` | Raw-embedding logistic regression / decision tree / random forest baselines on ProtoPregNet’s test set and exports predictions. |
| `protopregnet_non_interp_baseline.ipynb` | Non-interpretable adapter→linear baseline on raw embeddings (no prototype layer) and exports predictions. |
| `tabular_ehr_baselines_full_record.ipynb` | Full-record tabular OMOP baselines (LogReg / tree / RF) on the `tabular_data_prep` matrix. |
| `protopregnet_vs_baselines_significance.ipynb` | Paired bootstrap significance of ProtoPregNet versus all exported baselines. |

#### `modeling/prototype_quality/`

| File | Summary |
|---|---|
| `prototype_similarity_diagnostic.ipynb` | Diagnostics of prototype–patient similarities. |
| `prototype_visualization.ipynb` | Visualizations of learned prototypes for clinician interview. |

#### `modeling/appendix_analyses/`

| File | Summary |
|---|---|
| `subgroup_evaluations.ipynb` | Held-out subgroup evaluation (e.g. race, ethnicity, care site, outcome). |

#### `modeling/appendix_analyses/landmarks/`

| File | Summary |
|---|---|
| `01_landmark_prep.ipynb` | Builds the fixed person split, eligibility table, and shared caches for landmark analyses. |
| `02_landmark_train_eval.ipynb` | Trains and evaluates ProtoPregNet at one gestational-week cutoff. |
| `clmbr_landmark_inference.ipynb` | CLMBR-T embeddings at weekly gestational cutoffs. |
| `rf_landmark_eval.ipynb` | Time-dependent random forest evaluation at weekly landmarks. |
| `xgb_landmark_eval.ipynb` | Time-dependent XGBoost evaluation at weekly landmarks. |
| `rf_prototypes.ipynb` | RF + Breiman proximity prototypes (Optuna-best config). |
| `time_dependent_evaluation.ipynb` | Time-dependent metric comparisons across ProtoPregNet and tabular landmark models. |
