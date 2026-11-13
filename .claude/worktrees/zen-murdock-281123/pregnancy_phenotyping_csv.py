"""
Pregnancy Phenotyping Algorithm (HIPPS) for ETL'd DE-SynPuf OMOP data
================================================================

This script refactors the original N3C pregnancy cohort phenotyping
pipeline (7 Spark-based scripts) into a single, non-Spark Python module
that operates on OMOP Common Data Model tables stored as CSV files.

It is adapted to your current project layout:

- OMOP tables are read from `synpuf_etl_output/` (e.g. `person.csv`,
  `condition_occurrence.csv`, etc.).
- Concept definition files are read from `n3c_pregnancy_cohort/`
  (e.g. `HIP_concepts.xlsx`, `PPS_concepts.xlsx`, etc.).
- Intermediate and final outputs are written to `data/`.

The implementation preserves the overall functional structure of the
original 7-step HIPPS pipeline while using pandas instead of PySpark:

  1. Map external pregnancy concepts (Matcho) to standard OMOP concepts.
  2. Identify pregnancy-specific concepts and use N3C-derived
     "highly specific" concepts.
  3. Define outcome- and gestation-based pregnancy episodes (HIP).
  4. Identify gestational timing specific concepts (PPS) and load
     PPS concept definitions.
  5. Define PPS episodes based on temporal progression signatures.
  6. Merge HIP and PPS episodes, resolving overlaps.
  7. Infer pregnancy start date (ESD algorithm, simplified) and add
     demographic and metadata fields.

Usage
-----
    python pregnancy_phenotyping_csv.py

Outputs
-------
Intermediate CSVs and the final pregnancy cohort are saved in `data/`:

- step1_matcho_concepts_standardized.csv
- step2_highly_specific_concepts.csv
- step3_hip_episodes.csv
- step4_pps_concepts.csv
- step5_pps_episodes.csv
- step6_merged_episodes.csv
- step7_pregnancy_cohort_final.csv
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parent

# Clinical OMOP tables (person, condition_occurrence, etc.)
CLINICAL_OMOP_DIR = PROJECT_ROOT / "synpuf_etl_output"
# Vocabulary tables (CONCEPT.csv, CONCEPT_RELATIONSHIP.csv, etc.)
VOCAB_DIR = PROJECT_ROOT / "vocab_download_v5"

DATA_DIR = PROJECT_ROOT / "data"
CONCEPT_DIR = PROJECT_ROOT / "n3c_pregnancy_cohort"


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _read_omop(table: str) -> pd.DataFrame:
    """
    Read an OMOP table from the appropriate folder.

    - Vocabulary tables (concept, concept_relationship) are read from
      `vocab_download_v5/` where you downloaded the vocabularies
      (e.g. `CONCEPT.csv`, `CONCEPT_RELATIONSHIP.csv`).
    - Clinical tables (person, condition_occurrence, etc.) are read
      from `synpuf_etl_output/` (e.g. `person.csv`,
      `condition_occurrence.csv`, etc.).

    For clinical tables, if a single merged `{table}.csv` exists it is
    used; otherwise all `{table}_*.csv` partitions are concatenated.
    """
    table_lower = table.lower()

    # Vocabulary tables live under VOCAB_DIR with upper-case filenames
    if table_lower == "concept":
        vocab_path = VOCAB_DIR / "CONCEPT.csv"
        if not vocab_path.exists():
            raise FileNotFoundError(f"Expected vocabulary file not found: {vocab_path}")
        # OMOP vocab files are typically tab-delimited
        return pd.read_csv(
            vocab_path,
            sep="\t",
            dtype={"concept_id": "int64"},
            encoding="utf-8",
            on_bad_lines="warn",
            low_memory=False,
        )

    if table_lower == "concept_relationship":
        vocab_path = VOCAB_DIR / "CONCEPT_RELATIONSHIP.csv"
        if not vocab_path.exists():
            raise FileNotFoundError(f"Expected vocabulary file not found: {vocab_path}")
        return pd.read_csv(
            vocab_path,
            sep="\t",
            dtype={"concept_id_1": "int64", "concept_id_2": "int64"},
            encoding="utf-8",
            on_bad_lines="warn",
            low_memory=False,
        )

    # All other tables are assumed to be clinical OMOP tables
    merged = CLINICAL_OMOP_DIR / f"{table}.csv"
    if merged.exists():
        return pd.read_csv(merged)

    parts = sorted(CLINICAL_OMOP_DIR.glob(f"{table}_*.csv"))
    if not parts:
        raise FileNotFoundError(
            f"No CSV files found for OMOP table '{table}' in {CLINICAL_OMOP_DIR}"
        )

    dfs = [pd.read_csv(p) for p in parts]
    return pd.concat(dfs, ignore_index=True)


def _parse_date_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(pd.NaT, index=df.index)
    return pd.to_datetime(df[col], errors="coerce")


@dataclass
class PregnancyPhenotypingCSV:
    """
    Pregnancy phenotyping pipeline using local OMOP CSV tables.
    """

    def __post_init__(self) -> None:
        _ensure_dirs()

        print("Loading concept definition files from n3c_pregnancy_cohort/ ...")
        self.hip_concepts = pd.read_excel(CONCEPT_DIR / "HIP_concepts.xlsx")
        self.matcho_concepts = pd.read_excel(CONCEPT_DIR / "Matcho_concepts.xlsx")
        self.pps_concepts = pd.read_excel(CONCEPT_DIR / "PPS_concepts.xlsx")
        self.matcho_outcome_limits = pd.read_excel(CONCEPT_DIR / "Matcho_outcome_limits.xlsx")
        self.matcho_term_durations = pd.read_excel(CONCEPT_DIR / "Matcho_term_durations.xlsx")

        try:
            self.highly_specific_concepts = pd.read_excel(CONCEPT_DIR / "Highly_specific_concepts.xlsx")
            print("  - Loaded Highly_specific_concepts.xlsx")
        except FileNotFoundError:
            self.highly_specific_concepts = None
            print("  - Highly_specific_concepts.xlsx not found; will use PPS concepts / defaults.")

        print(f"  - HIP concepts: {len(self.hip_concepts)}")
        print(f"  - Matcho concepts: {len(self.matcho_concepts)}")
        print(f"  - PPS concepts: {len(self.pps_concepts)}")

        # Placeholders populated by pipeline steps
        self.matcho_concepts_std: Optional[pd.DataFrame] = None
        self.hip_episodes: Optional[pd.DataFrame] = None
        self.pps_episodes: Optional[pd.DataFrame] = None
        self.merged_episodes: Optional[pd.DataFrame] = None
        self.final_cohort: Optional[pd.DataFrame] = None

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------
    @staticmethod
    def _save_csv(df: pd.DataFrame, filename: str, step: Optional[str] = None) -> None:
        path = DATA_DIR / filename
        df.to_csv(path, index=False)
        msg = f"Saved: {filename} ({len(df)} rows)"
        if step:
            msg = f"[Step {step}] {msg}"
        print(msg)

    # ---------------------------------------------------------------------
    # STEP 1: Map external concepts to standard OMOP concepts
    # ---------------------------------------------------------------------
    def step1_map_concepts(self) -> pd.DataFrame:
        """
        Map Matcho pregnancy concepts to standard OMOP concepts using the
        local OMOP `concept.csv` and `concept_relationship.csv`.
        """
        print("\n" + "=" * 60)
        print("STEP 1: Map external concepts to standard OMOP concepts")
        print("=" * 60)
        
        # Check if output already exists
        output_file = DATA_DIR / "step1_matcho_concepts_standardized.csv"
        if output_file.exists():
            print(f"Loading cached result from {output_file.name}...")
            matcho_std = pd.read_csv(output_file)
            print(f"Loaded {len(matcho_std)} standardized concepts")
            self.matcho_concepts_std = matcho_std
            return matcho_std

        concept_df = _read_omop("concept")
        concept_rel = _read_omop("concept_relationship")

        concept_ids = self.matcho_concepts["concept_id"].tolist()
        concept_sub = concept_df[concept_df["concept_id"].isin(concept_ids)].copy()

        # Attach Matcho categories
        concept_sub = concept_sub.merge(
            self.matcho_concepts[["concept_id", "category"]],
            on="concept_id",
            how="inner",
        )

        print(f"Found {len(concept_sub)} Matcho concepts in local OMOP concept table")

        # Standard vs non-standard concepts
        standard = concept_sub[concept_sub["standard_concept"] == "S"].copy()
        non_standard = concept_sub[concept_sub["standard_concept"].isna()].copy()

        print(f"  - Standard concepts: {len(standard)}")
        print(f"  - Non-standard concepts: {len(non_standard)}")

        mapped = pd.DataFrame()
        if not non_standard.empty:
            # Map non-standard -> standard via "Maps to"
            rel_sub = concept_rel[
                (concept_rel["concept_id_1"].isin(non_standard["concept_id"]))
                & (concept_rel["relationship_id"] == "Maps to")
            ][["concept_id_1", "concept_id_2"]].rename(
                columns={"concept_id_1": "source_concept_id", "concept_id_2": "concept_id"}
            )

            if not rel_sub.empty:
                mapped = rel_sub.merge(
                    concept_df,
                    on="concept_id",
                    how="inner",
                )
                mapped = mapped.merge(
                    non_standard[["concept_id", "category"]].rename(
                        columns={"concept_id": "source_concept_id"}
                    ),
                    on="source_concept_id",
                    how="inner",
                )
                # Keep core columns + category
                keep_cols = [
                    "concept_id",
                    "concept_name",
                    "domain_id",
                    "vocabulary_id",
                    "concept_class_id",
                    "standard_concept",
                    "concept_code",
                    "category",
                ]
                mapped = mapped[keep_cols].drop_duplicates()
                print(f"  - Mapped {len(mapped)} non-standard concepts to standard")

        if not mapped.empty:
            matcho_std = pd.concat([standard, mapped], ignore_index=True)
        else:
            matcho_std = standard.copy()

        matcho_std = matcho_std.drop_duplicates(subset=["concept_id", "category"])

        print(f"Final standardized Matcho concepts: {len(matcho_std)}")
        print(f"Categories: {matcho_std['category'].value_counts().to_dict()}")

        self._save_csv(matcho_std, "step1_matcho_concepts_standardized.csv", "1")
        self.matcho_concepts_std = matcho_std
        return matcho_std

    # ---------------------------------------------------------------------
    # STEP 2: Get concept prevalence/specificity (use N3C-derived list)
    # ---------------------------------------------------------------------
    def step2_get_specificity(self) -> pd.DataFrame:
        """
        Use N3C-derived "highly specific" pregnancy concepts.
        Full specificity calculation requires large case/control cohorts
        and is out of scope for small local datasets.
        """
        print("\n" + "=" * 60)
        print("STEP 2: Get concept prevalence and specificity")
        print("=" * 60)
        
        # Check if output already exists
        output_file = DATA_DIR / "step2_highly_specific_concepts.csv"
        if output_file.exists():
            print(f"Loading cached result from {output_file.name}...")
            hs = pd.read_csv(output_file)
            print(f"Loaded {len(hs)} highly specific concepts")
            self.highly_specific_concepts = hs
            return hs

        if self.highly_specific_concepts is not None:
            hs = self.highly_specific_concepts.copy()
        else:
            # Fallback: use PPS concepts as proxy for highly specific concepts
            print("Highly_specific_concepts.xlsx not found; using PPS_concepts as proxy.")
            hs = self.pps_concepts[["domain_concept_id", "domain_concept_name"]].copy()
            hs = hs.rename(columns={"domain_concept_id": "concept_id", "domain_concept_name": "concept_name"})

        print(f"Using {len(hs)} highly specific concepts")
        self._save_csv(hs, "step2_highly_specific_concepts.csv", "2")
        self.highly_specific_concepts = hs
        return hs

    # ---------------------------------------------------------------------
    # STEP 3: Define pregnancy episodes with HIP algorithm (memory-efficient)
    # ---------------------------------------------------------------------
    def step3_define_hip_episodes(self) -> pd.DataFrame:
        """
        Define outcome-based pregnancy episodes using HIP concepts and
        Matcho outcome limits. Uses chunked reading to reduce memory usage.
        """
        print("\n" + "=" * 60)
        print("STEP 3: Define pregnancy episodes with HIP algorithm")
        print("=" * 60)
        
        # Check if output already exists
        output_file = DATA_DIR / "step3_hip_episodes.csv"
        if output_file.exists():
            print(f"Loading cached result from {output_file.name}...")
            episodes = pd.read_csv(output_file)
            # Convert date columns back to datetime
            for col in ["visit_date", "estimated_start_date"]:
                if col in episodes.columns:
                    episodes[col] = pd.to_datetime(episodes[col])
            print(f"Loaded {len(episodes)} HIP episodes")
            self.hip_episodes = episodes
            return episodes

        print("Reading person table...")
        person = _read_omop("person")

        hip_ids = self.hip_concepts["concept_id"].unique().tolist()
        hip_ids_set = set(hip_ids)
        hip_lookup: Dict[int, Dict[str, Optional[float]]] = (
            self.hip_concepts.drop_duplicates(subset=["concept_id"])
            .set_index("concept_id")[["category", "gest_value"]]
            .to_dict("index")
        )
        
        # Filter to women of reproductive age first
        women = person[
            (person.get("gender_concept_id", 0) == 8532)
            & (person.get("year_of_birth", 0) >= 1965)
            & (person.get("year_of_birth", 9999) <= 2005)
        ][["person_id", "year_of_birth", "month_of_birth", "day_of_birth"]].copy()
        print(f"Women of reproductive age: {len(women)}")
        women_ids = set(women["person_id"].tolist())

        print("Reading clinical tables in chunks for HIP concepts...")

        def _extract_domain_chunked(
            table_name: str,
            concept_col: str,
            date_col: str,
            domain_name: str,
            include_value_as_number: bool = False,
        ) -> pd.DataFrame:
            """Read table in chunks, filter immediately for HIP concepts and women."""
            chunks = []
            chunk_size = 100000
            
            # Get file path
            merged = CLINICAL_OMOP_DIR / f"{table_name}.csv"
            if merged.exists():
                files_to_read = [merged]
            else:
                parts = sorted(CLINICAL_OMOP_DIR.glob(f"{table_name}_*.csv"))
                if not parts:
                    return pd.DataFrame(columns=["person_id", "concept_id", "visit_date", "concept_name", "value_as_number", "domain"])
                files_to_read = parts

            print(f"  Processing {table_name}...")
            for file_path in files_to_read:
                try:
                    for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
                        if concept_col not in chunk.columns or date_col not in chunk.columns:
                            continue
                        
                        # Filter immediately: only HIP concepts and women
                        chunk = chunk[
                            chunk[concept_col].isin(hip_ids_set) & 
                            chunk["person_id"].isin(women_ids)
                        ].copy()
                        
                        if chunk.empty:
                            continue
                        
                        chunk = chunk.rename(columns={concept_col: "concept_id", date_col: "visit_date"})
                        chunk["visit_date"] = _parse_date_col(chunk, "visit_date")
                        chunk["concept_name"] = None
                        
                        if include_value_as_number and "value_as_number" in chunk.columns:
                            pass
                        else:
                            chunk["value_as_number"] = np.nan
                        
                        chunk["domain"] = domain_name
                        chunks.append(chunk[["person_id", "concept_id", "visit_date", "concept_name", "value_as_number", "domain"]])
                    
                except Exception as e:
                    print(f"    Warning: Error reading {file_path.name}: {e}")
                    continue
            
            if not chunks:
                return pd.DataFrame(columns=["person_id", "concept_id", "visit_date", "concept_name", "value_as_number", "domain"])
            
            result = pd.concat(chunks, ignore_index=True)
            print(f"    Found {len(result)} relevant records")
            return result

        # Process each domain table with chunked reading
        cond_df = _extract_domain_chunked("condition_occurrence", "condition_concept_id", "condition_start_date", "condition", False)
        proc_df = _extract_domain_chunked("procedure_occurrence", "procedure_concept_id", "procedure_date", "procedure", False)
        meas_df = _extract_domain_chunked("measurement", "measurement_concept_id", "measurement_date", "measurement", True)
        obs_df = _extract_domain_chunked("observation", "observation_concept_id", "observation_date", "observation", True)

        hip_records = pd.concat([cond_df, proc_df, meas_df, obs_df], ignore_index=True)
        hip_records = hip_records.dropna(subset=["visit_date"])

        print(f"\n  Total HIP-related records: {len(hip_records)}")

        # Attach HIP category / gest_value
        hip_records["category"] = hip_records["concept_id"].map(lambda cid: hip_lookup.get(cid, {}).get("category"))
        hip_records["gest_value"] = hip_records["concept_id"].map(lambda cid: hip_lookup.get(cid, {}).get("gest_value"))

        hip_records = hip_records.sort_values(["person_id", "visit_date"])
        self._save_csv(hip_records, "step3a_hip_records.csv", "3")

        episodes = self._define_hip_episodes(hip_records)
        self._save_csv(episodes, "step3_hip_episodes.csv", "3")
        self.hip_episodes = episodes
        return episodes

    def _define_hip_episodes(self, records_df: pd.DataFrame) -> pd.DataFrame:
        """
        Define HIP outcome episodes based on Matcho outcome limits.
        Implements the FULL original Script 3 logic:
        1. Separate processing for each outcome type (LB, SB, ECT, AB, SA, DELIV)
        2. Hierarchical merging of outcomes (LB -> SB -> ECT -> AB/SA -> DELIV)
        3. Gestation record processing and episode creation
        4. Merging gestation and outcome-based episodes
        5. Episode cleaning and validation
        6. Overlap removal
        """
        print("  [FULL SCRIPT 3 IMPLEMENTATION]")
        
        # Step 1: Process each outcome type separately
        lb_visits = self._get_outcome_visits(records_df, "LB")
        sb_visits = self._get_outcome_visits(records_df, "SB")
        ect_visits = self._get_outcome_visits(records_df, "ECT")
        ab_visits = self._get_outcome_visits(records_df, ["AB", "SA"])  # AB and SA treated together
        deliv_visits = self._get_outcome_visits(records_df, "DELIV")
        
        # Step 2: Hierarchical merging (LB -> SB -> ECT -> AB -> DELIV)
        print("  Hierarchically merging outcomes...")
        lb_sb_df = self._add_stillbirth(lb_visits, sb_visits)
        lb_sb_ect_df = self._add_ectopic(lb_sb_df, ect_visits)
        lb_sb_ect_ab_df = self._add_abortion(lb_sb_ect_df, ab_visits)
        all_outcomes_df = self._add_delivery(lb_sb_ect_ab_df, deliv_visits)
        
        # Step 3: Process gestation records
        print("  Processing gestation records...")
        gest_visits_df = self._get_gestation_visits(records_df)
        if not gest_visits_df.empty:
            gest_episodes_df = self._define_gestation_episodes(gest_visits_df)
            min_max_gest_df = self._get_min_max_gestation(gest_episodes_df)
        else:
            min_max_gest_df = pd.DataFrame()
        
        # Step 4: Calculate start dates for outcome episodes
        all_outcomes_df = self._calculate_start_dates(all_outcomes_df)
        
        # Step 5: Merge gestation and outcome episodes
        if not min_max_gest_df.empty:
            print("  Merging gestation and outcome episodes...")
            merged_df = self._merge_gestation_outcomes(all_outcomes_df, min_max_gest_df)
        else:
            merged_df = all_outcomes_df.copy()
            # Add empty gestation columns
            gest_cols = ["gest_id", "first_gest_date", "first_gest_week", "end_gest_date",
                        "end_gest_week", "max_gest_date", "max_gest_week", "min_gest_week",
                        "min_gest_date", "max_gest_start_date"]
            for col in gest_cols:
                merged_df[col] = pd.NA
        
        # Step 6: Clean episodes and remove invalid outcomes
        print("  Cleaning episodes and validating against gestational age...")
        cleaned_df = self._clean_episodes(merged_df)
        
        # Step 7: Remove overlapping episodes
        print("  Removing overlapping episodes...")
        final_df = self._remove_overlaps(cleaned_df)
        
        # Step 8: Finalize episode structure
        final_df = final_df.sort_values(["person_id", "visit_date"])
        final_df["episode"] = final_df.groupby("person_id").cumcount() + 1
        
        # Rename max_start_date to estimated_start_date for output
        if "max_start_date" in final_df.columns:
            final_df = final_df.rename(columns={"max_start_date": "estimated_start_date"})
        
        # Select and rename columns to match expected output
        output_cols = ["person_id", "visit_date", "category", "concept_id", "concept_name",
                      "gest_value", "episode", "estimated_start_date"]
        available_cols = [c for c in output_cols if c in final_df.columns]
        final_df = final_df[available_cols].copy()
        
        return final_df

    # =========================================================================
    # SCRIPT 3 HELPER METHODS: Full HIP Algorithm Implementation
    # =========================================================================
    
    def _get_outcome_visits(self, records_df: pd.DataFrame, categories) -> pd.DataFrame:
        """Get all visits for specific outcome category/categories."""
        if isinstance(categories, str):
            categories = [categories]
        
        df = records_df[records_df["category"].isin(categories)].copy()
        if df.empty:
            return pd.DataFrame()
        
        df = df.drop_duplicates(subset=["person_id", "visit_date"])
        df = df.sort_values(["person_id", "visit_date"])
        df["visit_date"] = pd.to_datetime(df["visit_date"])
        
        category_key = categories[0]
        lim = self.matcho_outcome_limits
        same_type = lim[(lim["first_preg_category"] == category_key) & (lim["outcome_preg_category"] == category_key)]
        min_days = int(same_type.iloc[0]["min_days"]) if not same_type.empty else 168
        
        df["prev_visit"] = df.groupby("person_id")["visit_date"].shift(1)
        df["days_diff"] = (df["visit_date"] - df["prev_visit"]).dt.days
        df["is_new_episode"] = df["prev_visit"].isna() | (df["days_diff"] >= min_days)
        df = df[df["is_new_episode"]].copy()
        df = df.drop(columns=["prev_visit", "days_diff", "is_new_episode"])
        
        print(f"    {category_key}: {len(df)} episodes")
        return df
    
    def _add_stillbirth(self, lb_df: pd.DataFrame, sb_df: pd.DataFrame) -> pd.DataFrame:
        """Add stillbirth visits following Script 3 logic."""
        if lb_df.empty and sb_df.empty:
            return pd.DataFrame()
        if sb_df.empty:
            return lb_df.copy()
        if lb_df.empty:
            return sb_df.copy()
        
        lim = self.matcho_outcome_limits
        lb_to_sb = lim[(lim["first_preg_category"] == "LB") & (lim["outcome_preg_category"] == "SB")]
        sb_to_lb = lim[(lim["first_preg_category"] == "SB") & (lim["outcome_preg_category"] == "LB")]
        before_min = int(lb_to_sb.iloc[0]["min_days"]) if not lb_to_sb.empty else 168
        after_min = int(sb_to_lb.iloc[0]["min_days"]) if not sb_to_lb.empty else 168
        
        lb_clean = lb_df.drop(columns=["gest_value", "value_as_number"], errors="ignore")
        sb_clean = sb_df.drop(columns=["gest_value", "value_as_number"], errors="ignore")
        
        temp_df = pd.concat([lb_clean, sb_clean], ignore_index=True)
        temp_df = temp_df.sort_values(["person_id", "visit_date"])
        
        temp_df["after_days"] = temp_df.groupby("person_id")["visit_date"].diff().dt.days
        temp_df["previous_category"] = temp_df.groupby("person_id")["category"].shift(1)
        temp_df["before_days"] = temp_df.groupby("person_id")["visit_date"].diff(-1).dt.days
        temp_df["next_category"] = temp_df.groupby("person_id")["category"].shift(-1)
        
        sb_temp = temp_df[temp_df["category"] == "SB"].copy()
        
        sb_null = sb_temp[(sb_temp["before_days"].isna()) & (sb_temp["after_days"].isna())]
        sb_not_lb_prev = sb_temp[(sb_temp["previous_category"] != "LB") & (sb_temp["next_category"].isna())]
        sb_not_lb_next = sb_temp[(sb_temp["next_category"] != "LB") & (sb_temp["previous_category"].isna())]
        sb_not_lb_both = sb_temp[(sb_temp["previous_category"] != "LB") & (sb_temp["next_category"] != "LB")]
        sb_after_lb = sb_temp[(sb_temp["previous_category"] == "LB") & (sb_temp["after_days"] >= before_min) & (sb_temp["next_category"].isna())]
        sb_before_lb = sb_temp[(sb_temp["next_category"] == "LB") & (sb_temp["before_days"].abs() >= after_min) & (sb_temp["previous_category"].isna())]
        sb_between_lb = sb_temp[(sb_temp["next_category"] == "LB") & (sb_temp["before_days"].abs() >= after_min) & (sb_temp["previous_category"] == "LB") & (sb_temp["after_days"] >= before_min)]
        
        valid_sb = pd.concat([sb_null, sb_not_lb_prev, sb_not_lb_next, sb_not_lb_both, sb_after_lb, sb_before_lb, sb_between_lb], ignore_index=True)
        valid_sb = valid_sb.drop(columns=["previous_category", "next_category", "before_days", "after_days"])
        
        final_df = pd.concat([lb_clean, valid_sb], ignore_index=True)
        final_df = final_df.drop_duplicates()
        
        print(f"    After adding SB: {len(final_df[final_df['category']=='LB'])} LB, {len(final_df[final_df['category']=='SB'])} SB")
        return final_df
    
    def _add_ectopic(self, prev_df: pd.DataFrame, ect_df: pd.DataFrame) -> pd.DataFrame:
        """Add ectopic pregnancy visits following Script 3 logic."""
        if ect_df.empty:
            return prev_df.copy()
        if prev_df.empty:
            return ect_df.copy()
        
        lim = self.matcho_outcome_limits
        lb_to_ect = lim[(lim["first_preg_category"] == "LB") & (lim["outcome_preg_category"] == "ECT")]
        ect_to_lb = lim[(lim["first_preg_category"] == "ECT") & (lim["outcome_preg_category"] == "LB")]
        ect_to_sb = lim[(lim["first_preg_category"] == "ECT") & (lim["outcome_preg_category"] == "SB")]
        
        before_min = int(lb_to_ect.iloc[0]["min_days"]) if not lb_to_ect.empty else 168
        after_min_lb = int(ect_to_lb.iloc[0]["min_days"]) if not ect_to_lb.empty else 168
        after_min_sb = int(ect_to_sb.iloc[0]["min_days"]) if not ect_to_sb.empty else 168
        
        ect_clean = ect_df.drop(columns=["gest_value", "value_as_number"], errors="ignore")
        
        temp_df = pd.concat([prev_df, ect_clean], ignore_index=True)
        temp_df = temp_df.sort_values(["person_id", "visit_date"])
        
        temp_df["after_days"] = temp_df.groupby("person_id")["visit_date"].diff().dt.days
        temp_df["previous_category"] = temp_df.groupby("person_id")["category"].shift(1)
        temp_df["before_days"] = temp_df.groupby("person_id")["visit_date"].diff(-1).dt.days
        temp_df["next_category"] = temp_df.groupby("person_id")["category"].shift(-1)
        
        ect_temp = temp_df[temp_df["category"] == "ECT"].copy()
        
        ect_null = ect_temp[(ect_temp["before_days"].isna()) & (ect_temp["after_days"].isna())]
        ect_not_lbsb_prev = ect_temp[(~ect_temp["previous_category"].isin(["LB", "SB"])) & (ect_temp["next_category"].isna())]
        ect_not_lbsb_next = ect_temp[(~ect_temp["next_category"].isin(["LB", "SB"])) & (ect_temp["previous_category"].isna())]
        ect_not_lbsb_both = ect_temp[(~ect_temp["previous_category"].isin(["LB", "SB"])) & (~ect_temp["next_category"].isin(["LB", "SB"]))]
        ect_after_lbsb = ect_temp[(ect_temp["previous_category"].isin(["LB", "SB"])) & (ect_temp["after_days"] >= before_min) & (ect_temp["next_category"].isna())]
        ect_before_lb = ect_temp[(ect_temp["next_category"] == "LB") & (ect_temp["before_days"].abs() >= after_min_lb) & (ect_temp["previous_category"].isna())]
        ect_before_sb = ect_temp[(ect_temp["next_category"] == "SB") & (ect_temp["before_days"].abs() >= after_min_sb) & (ect_temp["previous_category"].isna())]
        ect_between_lb = ect_temp[(ect_temp["next_category"] == "LB") & (ect_temp["before_days"].abs() >= after_min_lb) & (ect_temp["previous_category"].isin(["LB", "SB"])) & (ect_temp["after_days"] >= before_min)]
        ect_between_sb = ect_temp[(ect_temp["next_category"] == "SB") & (ect_temp["before_days"].abs() >= after_min_sb) & (ect_temp["previous_category"].isin(["LB", "SB"])) & (ect_temp["after_days"] >= before_min)]
        
        valid_ect = pd.concat([ect_null, ect_not_lbsb_prev, ect_not_lbsb_next, ect_not_lbsb_both, ect_after_lbsb, ect_before_lb, ect_before_sb, ect_between_lb, ect_between_sb], ignore_index=True)
        valid_ect = valid_ect.drop(columns=["previous_category", "next_category", "before_days", "after_days"])
        
        final_df = pd.concat([prev_df, valid_ect], ignore_index=True)
        final_df = final_df.drop_duplicates()
        
        print(f"    After adding ECT: {len(final_df[final_df['category']=='ECT'])} ECT")
        return final_df
    
    def _add_abortion(self, prev_df: pd.DataFrame, ab_df: pd.DataFrame) -> pd.DataFrame:
        """Add abortion visits following Script 3 logic."""
        if ab_df.empty:
            return prev_df.copy()
        if prev_df.empty:
            return ab_df.copy()
        
        lim = self.matcho_outcome_limits
        lb_to_ab = lim[(lim["first_preg_category"] == "LB") & (lim["outcome_preg_category"] == "AB")]
        ect_to_ab = lim[(lim["first_preg_category"] == "ECT") & (lim["outcome_preg_category"] == "AB")]
        ab_to_lb = lim[(lim["first_preg_category"] == "AB") & (lim["outcome_preg_category"] == "LB")]
        ab_to_sb = lim[(lim["first_preg_category"] == "AB") & (lim["outcome_preg_category"] == "SB")]
        ab_to_ect = lim[(lim["first_preg_category"] == "AB") & (lim["outcome_preg_category"] == "ECT")]
        
        before_min_lb = int(lb_to_ab.iloc[0]["min_days"]) if not lb_to_ab.empty else 168
        before_min_ect = int(ect_to_ab.iloc[0]["min_days"]) if not ect_to_ab.empty else 56
        after_min_lb = int(ab_to_lb.iloc[0]["min_days"]) if not ab_to_lb.empty else 168
        after_min_sb = int(ab_to_sb.iloc[0]["min_days"]) if not ab_to_sb.empty else 168
        after_min_ect = int(ab_to_ect.iloc[0]["min_days"]) if not ab_to_ect.empty else 56
        
        ab_clean = ab_df.drop(columns=["gest_value", "value_as_number"], errors="ignore")
        
        temp_df = pd.concat([prev_df, ab_clean], ignore_index=True)
        temp_df = temp_df.sort_values(["person_id", "visit_date"])
        temp_df["temp_category"] = temp_df["category"].apply(lambda x: "AB" if x == "SA" else x)
        
        temp_df["after_days"] = temp_df.groupby("person_id")["visit_date"].diff().dt.days
        temp_df["previous_category"] = temp_df.groupby("person_id")["temp_category"].shift(1)
        temp_df["before_days"] = temp_df.groupby("person_id")["visit_date"].diff(-1).dt.days
        temp_df["next_category"] = temp_df.groupby("person_id")["temp_category"].shift(-1)
        
        ab_temp = temp_df[temp_df["temp_category"] == "AB"].copy()
        
        ab_null = ab_temp[(ab_temp["before_days"].isna()) & (ab_temp["after_days"].isna())]
        ab_not_all_prev = ab_temp[(~ab_temp["previous_category"].isin(["LB", "SB", "ECT"])) & (ab_temp["next_category"].isna())]
        ab_not_all_next = ab_temp[(~ab_temp["next_category"].isin(["LB", "SB", "ECT"])) & (ab_temp["previous_category"].isna())]
        ab_not_all_both = ab_temp[(~ab_temp["previous_category"].isin(["LB", "SB", "ECT"])) & (~ab_temp["next_category"].isin(["LB", "SB", "ECT"]))]
        ab_after_lbsb = ab_temp[(ab_temp["previous_category"].isin(["LB", "SB"])) & (ab_temp["after_days"] >= before_min_lb) & (ab_temp["next_category"].isna())]
        ab_before_lb = ab_temp[(ab_temp["next_category"] == "LB") & (ab_temp["before_days"].abs() >= after_min_lb) & (ab_temp["previous_category"].isna())]
        ab_before_sb = ab_temp[(ab_temp["next_category"] == "SB") & (ab_temp["before_days"].abs() >= after_min_sb) & (ab_temp["previous_category"].isna())]
        ab_between_lb = ab_temp[(ab_temp["next_category"] == "LB") & (ab_temp["before_days"].abs() >= after_min_lb) & (ab_temp["previous_category"].isin(["LB", "SB"])) & (ab_temp["after_days"] >= before_min_lb)]
        ab_between_sb = ab_temp[(ab_temp["next_category"] == "SB") & (ab_temp["before_days"].abs() >= after_min_sb) & (ab_temp["previous_category"].isin(["LB", "SB"])) & (ab_temp["after_days"] >= before_min_lb)]
        ab_after_ect = ab_temp[(ab_temp["previous_category"] == "ECT") & (ab_temp["after_days"] >= before_min_ect) & (ab_temp["next_category"].isna())]
        ab_before_ect = ab_temp[(ab_temp["next_category"] == "ECT") & (ab_temp["before_days"].abs() >= after_min_ect) & (ab_temp["previous_category"].isna())]
        ab_between_ect = ab_temp[(ab_temp["next_category"] == "ECT") & (ab_temp["before_days"].abs() >= after_min_ect) & (ab_temp["previous_category"] == "ECT") & (ab_temp["after_days"] >= before_min_ect)]
        ab_between_ect_lbsb = ab_temp[(ab_temp["next_category"] == "ECT") & (ab_temp["before_days"].abs() >= after_min_ect) & (ab_temp["previous_category"].isin(["LB", "SB"])) & (ab_temp["after_days"] >= before_min_lb)]
        ab_between_lbsb_ect = ab_temp[(ab_temp["next_category"].isin(["LB", "SB"])) & (ab_temp["before_days"].abs() >= after_min_lb) & (ab_temp["previous_category"] == "ECT") & (ab_temp["after_days"] >= before_min_ect)]
        
        valid_ab = pd.concat([ab_null, ab_not_all_prev, ab_not_all_next, ab_not_all_both, ab_after_lbsb, ab_before_lb, ab_before_sb, ab_between_lb, ab_between_sb, ab_after_ect, ab_before_ect, ab_between_ect, ab_between_ect_lbsb, ab_between_lbsb_ect], ignore_index=True)
        valid_ab = valid_ab.drop(columns=["temp_category", "previous_category", "next_category", "before_days", "after_days"])
        
        final_df = pd.concat([prev_df, valid_ab], ignore_index=True)
        final_df = final_df.drop_duplicates()
        
        print(f"    After adding AB/SA: {len(final_df[final_df['category'].isin(['AB','SA'])])} AB/SA")
        return final_df
    
    def _add_delivery(self, prev_df: pd.DataFrame, deliv_df: pd.DataFrame) -> pd.DataFrame:
        """Add delivery record only visits following Script 3 logic."""
        if deliv_df.empty:
            return prev_df.copy()
        if prev_df.empty:
            return deliv_df.copy()
        
        lim = self.matcho_outcome_limits
        lb_to_deliv = lim[(lim["first_preg_category"] == "LB") & (lim["outcome_preg_category"] == "DELIV")]
        ect_to_deliv = lim[(lim["first_preg_category"] == "ECT") & (lim["outcome_preg_category"] == "DELIV")]
        deliv_to_lb = lim[(lim["first_preg_category"] == "DELIV") & (lim["outcome_preg_category"] == "LB")]
        deliv_to_sb = lim[(lim["first_preg_category"] == "DELIV") & (lim["outcome_preg_category"] == "SB")]
        deliv_to_ect = lim[(lim["first_preg_category"] == "DELIV") & (lim["outcome_preg_category"] == "ECT")]
        
        before_min_lb = int(lb_to_deliv.iloc[0]["min_days"]) if not lb_to_deliv.empty else 168
        before_min_ect = int(ect_to_deliv.iloc[0]["min_days"]) if not ect_to_deliv.empty else 56
        after_min_lb = int(deliv_to_lb.iloc[0]["min_days"]) if not deliv_to_lb.empty else 168
        after_min_sb = int(deliv_to_sb.iloc[0]["min_days"]) if not deliv_to_sb.empty else 168
        after_min_ect = int(deliv_to_ect.iloc[0]["min_days"]) if not deliv_to_ect.empty else 56
        
        deliv_clean = deliv_df.drop(columns=["gest_value", "value_as_number"], errors="ignore")
        
        temp_df = pd.concat([prev_df, deliv_clean], ignore_index=True)
        temp_df = temp_df.sort_values(["person_id", "visit_date"])
        temp_df["temp_category"] = temp_df["category"].apply(lambda x: "AB" if x == "SA" else x)
        
        temp_df["after_days"] = temp_df.groupby("person_id")["visit_date"].diff().dt.days
        temp_df["previous_category"] = temp_df.groupby("person_id")["temp_category"].shift(1)
        temp_df["before_days"] = temp_df.groupby("person_id")["visit_date"].diff(-1).dt.days
        temp_df["next_category"] = temp_df.groupby("person_id")["temp_category"].shift(-1)
        
        deliv_temp = temp_df[temp_df["temp_category"] == "DELIV"].copy()
        
        deliv_null = deliv_temp[(deliv_temp["before_days"].isna()) & (deliv_temp["after_days"].isna())]
        deliv_not_all_prev = deliv_temp[(~deliv_temp["previous_category"].isin(["LB", "SB", "ECT", "AB"])) & (deliv_temp["next_category"].isna())]
        deliv_not_all_next = deliv_temp[(~deliv_temp["next_category"].isin(["LB", "SB", "ECT", "AB"])) & (deliv_temp["previous_category"].isna())]
        deliv_not_all_both = deliv_temp[(~deliv_temp["previous_category"].isin(["LB", "SB", "ECT", "AB"])) & (~deliv_temp["next_category"].isin(["LB", "SB", "ECT", "AB"]))]
        deliv_after_lbsb = deliv_temp[(deliv_temp["previous_category"].isin(["LB", "SB"])) & (deliv_temp["after_days"] >= before_min_lb) & (deliv_temp["next_category"].isna())]
        deliv_before_lb = deliv_temp[(deliv_temp["next_category"] == "LB") & (deliv_temp["before_days"].abs() >= after_min_lb) & (deliv_temp["previous_category"].isna())]
        deliv_before_sb = deliv_temp[(deliv_temp["next_category"] == "SB") & (deliv_temp["before_days"].abs() >= after_min_sb) & (deliv_temp["previous_category"].isna())]
        deliv_between_lb = deliv_temp[(deliv_temp["next_category"] == "LB") & (deliv_temp["before_days"].abs() >= after_min_lb) & (deliv_temp["previous_category"].isin(["LB", "SB"])) & (deliv_temp["after_days"] >= before_min_lb)]
        deliv_between_sb = deliv_temp[(deliv_temp["next_category"] == "SB") & (deliv_temp["before_days"].abs() >= after_min_sb) & (deliv_temp["previous_category"].isin(["LB", "SB"])) & (deliv_temp["after_days"] >= before_min_lb)]
        deliv_after_ect_ab = deliv_temp[(deliv_temp["previous_category"].isin(["ECT", "AB"])) & (deliv_temp["after_days"] >= before_min_ect) & (deliv_temp["next_category"].isna())]
        deliv_before_ect_ab = deliv_temp[(deliv_temp["next_category"].isin(["ECT", "AB"])) & (deliv_temp["before_days"].abs() >= after_min_ect) & (deliv_temp["previous_category"].isna())]
        deliv_between_ect_ab = deliv_temp[(deliv_temp["next_category"].isin(["ECT", "AB"])) & (deliv_temp["before_days"].abs() >= after_min_ect) & (deliv_temp["previous_category"].isin(["ECT", "AB"])) & (deliv_temp["after_days"] >= before_min_ect)]
        deliv_between_ect_ab_lbsb = deliv_temp[(deliv_temp["next_category"].isin(["ECT", "AB"])) & (deliv_temp["before_days"].abs() >= after_min_ect) & (deliv_temp["previous_category"].isin(["LB", "SB"])) & (deliv_temp["after_days"] >= before_min_lb)]
        deliv_between_lbsb_ect_ab = deliv_temp[(deliv_temp["next_category"].isin(["LB", "SB"])) & (deliv_temp["before_days"].abs() >= after_min_lb) & (deliv_temp["previous_category"].isin(["ECT", "AB"])) & (deliv_temp["after_days"] >= before_min_ect)]
        
        valid_deliv = pd.concat([deliv_null, deliv_not_all_prev, deliv_not_all_next, deliv_not_all_both, deliv_after_lbsb, deliv_before_lb, deliv_before_sb, deliv_between_lb, deliv_between_sb, deliv_after_ect_ab, deliv_before_ect_ab, deliv_between_ect_ab, deliv_between_ect_ab_lbsb, deliv_between_lbsb_ect_ab], ignore_index=True)
        valid_deliv = valid_deliv.drop(columns=["temp_category", "previous_category", "next_category", "before_days", "after_days"])
        
        final_df = pd.concat([prev_df, valid_deliv], ignore_index=True)
        final_df = final_df.drop_duplicates()
        
        print(f"    After adding DELIV: {len(final_df[final_df['category']=='DELIV'])} DELIV")
        print(f"    TOTAL OUTCOMES: {len(final_df)} episodes")
        return final_df
    
    def _get_gestation_visits(self, records_df: pd.DataFrame) -> pd.DataFrame:
        """Get all records with gestation period information."""
        df = records_df.copy()
        gest_df = df[df["gest_value"].notna()].copy()
        
        gest_concepts = [3002209, 3048230, 3012266]
        other_gest_df = df[(df["concept_id"].isin(gest_concepts)) & (df["value_as_number"].notna())].copy()
        other_gest_df["gest_value"] = other_gest_df["value_as_number"].astype(int)
        
        all_gest_df = pd.concat([gest_df, other_gest_df], ignore_index=True)
        all_gest_df = all_gest_df.drop_duplicates()
        
        return all_gest_df
    
    def _define_gestation_episodes(self, gest_visits_df: pd.DataFrame) -> pd.DataFrame:
        """Define pregnancy episodes based on gestational records."""
        df = gest_visits_df.copy()
        min_days = 70
        buffer_days = 28
        
        df = df[df["visit_date"].notna()]
        df = df[(df["gest_value"] > 0) & (df["gest_value"] <= 44)]
        df = df.sort_values(["person_id", "visit_date", "gest_value"], ascending=[True, True, False])
        df = df.drop_duplicates(subset=["person_id", "visit_date"], keep="first")
        
        df["gest_day"] = df["gest_value"] * 7
        df = df.sort_values(["person_id", "visit_date"])
        
        df["prev_week"] = df.groupby("person_id")["gest_value"].shift(1)
        df["prev_date"] = df.groupby("person_id")["visit_date"].shift(1)
        df["week_diff"] = df["gest_value"] - df["prev_week"]
        df["day_diff"] = df["week_diff"] * 7 + buffer_days
        df["date_diff"] = (pd.to_datetime(df["visit_date"]) - pd.to_datetime(df["prev_date"])).dt.days
        
        df["new_diff"] = df.apply(lambda row: 1 if (row["date_diff"] < min_days and row["week_diff"] <= 0) else row["week_diff"], axis=1)
        df["new_diff2"] = df.apply(lambda row: -1 if (row["date_diff"] >= row["day_diff"] and row["week_diff"] > 0) else row["new_diff"], axis=1)
        
        df["is_new_episode"] = (df["new_diff2"] <= 0) | (df["prev_week"].isna())
        df["episode"] = df.groupby("person_id")["is_new_episode"].cumsum()
        
        return df
    
    def _get_min_max_gestation(self, gest_episodes_df: pd.DataFrame) -> pd.DataFrame:
        """Get min/max gestational age per episode."""
        df = gest_episodes_df.copy()
        
        first_gest = df.groupby(["person_id", "episode"]).agg(first_gest_date=("visit_date", "min")).reset_index()
        first_gest = first_gest.merge(df, on=["person_id", "episode"])
        first_gest = first_gest[first_gest["visit_date"] == first_gest["first_gest_date"]]
        first_gest = first_gest.groupby(["person_id", "episode"]).agg(first_gest_date=("first_gest_date", "first"), first_gest_week=("gest_value", "max")).reset_index()
        
        min_gest = df.groupby(["person_id", "episode"]).agg(min_gest_week=("gest_value", "min")).reset_index()
        min_gest = min_gest.merge(df, on=["person_id", "episode"])
        min_gest = min_gest[min_gest["gest_value"] == min_gest["min_gest_week"]]
        min_gest = min_gest.groupby(["person_id", "episode"]).agg(min_gest_week=("min_gest_week", "first"), min_gest_date=("visit_date", "min"), min_gest_date_2=("visit_date", "max")).reset_index()
        
        end_gest = df.groupby(["person_id", "episode"]).agg(end_gest_date=("visit_date", "max")).reset_index()
        end_gest = end_gest.merge(df, on=["person_id", "episode"])
        end_gest = end_gest[end_gest["visit_date"] == end_gest["end_gest_date"]]
        end_gest = end_gest.groupby(["person_id", "episode"]).agg(end_gest_date=("end_gest_date", "first"), end_gest_week=("gest_value", "max")).reset_index()
        
        max_gest = df.groupby(["person_id", "episode"]).agg(max_gest_week=("gest_value", "max")).reset_index()
        max_gest = max_gest.merge(df, on=["person_id", "episode"])
        max_gest = max_gest[max_gest["gest_value"] == max_gest["max_gest_week"]]
        max_gest = max_gest.groupby(["person_id", "episode"]).agg(max_gest_week=("max_gest_week", "first"), max_gest_date=("visit_date", "min")).reset_index()
        
        result = first_gest.merge(end_gest, on=["person_id", "episode"])
        result = result.merge(min_gest, on=["person_id", "episode"])
        result = result.merge(max_gest, on=["person_id", "episode"])
        
        return result
    
    def _calculate_start_dates(self, outcomes_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate estimated start dates."""
        df = outcomes_df.copy()
        term_durations = self.matcho_term_durations.set_index("category").to_dict("index")
        
        def _estimate_start(row):
            cat = row["category"]
            end = pd.to_datetime(row["visit_date"])
            if pd.isna(end):
                return pd.NaT
            if cat in term_durations:
                max_term = int(term_durations[cat]["max_term"])
            else:
                max_term = 280
            return end - timedelta(days=max_term)
        
        df["min_start_date"] = df.apply(lambda row: pd.to_datetime(row["visit_date"]) - timedelta(days=int(term_durations.get(row["category"], {}).get("min_term", 140))) if row["category"] in term_durations else pd.NaT, axis=1)
        df["max_start_date"] = df.apply(_estimate_start, axis=1)
        
        return df
    
    def _merge_gestation_outcomes(self, outcomes_df: pd.DataFrame, gest_df: pd.DataFrame) -> pd.DataFrame:
        """Merge gestation and outcome episodes."""
        outcomes_df = outcomes_df.copy()
        outcomes_df["visit_id"] = outcomes_df["person_id"].astype(str) + "_" + outcomes_df["visit_date"].astype(str)
        
        gest_df = gest_df.copy()
        gest_df["gest_id"] = gest_df["person_id"].astype(str) + "_" + gest_df["max_gest_date"].astype(str)
        
        merged = outcomes_df.merge(gest_df, on="person_id", how="outer", suffixes=("", "_gest"))
        
        gest_only = merged[merged["visit_date"].isna()].copy()
        if not gest_only.empty:
            gest_only["category"] = "PREG"
            gest_only["visit_date"] = gest_only["max_gest_date"]
            gest_only["max_start_date"] = gest_only["max_gest_date"] - pd.to_timedelta(gest_only["max_gest_week"] * 7, unit="D")
        
        outcome_only = merged[merged["gest_id"].isna()].copy()
        both = merged[(merged["visit_date"].notna()) & (merged["gest_id"].notna())].copy()
        
        result = pd.concat([outcome_only, gest_only, both], ignore_index=True)
        result = result.drop_duplicates()
        
        return result
    
    def _clean_episodes(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """Clean episodes and reclassify if inconsistent with gestational age."""
        df = merged_df.copy()
        
        if "max_gest_week" in df.columns and "max_start_date" in df.columns:
            df["gest_at_outcome"] = (pd.to_datetime(df["visit_date"]) - pd.to_datetime(df["max_start_date"])).dt.days
            term_dur = self.matcho_term_durations.set_index("category").to_dict("index")
            
            def check_valid(row):
                if pd.isna(row.get("gest_at_outcome")) or row["category"] == "PREG":
                    return row["category"]
                if row["category"] not in term_dur:
                    return row["category"]
                min_term = term_dur[row["category"]]["min_term"]
                max_term = term_dur[row["category"]]["max_term"]
                if row["gest_at_outcome"] < min_term or row["gest_at_outcome"] > max_term:
                    return "PREG"
                return row["category"]
            
            df["category"] = df.apply(check_valid, axis=1)
        
        return df
    
    def _remove_overlaps(self, cleaned_df: pd.DataFrame) -> pd.DataFrame:
        """Remove overlapping episodes."""
        df = cleaned_df.copy()
        df = df.sort_values(["person_id", "visit_date"])
        
        df["prev_date"] = df.groupby("person_id")["visit_date"].shift(1)
        df["prev_category"] = df.groupby("person_id")["category"].shift(1)
        
        if "max_start_date" in df.columns:
            df["prev_date_diff"] = (pd.to_datetime(df["max_start_date"]) - pd.to_datetime(df["prev_date"])).dt.days
            df["has_overlap"] = df["prev_date_diff"] < 0
            
            df["remove"] = False
            for idx in range(1, len(df)):
                if df.iloc[idx]["has_overlap"] and df.iloc[idx-1]["category"] == "PREG":
                    df.at[df.index[idx-1], "remove"] = True
            
            df = df[~df["remove"]].copy()
            df = df.drop(columns=["remove", "has_overlap", "prev_date_diff"])
        
        df = df.drop(columns=["prev_date", "prev_category"], errors="ignore")
        
        return df

    # ---------------------------------------------------------------------
    # STEP 4: Identify gestational timing-specific concepts (PPS)
    # ---------------------------------------------------------------------
    def step4_identify_gt_concepts(self) -> pd.DataFrame:
        """
        Persist PPS concepts used for gestational timing.
        """
        print("\n" + "=" * 60)
        print("STEP 4: Identify gestational timing specific concepts")
        print("=" * 60)
        
        # Check if output already exists
        output_file = DATA_DIR / "step4_pps_concepts.csv"
        if output_file.exists():
            print(f"Loading cached result from {output_file.name}...")
            pps = pd.read_csv(output_file)
            print(f"Loaded {len(pps)} PPS concepts")
            return pps

        pps = self.pps_concepts.copy()
        self._save_csv(pps, "step4_pps_concepts.csv", "4")
        return pps

    # ---------------------------------------------------------------------
    # STEP 5: Define pregnancy episodes with PPS algorithm (memory-efficient)
    # ---------------------------------------------------------------------
    def step5_define_pps_episodes(self) -> pd.DataFrame:
        """
        Define PPS episodes using gestational timing concepts. Uses chunked
        reading to reduce memory usage.
        """
        print("\n" + "=" * 60)
        print("STEP 5: Define pregnancy episodes with PPS algorithm")
        print("=" * 60)
        
        # Check if output already exists
        output_file = DATA_DIR / "step5_pps_episodes.csv"
        if output_file.exists():
            print(f"Loading cached result from {output_file.name}...")
            episodes = pd.read_csv(output_file)
            # Convert date columns back to datetime
            for col in ["episode_min_date", "episode_max_date", "episode_max_date_plus_two_months"]:
                if col in episodes.columns:
                    episodes[col] = pd.to_datetime(episodes[col])
            print(f"Loaded {len(episodes)} PPS episodes")
            self.pps_episodes = episodes
            return episodes

        print("Reading person table...")
        person = _read_omop("person")
        
        # Filter to women of reproductive age first
        women = person[
            (person.get("gender_concept_id", 0) == 8532)
            & (person.get("year_of_birth", 0) >= 1965)
            & (person.get("year_of_birth", 9999) <= 2005)
        ][["person_id"]].copy()
        print(f"Women of reproductive age: {len(women)}")
        women_ids = set(women["person_id"].tolist())

        pps_ids = self.pps_concepts["domain_concept_id"].tolist()
        pps_ids_set = set(pps_ids)
        gt_lookup = self.pps_concepts.set_index("domain_concept_id")[["min_month", "max_month"]].to_dict("index")

        print("Reading clinical tables in chunks for PPS concepts...")

        def _extract_chunked(
            table_name: str,
            concept_col: str,
            date_col: str,
        ) -> pd.DataFrame:
            """Read table in chunks, filter immediately for PPS concepts and women."""
            chunks = []
            chunk_size = 100000
            
            merged = CLINICAL_OMOP_DIR / f"{table_name}.csv"
            if merged.exists():
                files_to_read = [merged]
            else:
                parts = sorted(CLINICAL_OMOP_DIR.glob(f"{table_name}_*.csv"))
                if not parts:
                    return pd.DataFrame(columns=["person_id", "concept_id", "visit_date", "concept_name"])
                files_to_read = parts

            print(f"  Processing {table_name}...")
            for file_path in files_to_read:
                try:
                    for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
                        if concept_col not in chunk.columns or date_col not in chunk.columns:
                            continue
                        
                        chunk = chunk[
                            chunk[concept_col].isin(pps_ids_set) & 
                            chunk["person_id"].isin(women_ids)
                        ].copy()
                        
                        if chunk.empty:
                            continue
                        
                        chunk = chunk.rename(columns={concept_col: "concept_id", date_col: "visit_date"})
                        chunk["visit_date"] = _parse_date_col(chunk, "visit_date")
                        chunk["concept_name"] = None
                        chunks.append(chunk[["person_id", "concept_id", "visit_date", "concept_name"]])
                    
                except Exception as e:
                    print(f"    Warning: Error reading {file_path.name}: {e}")
                    continue
            
            if not chunks:
                return pd.DataFrame(columns=["person_id", "concept_id", "visit_date", "concept_name"])
            
            result = pd.concat(chunks, ignore_index=True)
            print(f"    Found {len(result)} relevant records")
            return result

        cond_df = _extract_chunked("condition_occurrence", "condition_concept_id", "condition_start_date")
        proc_df = _extract_chunked("procedure_occurrence", "procedure_concept_id", "procedure_date")
        meas_df = _extract_chunked("measurement", "measurement_concept_id", "measurement_date")
        obs_df = _extract_chunked("observation", "observation_concept_id", "observation_date")

        pps_records = pd.concat([cond_df, proc_df, meas_df, obs_df], ignore_index=True)
        pps_records = pps_records.dropna(subset=["visit_date"]).drop_duplicates()

        if pps_records.empty:
            print("No PPS concept records found.")
            self.pps_episodes = pd.DataFrame()
            return self.pps_episodes
        
        print(f"\n  Total PPS-related records: {len(pps_records)}")

        # Attach gestational timing ranges
        pps_records["min_month"] = pps_records["concept_id"].map(lambda cid: gt_lookup.get(cid, {}).get("min_month"))
        pps_records["max_month"] = pps_records["concept_id"].map(lambda cid: gt_lookup.get(cid, {}).get("max_month"))

        pps_records = pps_records.sort_values(["person_id", "visit_date"])
        self._save_csv(pps_records, "step5a_pps_records.csv", "5")

        episodes = self._define_pps_episodes(pps_records)
        self._save_csv(episodes, "step5_pps_episodes.csv", "5")
        self.pps_episodes = episodes
        return episodes

    @staticmethod
    def _define_pps_episodes(records_df: pd.DataFrame) -> pd.DataFrame:
        """
        Define PPS episodes based on temporal consistency.
        Implements Script 5 logic including bridge comparison (lines 119-196).
        """
        episodes: List[Dict[str, object]] = []

        for person_id, grp in records_df.groupby("person_id"):
            grp = grp.sort_values("visit_date").reset_index(drop=True)
            if grp.empty:
                continue

            episode_num = 1
            indices: List[int] = [0]

            for i in range(1, len(grp)):
                cur = grp.iloc[i]
                prev = grp.iloc[i - 1]
                delta_months = (cur["visit_date"] - prev["visit_date"]).days / 30.0

                # Check consistency with previous records
                agreement_t_c = False
                
                # Check against all previous records in current episode
                for j in range(1, (i + 1)):
                    if i - j < 0:
                        break
                    prev_j = grp.iloc[i - j]
                    delta_t = (cur["visit_date"] - prev_j["visit_date"]).days / 30.0
                    
                    min_m_prev = prev_j.get("min_month", np.nan)
                    max_m_prev = prev_j.get("max_month", np.nan)
                    min_m_cur = cur.get("min_month", np.nan)
                    max_m_cur = cur.get("max_month", np.nan)
                    
                    if not (np.isnan(min_m_prev) or np.isnan(max_m_prev) or np.isnan(min_m_cur) or np.isnan(max_m_cur)):
                        expected_max = (max_m_cur - min_m_prev) + 2.0
                        expected_min = (min_m_cur - max_m_prev) - 2.0
                        
                        if expected_min <= delta_t <= expected_max:
                            agreement_t_c = True
                            break
                
                # Bridge check: compare records surrounding current record (handles outliers)
                if not agreement_t_c:
                    len_to_start = i
                    len_to_end = (len(grp) - 1) - i
                    bridge_len = min(len_to_start, len_to_end)
                    
                    for b in range(1, (bridge_len + 1)):
                        if i + b >= len(grp) or i - b < 0:
                            break
                        
                        before_rec = grp.iloc[i - b]
                        after_rec = grp.iloc[i + b]
                        
                        bridge_delta_t = (after_rec["visit_date"] - before_rec["visit_date"]).days / 30.0
                        
                        min_m_before = before_rec.get("min_month", np.nan)
                        max_m_before = before_rec.get("max_month", np.nan)
                        min_m_after = after_rec.get("min_month", np.nan)
                        max_m_after = after_rec.get("max_month", np.nan)
                        
                        if not (np.isnan(min_m_before) or np.isnan(max_m_before) or np.isnan(min_m_after) or np.isnan(max_m_after)):
                            bridge_expected_max = (max_m_after - min_m_before) + 2.0
                            bridge_expected_min = (min_m_after - max_m_before) - 2.0
                            
                            if bridge_expected_min <= bridge_delta_t <= bridge_expected_max:
                                agreement_t_c = True
                                break

                # Determine if this starts a new episode
                if (not agreement_t_c and delta_months > 2.0) or (delta_months > 10.0):
                    ep_rec = grp.iloc[indices]
                    episodes.append(
                        {
                            "person_id": person_id,
                            "episode": episode_num,
                            "episode_min_date": ep_rec["visit_date"].min(),
                            "episode_max_date": ep_rec["visit_date"].max(),
                            "num_concepts": len(indices),
                        }
                    )
                    episode_num += 1
                    indices = [i]
                else:
                    indices.append(i)

            if indices:
                ep_rec = grp.iloc[indices]
                episodes.append(
                    {
                        "person_id": person_id,
                        "episode": episode_num,
                        "episode_min_date": ep_rec["visit_date"].min(),
                        "episode_max_date": ep_rec["visit_date"].max(),
                        "num_concepts": len(indices),
                    }
                )

        if not episodes:
            return pd.DataFrame()

        eps = pd.DataFrame(episodes)
        
        # Filter episodes > 12 months
        eps["episode_length_months"] = (eps["episode_max_date"] - eps["episode_min_date"]).dt.days / 30.0
        eps = eps[eps["episode_length_months"] <= 12].copy()
        
        # Add episode frequency filter (lines 239-249 of original)
        # Calculate cohort time range
        cohort_max_date = eps["episode_max_date"].max()
        cohort_min_date = eps["episode_min_date"].min()
        cohort_period_years = (cohort_max_date - cohort_min_date).days / 365.0
        
        # Calculate episodes per person
        episode_counts = eps.groupby("person_id")["episode"].max().reset_index()
        episode_counts.columns = ["person_id", "max_episode_num"]
        episode_counts["episode_freq"] = episode_counts["max_episode_num"] / cohort_period_years
        
        # Keep only patients with < 5 episodes/year
        valid_patients = episode_counts[episode_counts["episode_freq"] < 5]["person_id"]
        eps = eps[eps["person_id"].isin(valid_patients)].copy()
        
        print(f"  After frequency filter: {len(eps)} episodes ({len(valid_patients)} patients)")
        
        eps["episode_max_date_plus_two_months"] = eps["episode_max_date"] + pd.to_timedelta(60, unit="D")

        eps = eps.sort_values(["person_id", "episode_min_date"])
        eps["episode"] = eps.groupby("person_id").cumcount() + 1

        return eps

        eps = eps.sort_values(["person_id", "episode_min_date"])
        eps["episode"] = eps.groupby("person_id").cumcount() + 1

        return eps

    # ---------------------------------------------------------------------
    # STEP 6: Merge HIP and PPS episodes
    # ---------------------------------------------------------------------
    def step6_merge_episodes(self) -> pd.DataFrame:
        """
        Merge HIP (algo1) and PPS (algo2) episodes by temporal overlap.
        HIP episodes provide outcomes; PPS episodes support progression.
        """
        print("\n" + "=" * 60)
        print("STEP 6: Merge HIP and PPS episodes")
        print("=" * 60)
        
        # Check if output already exists
        output_file = DATA_DIR / "step6_merged_episodes.csv"
        if output_file.exists():
            print(f"Loading cached result from {output_file.name}...")
            merged = pd.read_csv(output_file)
            # Convert date columns back to datetime
            for col in ["pregnancy_start", "pregnancy_end", "pps_start", "pps_end", 
                        "recorded_episode_start", "recorded_episode_end", "episode_max_date_plus_two_months"]:
                if col in merged.columns:
                    merged[col] = pd.to_datetime(merged[col])
            print(f"Loaded {len(merged)} merged episodes")
            self.merged_episodes = merged
            return merged

        hip = self.hip_episodes.copy() if self.hip_episodes is not None else pd.DataFrame()
        pps = self.pps_episodes.copy() if self.pps_episodes is not None else pd.DataFrame()

        if hip.empty and pps.empty:
            print("No episodes to merge.")
            self.merged_episodes = pd.DataFrame()
            return self.merged_episodes

        # Prepare HIP
        if not hip.empty:
            hip = hip.rename(columns={"estimated_start_date": "pregnancy_start", "visit_date": "pregnancy_end"})
            hip["algo1_id"] = hip["person_id"].astype(str) + "_" + hip["episode"].astype(str) + "_HIP"
            hip["HIP_flag"] = 1
            hip = hip.drop_duplicates(subset=["person_id", "episode"])

        # Prepare PPS
        if not pps.empty:
            pps = pps.rename(columns={"episode_min_date": "pps_start", "episode_max_date": "pps_end"})
            pps["algo2_id"] = pps["person_id"].astype(str) + "_" + pps["episode"].astype(str) + "_PPS"
            pps["PPS_flag"] = 1
            pps = pps.drop_duplicates(subset=["person_id", "episode"])

        merged_rows: List[Dict[str, object]] = []

        if not hip.empty and not pps.empty:
            hip_merged = set()
            pps_merged = set()
            persons = sorted(set(hip["person_id"]).union(pps["person_id"]))

            for pid in persons:
                hip_p = hip[hip["person_id"] == pid]
                pps_p = pps[pps["person_id"] == pid]

                for _, h in hip_p.iterrows():
                    h_start = h["pregnancy_start"]
                    h_end = h["pregnancy_end"]

                    best_pps = None
                    best_overlap = 0

                    for _, p in pps_p.iterrows():
                        p_start = p["pps_start"]
                        p_end_plus = p.get("episode_max_date_plus_two_months", p["pps_end"])

                        overlap_start = max(h_start, p_start)
                        overlap_end = min(h_end, p_end_plus)
                        if overlap_start <= overlap_end:
                            overlap_days = (overlap_end - overlap_start).days
                            if overlap_days > best_overlap:
                                best_overlap = overlap_days
                                best_pps = p

                    row = {
                        "person_id": pid,
                        "algo1_id": h["algo1_id"],
                        "pregnancy_start": h_start,
                        "pregnancy_end": h_end,
                        "category": h.get("category"),
                        "gest_date": h.get("gest_date"),
                        "HIP_flag": 1,
                    }

                    if best_pps is not None:
                        row.update(
                            {
                                "algo2_id": best_pps["algo2_id"],
                                "pps_start": best_pps["pps_start"],
                                "pps_end": best_pps["pps_end"],
                                "episode_max_date_plus_two_months": best_pps.get(
                                    "episode_max_date_plus_two_months", best_pps["pps_end"]
                                ),
                                "PPS_flag": 1,
                            }
                        )
                        pps_merged.add(best_pps["algo2_id"])
                    else:
                        row["PPS_flag"] = 0

                    merged_rows.append(row)
                    hip_merged.add(h["algo1_id"])

                # PPS-only episodes
                for _, p in pps_p.iterrows():
                    if p["algo2_id"] not in pps_merged:
                        merged_rows.append(
                            {
                                "person_id": pid,
                                "algo2_id": p["algo2_id"],
                                "pps_start": p["pps_start"],
                                "pps_end": p["pps_end"],
                                "episode_max_date_plus_two_months": p.get(
                                    "episode_max_date_plus_two_months", p["pps_end"]
                                ),
                                "HIP_flag": 0,
                                "PPS_flag": 1,
                            }
                        )
                        pps_merged.add(p["algo2_id"])
        elif not hip.empty:
            merged_rows = [
                {
                    "person_id": h["person_id"],
                    "algo1_id": h["algo1_id"],
                    "pregnancy_start": h["pregnancy_start"],
                    "pregnancy_end": h["pregnancy_end"],
                    "category": h.get("category"),
                    "gest_date": h.get("gest_date"),
                    "HIP_flag": 1,
                    "PPS_flag": 0,
                }
                for _, h in hip.iterrows()
            ]
        else:  # PPS-only
            merged_rows = [
                {
                    "person_id": p["person_id"],
                    "algo2_id": p["algo2_id"],
                    "pps_start": p["pps_start"],
                    "pps_end": p["pps_end"],
                    "episode_max_date_plus_two_months": p.get("episode_max_date_plus_two_months", p["pps_end"]),
                    "HIP_flag": 0,
                    "PPS_flag": 1,
                }
                for _, p in pps.iterrows()
            ]

        merged = pd.DataFrame(merged_rows)

        # Episode start/end in recorded data
        def _merged_start(row):
            dates = []
            for key in ["pregnancy_start", "pps_start", "gest_date"]:
                val = row.get(key)
                if pd.notna(val):
                    dates.append(val)
            return min(dates) if dates else pd.NaT

        def _merged_end(row):
            dates = []
            for key in ["pregnancy_end", "pps_end"]:
                val = row.get(key)
                if pd.notna(val):
                    dates.append(val)
            return max(dates) if dates else pd.NaT

        merged["recorded_episode_start"] = merged.apply(_merged_start, axis=1)
        merged["recorded_episode_end"] = merged.apply(_merged_end, axis=1)

        merged["HIP_flag"] = merged["HIP_flag"].fillna(0).astype(int)
        merged["PPS_flag"] = merged["PPS_flag"].fillna(0).astype(int)

        merged["HIP_outcome_category"] = merged.get("category")
        merged["PPS_outcome_category"] = None

        merged["recorded_episode_start"] = pd.to_datetime(merged["recorded_episode_start"])
        merged["recorded_episode_end"] = pd.to_datetime(merged["recorded_episode_end"])
        merged["recorded_episode_length"] = (
            merged["recorded_episode_end"] - merged["recorded_episode_start"]
        ).dt.days / 30.0

        merged = merged.drop_duplicates(subset=["person_id", "recorded_episode_start", "recorded_episode_end"])
        merged = merged[merged["recorded_episode_start"].notna()]

        merged = merged.sort_values(["person_id", "recorded_episode_start"])
        merged["episode_number"] = merged.groupby("person_id").cumcount() + 1

        print(f"  Merged episodes: {len(merged)}")
        print(f"  Persons with episodes: {merged['person_id'].nunique()}")

        self._save_csv(merged, "step6_merged_episodes.csv", "6")
        self.merged_episodes = merged
        return merged

    # ---------------------------------------------------------------------
    # STEP 7: Infer start date with ESD algorithm (simplified) and metadata
    # ---------------------------------------------------------------------
    def step7_add_metadata(self) -> pd.DataFrame:
        """
        Infer pregnancy start dates and add demographic metadata using
        local OMOP `person.csv` (and optional race/ethnicity concept names
        if present in `concept.csv`).
        """
        print("\n" + "=" * 60)
        print("STEP 7: Infer start dates and add metadata")
        print("=" * 60)
        
        # Check if output already exists
        output_file = DATA_DIR / "step7_pregnancy_cohort_final.csv"
        if output_file.exists():
            print(f"Loading cached result from {output_file.name}...")
            final = pd.read_csv(output_file)
            # Convert date columns back to datetime
            for col in ["inferred_episode_start", "inferred_episode_end", 
                        "recorded_episode_start", "recorded_episode_end", "date_of_birth"]:
                if col in final.columns:
                    final[col] = pd.to_datetime(final[col])
            print(f"Loaded {len(final)} final episodes")
            self.final_cohort = final
            return final

        episodes = self.merged_episodes.copy() if self.merged_episodes is not None else pd.DataFrame()
        if episodes.empty:
            print("No merged episodes to enrich.")
            self.final_cohort = pd.DataFrame()
            return self.final_cohort

        person = _read_omop("person")
        concept = _read_omop("concept")

        # Attach race / ethnicity names if possible
        race_map = (
            concept[["concept_id", "concept_name"]]
            .drop_duplicates()
            .set_index("concept_id")["concept_name"]
        )
        person["race_concept_name"] = person["race_concept_id"].map(race_map)
        person["ethnicity_concept_name"] = person["ethnicity_concept_id"].map(race_map)

        # One row per person
        person = person.drop_duplicates(subset=["person_id"])
        episodes = episodes.merge(person, on="person_id", how="left")

        # Date of birth
        episodes["day_of_birth"] = episodes.get("day_of_birth", 1).fillna(1).astype(int)
        episodes["month_of_birth"] = episodes.get("month_of_birth", 1).fillna(1).astype(int)
        episodes["date_of_birth"] = pd.to_datetime(
            episodes["year_of_birth"].astype(str)
            + "-"
            + episodes["month_of_birth"].astype(str)
            + "-"
            + episodes["day_of_birth"].astype(str),
            errors="coerce",
        )

        term_durations = self.matcho_term_durations.set_index("category").to_dict("index")

        def _infer_start(row):
            # Prefer HIP pregnancy_start if present
            if pd.notna(row.get("pregnancy_start")):
                return row["pregnancy_start"]
            end_date = row["recorded_episode_end"]
            if pd.isna(end_date):
                return pd.NaT
            cat = row.get("HIP_outcome_category") or "PREG"
            if cat in term_durations:
                max_term = int(term_durations[cat]["max_term"])
            else:
                max_term = 280
            return end_date - timedelta(days=max_term)

        episodes["inferred_episode_start"] = episodes.apply(_infer_start, axis=1)
        episodes["inferred_episode_end"] = episodes["recorded_episode_end"]

        episodes["inferred_episode_start"] = pd.to_datetime(episodes["inferred_episode_start"])
        episodes["gestational_age_days"] = (
            episodes["inferred_episode_end"] - episodes["inferred_episode_start"]
        ).dt.days

        # Assign final outcome category using the prioritization logic from original script
        # Priority order: LB > SB > ECT > AB (includes SA) > DELIV
        def _assign_final_outcome(row):
            hip_cat = row.get("HIP_outcome_category")
            pps_cat = row.get("PPS_outcome_category")
            
            # If both are null, default to PREG
            if pd.isna(hip_cat) and pd.isna(pps_cat):
                return "PREG"
            
            # If only one is available, use it
            if pd.isna(pps_cat):
                return hip_cat if pd.notna(hip_cat) else "PREG"
            if pd.isna(hip_cat):
                return pps_cat if pd.notna(pps_cat) else "PREG"
            
            # Both are available - apply priority order
            # Define priority: lower number = higher priority
            priority_map = {
                "LB": 1,
                "SB": 2,
                "ECT": 3,
                "AB": 4,
                "SA": 4,  # SA and AB have same priority
                "DELIV": 5,
                "PREG": 6
            }
            
            hip_priority = priority_map.get(hip_cat, 6)
            pps_priority = priority_map.get(pps_cat, 6)
            
            # Return the category with higher priority (lower number)
            if hip_priority <= pps_priority:
                return hip_cat
            else:
                return pps_cat
        
        episodes["final_outcome_category"] = episodes.apply(_assign_final_outcome, axis=1)

        # Age at episode start
        episodes["age_at_episode_start"] = (
            episodes["inferred_episode_start"] - episodes["date_of_birth"]
        ).dt.days / 365.25

        # Preterm status: < 37 weeks
        episodes["preterm_status"] = (episodes["gestational_age_days"] < 259).astype(int)

        # Precision category (simplified)
        episodes["precision_category"] = "non-specific"
        episodes.loc[episodes["HIP_flag"] == 1, "precision_category"] = "month"
        episodes.loc[
            (episodes["HIP_flag"] == 1) & (episodes["PPS_flag"] == 1),
            "precision_category",
        ] = "two-week"

        # Race standardization
        race_mapping = {
            "Asian": "Asian",
            "Asian Indian": "Asian",
            "Chinese": "Asian",
            "Korean": "Asian",
            "Vietnamese": "Asian",
            "Filipino": "Asian",
            "Japanese": "Asian",
            "Black": "Black",
            "Black or African American": "Black",
            "White": "White",
            "Native Hawaiian or Other Pacific Islander": "Native Hawaiian or Other Pacific Islander",
            "Other Pacific Islander": "Native Hawaiian or Other Pacific Islander",
            "Polynesian": "Native Hawaiian or Other Pacific Islander",
            "American Indian or Alaska Native": "American Indian or Alaska Native",
        }
        episodes["race"] = episodes["race_concept_name"].map(race_mapping).fillna("Unknown")

        # Ethnicity standardization
        def _std_eth(x: Optional[str]) -> str:
            if x == "Hispanic or Latino":
                return "Hispanic or Latino"
            if x == "Not Hispanic or Latino":
                return "Not Hispanic or Latino"
            return "Unknown"

        episodes["ethnicity"] = episodes["ethnicity_concept_name"].apply(_std_eth)

        # Combined race_ethnicity
        def _race_eth(row) -> str:
            if row["ethnicity"] == "Hispanic or Latino":
                return "Hispanic or Latino Any Race"
            if row["race"] == "Unknown":
                return "Unknown"
            return f"{row['race']} Non-Hispanic"

        episodes["race_ethnicity"] = episodes.apply(_race_eth, axis=1)

        final_cols = [
            "person_id",
            "episode_number",
            "inferred_episode_start",
            "inferred_episode_end",
            "recorded_episode_start",
            "recorded_episode_end",
            "final_outcome_category",
            "gestational_age_days",
            "HIP_flag",
            "PPS_flag",
            "precision_category",
            "year_of_birth",
            "date_of_birth",
            "age_at_episode_start",
            "race",
            "ethnicity",
            "race_ethnicity",
            "preterm_status",
            "location_id",
        ]
        final_cols = [c for c in final_cols if c in episodes.columns]
        final = episodes[final_cols].copy()

        print(f"  Final episodes: {len(final)}")
        print(f"  Unique persons: {final['person_id'].nunique()}")
        print(f"  Outcomes: {final['final_outcome_category'].value_counts().to_dict()}")
        print(f"  Preterm episodes: {int(final['preterm_status'].sum())}")

        self._save_csv(final, "step7_pregnancy_cohort_final.csv", "7")
        self.final_cohort = final
        return final

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run_full_pipeline(self) -> pd.DataFrame:
        """
        Run the entire pregnancy phenotyping pipeline on local CSV data.
        """
        print("\n" + "=" * 60)
        print("PREGNANCY PHENOTYPING PIPELINE (HIPPS) - CSV VERSION")
        print("=" * 60)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            self.step1_map_concepts()
            self.step2_get_specificity()
            self.step3_define_hip_episodes()
            self.step4_identify_gt_concepts()
            self.step5_define_pps_episodes()
            self.step6_merge_episodes()
            final = self.step7_add_metadata()
        except Exception as exc:
            print(f"\nERROR during pipeline: {exc}")
            raise
        else:
            print("\n" + "=" * 60)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"\nOutputs written under: {DATA_DIR}")
            return final


def main() -> None:
    pipeline = PregnancyPhenotypingCSV()
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()

