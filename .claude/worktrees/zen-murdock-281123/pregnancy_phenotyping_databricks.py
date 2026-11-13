"""
Pregnancy Phenotyping Algorithm (HIPPS) for Databricks / Unity Catalog
=======================================================================

This script implements the N3C pregnancy cohort phenotyping pipeline
(7-step HIPPS algorithm) adapted for Databricks with Unity Catalog.

Data source:
- OMOP clinical tables:  curated.data.<table>
- Concept reference CSVs: obgyn_analytics.dev.<table>  (uploaded as Delta tables)
- Output tables:          obgyn_analytics.dev.<table>

All intermediate and final outputs are persisted as Delta tables in
obgyn_analytics.dev and include a RunDate timestamp column.

Requirements:
- Active Databricks cluster with SparkSession
- Unity Catalog with curated.data and obgyn_analytics.dev schemas populated

Usage:
    Run as a Databricks notebook cell or %run from a notebook.
    Instantiate and call run_full_pipeline():

        pipeline = PregnancyPhenotypingDatabricks()
        pipeline.run_full_pipeline()
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Catalog / schema configuration
# ---------------------------------------------------------------------------
DATA_CATALOG_SCHEMA = "curated.data"          # OMOP clinical tables
CONCEPT_CATALOG_SCHEMA = "obgyn_analytics.dev"  # N3C concept reference tables (CSVs uploaded as Delta)
OUTPUT_CATALOG_SCHEMA = "obgyn_analytics.dev"   # All output tables written here


def _get_spark() -> SparkSession:
    return SparkSession.builder.getOrCreate()


@dataclass
class PregnancyPhenotypingDatabricks:
    """
    Pregnancy phenotyping pipeline using Databricks / Unity Catalog.
    """

    data_catalog_schema: str = DATA_CATALOG_SCHEMA

    def __post_init__(self) -> None:
        spark = _get_spark()

        print(f"Loading concept definition files from {CONCEPT_CATALOG_SCHEMA} ...")
        self.hip_concepts = spark.table(f"{CONCEPT_CATALOG_SCHEMA}.HIP_concepts").toPandas()
        self.matcho_concepts = spark.table(f"{CONCEPT_CATALOG_SCHEMA}.Matcho_concepts").toPandas()
        self.pps_concepts = spark.table(f"{CONCEPT_CATALOG_SCHEMA}.PPS_concepts").toPandas()
        self.matcho_outcome_limits = spark.table(f"{CONCEPT_CATALOG_SCHEMA}.Matcho_outcome_limits").toPandas()
        self.matcho_term_durations = spark.table(f"{CONCEPT_CATALOG_SCHEMA}.Matcho_term_durations").toPandas()

        try:
            self.highly_specific_concepts = spark.table(
                f"{CONCEPT_CATALOG_SCHEMA}.Highly_specific_concepts"
            ).toPandas()
            print("  - Loaded Highly_specific_concepts")
        except Exception:
            self.highly_specific_concepts = None
            print("  - Highly_specific_concepts not found; will use PPS concepts / defaults.")

        print(f"  - HIP concepts: {len(self.hip_concepts)}")
        print(f"  - Matcho concepts: {len(self.matcho_concepts)}")
        print(f"  - PPS concepts: {len(self.pps_concepts)}")

        # Placeholders populated by pipeline steps
        self.matcho_concepts_std: Optional[pd.DataFrame] = None
        self.hip_episodes: Optional[pd.DataFrame] = None
        self.pps_episodes: Optional[pd.DataFrame] = None
        self.merged_episodes: Optional[pd.DataFrame] = None
        self.final_cohort: Optional[pd.DataFrame] = None

    # -------------------------------------------------------------------------
    # Databricks helpers: SQL execution, table persistence, caching
    # -------------------------------------------------------------------------
    def _sql(self, query: str) -> pd.DataFrame:
        """Run a Spark SQL query and return results as a pandas DataFrame."""
        return _get_spark().sql(query).toPandas()

    def _table_exists(self, table_name: str) -> bool:
        """Check whether a table exists in the output catalog/schema."""
        return _get_spark().catalog.tableExists(f"{OUTPUT_CATALOG_SCHEMA}.{table_name}")

    def _save_table(self, df: pd.DataFrame, table_name: str, step: Optional[str] = None) -> None:
        """Persist a pandas DataFrame as a Delta table with a RunDate timestamp."""
        out = df.copy()
        out["RunDate"] = datetime.now()
        spark = _get_spark()
        spark.createDataFrame(out).write.mode("overwrite").saveAsTable(
            f"{OUTPUT_CATALOG_SCHEMA}.{table_name}"
        )
        msg = f"Saved table: {OUTPUT_CATALOG_SCHEMA}.{table_name} ({len(df)} rows)"
        if step:
            msg = f"[Step {step}] {msg}"
        print(msg)

    def _load_table(self, table_name: str) -> pd.DataFrame:
        """Load a cached output table, dropping the RunDate column."""
        df = _get_spark().table(f"{OUTPUT_CATALOG_SCHEMA}.{table_name}").toPandas()
        if "RunDate" in df.columns:
            df = df.drop(columns=["RunDate"])
        return df

    # -------------------------------------------------------------------------
    # STEP 1: Map external concepts to standard OMOP concepts
    # -------------------------------------------------------------------------
    def step1_map_concepts(self) -> pd.DataFrame:
        """
        Map Matcho pregnancy concepts to standard OMOP concepts using
        the concept and concept_relationship tables in curated.data.
        """
        print("\n" + "=" * 60)
        print("STEP 1: Map external concepts to standard OMOP concepts")
        print("=" * 60)

        if self._table_exists("step1_matcho_concepts_standardized"):
            print("Loading cached result from step1_matcho_concepts_standardized...")
            matcho_std = self._load_table("step1_matcho_concepts_standardized")
            print(f"Loaded {len(matcho_std)} standardized concepts")
            self.matcho_concepts_std = matcho_std
            return matcho_std

        concept_ids = self.matcho_concepts["concept_id"].tolist()
        concept_ids_str = ','.join(map(str, concept_ids))

        concept_query = f"""
            SELECT concept_id, concept_name, domain_id, vocabulary_id,
                   concept_class_id, standard_concept, concept_code
            FROM {self.data_catalog_schema}.concept
            WHERE concept_id IN ({concept_ids_str})
        """
        concept_df = self._sql(concept_query)

        concept_sub = concept_df.merge(
            self.matcho_concepts[["concept_id", "category"]],
            on="concept_id",
            how="inner",
        )

        print(f"Found {len(concept_sub)} Matcho concepts in database")

        standard = concept_sub[concept_sub["standard_concept"] == "S"].copy()
        non_standard = concept_sub[concept_sub["standard_concept"].isna()].copy()

        print(f"  - Standard concepts: {len(standard)}")
        print(f"  - Non-standard concepts: {len(non_standard)}")

        mapped = pd.DataFrame()
        if not non_standard.empty:
            non_std_ids = non_standard["concept_id"].tolist()
            non_std_ids_str = ','.join(map(str, non_std_ids))

            mapping_query = f"""
                SELECT cr.concept_id_1, cr.concept_id_2, cr.relationship_id,
                       c.concept_id, c.concept_name, c.domain_id, c.vocabulary_id,
                       c.concept_class_id, c.standard_concept, c.concept_code
                FROM {self.data_catalog_schema}.concept_relationship cr
                JOIN {self.data_catalog_schema}.concept c ON cr.concept_id_2 = c.concept_id
                WHERE cr.concept_id_1 IN ({non_std_ids_str})
                  AND cr.relationship_id = 'Maps to'
                  AND c.standard_concept = 'S'
            """
            mapped_df = self._sql(mapping_query)

            if not mapped_df.empty:
                mapped = mapped_df.merge(
                    non_standard[["concept_id", "category"]].rename(
                        columns={"concept_id": "concept_id_1"}
                    ),
                    on="concept_id_1",
                    how="inner",
                )
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

        self._save_table(matcho_std, "step1_matcho_concepts_standardized", "1")
        self.matcho_concepts_std = matcho_std
        return matcho_std

    # -------------------------------------------------------------------------
    # STEP 2: Get concept prevalence/specificity (use N3C-derived list)
    # -------------------------------------------------------------------------
    def step2_get_specificity(self) -> pd.DataFrame:
        """
        Use N3C-derived highly specific pregnancy concepts.
        """
        print("\n" + "=" * 60)
        print("STEP 2: Get concept prevalence and specificity")
        print("=" * 60)

        if self._table_exists("step2_highly_specific_concepts"):
            print("Loading cached result from step2_highly_specific_concepts...")
            hs = self._load_table("step2_highly_specific_concepts")
            print(f"Loaded {len(hs)} highly specific concepts")
            self.highly_specific_concepts = hs
            return hs

        if self.highly_specific_concepts is not None:
            hs = self.highly_specific_concepts.copy()
        else:
            print("Highly_specific_concepts not found; using PPS_concepts as proxy.")
            hs = self.pps_concepts[["domain_concept_id", "domain_concept_name"]].copy()
            hs = hs.rename(columns={"domain_concept_id": "concept_id", "domain_concept_name": "concept_name"})

        print(f"Using {len(hs)} highly specific concepts")
        self._save_table(hs, "step2_highly_specific_concepts", "2")
        self.highly_specific_concepts = hs
        return hs

    # -------------------------------------------------------------------------
    # STEP 3: Define pregnancy episodes with HIP algorithm
    # -------------------------------------------------------------------------
    def step3_define_hip_episodes(self) -> pd.DataFrame:
        """
        Define outcome-based pregnancy episodes using HIP concepts.
        """
        print("\n" + "=" * 60)
        print("STEP 3: Define pregnancy episodes with HIP algorithm")
        print("=" * 60)

        if self._table_exists("step3_hip_episodes"):
            print("Loading cached result from step3_hip_episodes...")
            episodes = self._load_table("step3_hip_episodes")
            for col in ["visit_date", "estimated_start_date"]:
                if col in episodes.columns:
                    episodes[col] = pd.to_datetime(episodes[col])
            print(f"Loaded {len(episodes)} HIP episodes")
            self.hip_episodes = episodes
            return episodes

        hip_ids = self.hip_concepts["concept_id"].unique().tolist()
        hip_ids_str = ','.join(map(str, hip_ids))
        hip_lookup: Dict[int, Dict[str, Optional[float]]] = (
            self.hip_concepts.drop_duplicates(subset=["concept_id"])
            .set_index("concept_id")[["category", "gest_value"]]
            .to_dict("index")
        )

        print("Querying women of reproductive age...")
        women_query = f"""
            SELECT person_id, year_of_birth, month_of_birth, day_of_birth
            FROM {self.data_catalog_schema}.person
            WHERE gender_concept_id = 8532
              AND year_of_birth >= 1965
              AND year_of_birth <= 2005
        """
        women = self._sql(women_query)
        print(f"Women of reproductive age: {len(women)}")

        women_ids_str = ','.join(map(str, women["person_id"].tolist()))

        print("Querying HIP concept records from clinical tables...")

        queries = {
            'condition': f"""
                SELECT person_id, condition_concept_id as concept_id,
                       condition_start_date as visit_date,
                       CAST(NULL AS DOUBLE) as value_as_number
                FROM {self.data_catalog_schema}.condition_occurrence
                WHERE condition_concept_id IN ({hip_ids_str})
                  AND person_id IN ({women_ids_str})
            """,
            'procedure': f"""
                SELECT person_id, procedure_concept_id as concept_id,
                       procedure_date as visit_date,
                       CAST(NULL AS DOUBLE) as value_as_number
                FROM {self.data_catalog_schema}.procedure_occurrence
                WHERE procedure_concept_id IN ({hip_ids_str})
                  AND person_id IN ({women_ids_str})
            """,
            'measurement': f"""
                SELECT person_id, measurement_concept_id as concept_id,
                       measurement_date as visit_date,
                       value_as_number
                FROM {self.data_catalog_schema}.measurement
                WHERE measurement_concept_id IN ({hip_ids_str})
                  AND person_id IN ({women_ids_str})
            """,
            'observation': f"""
                SELECT person_id, observation_concept_id as concept_id,
                       observation_date as visit_date,
                       value_as_number
                FROM {self.data_catalog_schema}.observation
                WHERE observation_concept_id IN ({hip_ids_str})
                  AND person_id IN ({women_ids_str})
            """
        }

        all_records = []
        for domain, query in queries.items():
            print(f"  Querying {domain}...")
            try:
                df = self._sql(query)
                df["domain"] = domain
                df["concept_name"] = None
                all_records.append(df)
                print(f"    Found {len(df)} records")
            except Exception as e:
                print(f"    Error: {e}")

        hip_records = pd.concat(all_records, ignore_index=True)
        hip_records["visit_date"] = pd.to_datetime(hip_records["visit_date"])
        hip_records = hip_records.dropna(subset=["visit_date"])

        print(f"\n  Total HIP-related records: {len(hip_records)}")

        hip_records["category"] = hip_records["concept_id"].map(lambda cid: hip_lookup.get(cid, {}).get("category"))
        hip_records["gest_value"] = hip_records["concept_id"].map(lambda cid: hip_lookup.get(cid, {}).get("gest_value"))

        hip_records = hip_records.sort_values(["person_id", "visit_date"])
        self._save_table(hip_records, "step3a_hip_records", "3")

        episodes = self._define_hip_episodes(hip_records)
        self._save_table(episodes, "step3_hip_episodes", "3")
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

        if "max_start_date" in final_df.columns:
            final_df = final_df.rename(columns={"max_start_date": "estimated_start_date"})

        output_cols = ["person_id", "visit_date", "category", "concept_id", "concept_name",
                      "gest_value", "episode", "estimated_start_date"]
        available_cols = [c for c in output_cols if c in final_df.columns]
        final_df = final_df[available_cols].copy()

        return final_df

    # =========================================================================
    # SCRIPT 3 HELPER METHODS: Full HIP Algorithm Implementation
    # =========================================================================

    def _get_outcome_visits(self, records_df: pd.DataFrame, categories) -> pd.DataFrame:
        """
        Get all visits for specific outcome category/categories.
        Filters to first visit and visits >= min_days apart (per Matcho limits).
        """
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
        same_type = lim[(lim["first_preg_category"] == category_key) &
                       (lim["outcome_preg_category"] == category_key)]
        if not same_type.empty:
            min_days = int(same_type.iloc[0]["min_days"])
        else:
            min_days = 168

        df["prev_visit"] = df.groupby("person_id")["visit_date"].shift(1)
        df["days_diff"] = (df["visit_date"] - df["prev_visit"]).dt.days

        df["is_new_episode"] = df["prev_visit"].isna() | (df["days_diff"] >= min_days)
        df = df[df["is_new_episode"]].copy()
        df = df.drop(columns=["prev_visit", "days_diff", "is_new_episode"])

        print(f"    {category_key}: {len(df)} episodes")
        return df

    def _add_stillbirth(self, lb_df: pd.DataFrame, sb_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add stillbirth visits to livebirth table following Script 3 logic (lines 389-474).
        Only adds SB records that don't conflict with LB based on Matcho outcome limits.
        """
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

        sb_after_lb = sb_temp[
            (sb_temp["previous_category"] == "LB") &
            (sb_temp["after_days"] >= before_min) &
            (sb_temp["next_category"].isna())
        ]

        sb_before_lb = sb_temp[
            (sb_temp["next_category"] == "LB") &
            (sb_temp["before_days"].abs() >= after_min) &
            (sb_temp["previous_category"].isna())
        ]

        sb_between_lb = sb_temp[
            (sb_temp["next_category"] == "LB") &
            (sb_temp["before_days"].abs() >= after_min) &
            (sb_temp["previous_category"] == "LB") &
            (sb_temp["after_days"] >= before_min)
        ]

        valid_sb = pd.concat([sb_null, sb_not_lb_prev, sb_not_lb_next, sb_not_lb_both,
                             sb_after_lb, sb_before_lb, sb_between_lb], ignore_index=True)
        valid_sb = valid_sb.drop(columns=["previous_category", "next_category", "before_days", "after_days"])

        final_df = pd.concat([lb_clean, valid_sb], ignore_index=True)
        final_df = final_df.drop_duplicates()

        print(f"    After adding SB: {len(final_df[final_df['category']=='LB'])} LB, {len(final_df[final_df['category']=='SB'])} SB")
        return final_df

    def _add_ectopic(self, prev_df: pd.DataFrame, ect_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ectopic pregnancy visits following Script 3 logic (lines 476-575).
        Only adds ECT records that don't conflict with LB or SB.
        """
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

        ect_after_lbsb = ect_temp[
            (ect_temp["previous_category"].isin(["LB", "SB"])) &
            (ect_temp["after_days"] >= before_min) &
            (ect_temp["next_category"].isna())
        ]

        ect_before_lb = ect_temp[
            (ect_temp["next_category"] == "LB") &
            (ect_temp["before_days"].abs() >= after_min_lb) &
            (ect_temp["previous_category"].isna())
        ]

        ect_before_sb = ect_temp[
            (ect_temp["next_category"] == "SB") &
            (ect_temp["before_days"].abs() >= after_min_sb) &
            (ect_temp["previous_category"].isna())
        ]

        ect_between_lb = ect_temp[
            (ect_temp["next_category"] == "LB") &
            (ect_temp["before_days"].abs() >= after_min_lb) &
            (ect_temp["previous_category"].isin(["LB", "SB"])) &
            (ect_temp["after_days"] >= before_min)
        ]

        ect_between_sb = ect_temp[
            (ect_temp["next_category"] == "SB") &
            (ect_temp["before_days"].abs() >= after_min_sb) &
            (ect_temp["previous_category"].isin(["LB", "SB"])) &
            (ect_temp["after_days"] >= before_min)
        ]

        valid_ect = pd.concat([ect_null, ect_not_lbsb_prev, ect_not_lbsb_next, ect_not_lbsb_both,
                              ect_after_lbsb, ect_before_lb, ect_before_sb,
                              ect_between_lb, ect_between_sb], ignore_index=True)
        valid_ect = valid_ect.drop(columns=["previous_category", "next_category", "before_days", "after_days"])

        final_df = pd.concat([prev_df, valid_ect], ignore_index=True)
        final_df = final_df.drop_duplicates()

        print(f"    After adding ECT: {len(final_df[final_df['category']=='ECT'])} ECT")
        return final_df

    def _add_abortion(self, prev_df: pd.DataFrame, ab_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add abortion visits (AB and SA) following Script 3 logic (lines 577-729).
        Only adds AB/SA records that don't conflict with LB, SB, or ECT.
        """
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

        ab_after_lbsb = ab_temp[
            (ab_temp["previous_category"].isin(["LB", "SB"])) &
            (ab_temp["after_days"] >= before_min_lb) &
            (ab_temp["next_category"].isna())
        ]

        ab_before_lb = ab_temp[(ab_temp["next_category"] == "LB") & (ab_temp["before_days"].abs() >= after_min_lb) & (ab_temp["previous_category"].isna())]
        ab_before_sb = ab_temp[(ab_temp["next_category"] == "SB") & (ab_temp["before_days"].abs() >= after_min_sb) & (ab_temp["previous_category"].isna())]

        ab_between_lb = ab_temp[(ab_temp["next_category"] == "LB") & (ab_temp["before_days"].abs() >= after_min_lb) & (ab_temp["previous_category"].isin(["LB", "SB"])) & (ab_temp["after_days"] >= before_min_lb)]
        ab_between_sb = ab_temp[(ab_temp["next_category"] == "SB") & (ab_temp["before_days"].abs() >= after_min_sb) & (ab_temp["previous_category"].isin(["LB", "SB"])) & (ab_temp["after_days"] >= before_min_lb)]

        ab_after_ect = ab_temp[(ab_temp["previous_category"] == "ECT") & (ab_temp["after_days"] >= before_min_ect) & (ab_temp["next_category"].isna())]
        ab_before_ect = ab_temp[(ab_temp["next_category"] == "ECT") & (ab_temp["before_days"].abs() >= after_min_ect) & (ab_temp["previous_category"].isna())]
        ab_between_ect = ab_temp[(ab_temp["next_category"] == "ECT") & (ab_temp["before_days"].abs() >= after_min_ect) & (ab_temp["previous_category"] == "ECT") & (ab_temp["after_days"] >= before_min_ect)]
        ab_between_ect_lbsb = ab_temp[(ab_temp["next_category"] == "ECT") & (ab_temp["before_days"].abs() >= after_min_ect) & (ab_temp["previous_category"].isin(["LB", "SB"])) & (ab_temp["after_days"] >= before_min_lb)]
        ab_between_lbsb_ect = ab_temp[(ab_temp["next_category"].isin(["LB", "SB"])) & (ab_temp["before_days"].abs() >= after_min_lb) & (ab_temp["previous_category"] == "ECT") & (ab_temp["after_days"] >= before_min_ect)]

        valid_ab = pd.concat([ab_null, ab_not_all_prev, ab_not_all_next, ab_not_all_both,
                             ab_after_lbsb, ab_before_lb, ab_before_sb, ab_between_lb, ab_between_sb,
                             ab_after_ect, ab_before_ect, ab_between_ect, ab_between_ect_lbsb, ab_between_lbsb_ect],
                            ignore_index=True)
        valid_ab = valid_ab.drop(columns=["temp_category", "previous_category", "next_category", "before_days", "after_days"])

        final_df = pd.concat([prev_df, valid_ab], ignore_index=True)
        final_df = final_df.drop_duplicates()

        print(f"    After adding AB/SA: {len(final_df[final_df['category'].isin(['AB','SA'])])} AB/SA")
        return final_df

    def _add_delivery(self, prev_df: pd.DataFrame, deliv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add delivery record only visits following Script 3 logic (lines 731-885).
        Only adds DELIV records that don't conflict with any more specific outcome.
        """
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

        valid_deliv = pd.concat([deliv_null, deliv_not_all_prev, deliv_not_all_next, deliv_not_all_both,
                                deliv_after_lbsb, deliv_before_lb, deliv_before_sb, deliv_between_lb, deliv_between_sb,
                                deliv_after_ect_ab, deliv_before_ect_ab, deliv_between_ect_ab,
                                deliv_between_ect_ab_lbsb, deliv_between_lbsb_ect_ab],
                               ignore_index=True)
        valid_deliv = valid_deliv.drop(columns=["temp_category", "previous_category", "next_category", "before_days", "after_days"])

        final_df = pd.concat([prev_df, valid_deliv], ignore_index=True)
        final_df = final_df.drop_duplicates()

        print(f"    After adding DELIV: {len(final_df[final_df['category']=='DELIV'])} DELIV")
        print(f"    TOTAL OUTCOMES: {len(final_df)} episodes")
        return final_df

    def _get_gestation_visits(self, records_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get all records with gestation period information.
        Includes records with gest_value or gestational age concepts.
        """
        df = records_df.copy()

        gest_df = df[df["gest_value"].notna()].copy()

        gest_concepts = [3002209, 3048230, 3012266]
        other_gest_df = df[
            (df["concept_id"].isin(gest_concepts)) &
            (df["value_as_number"].notna())
        ].copy()

        other_gest_df["gest_value"] = other_gest_df["value_as_number"].astype(int)

        all_gest_df = pd.concat([gest_df, other_gest_df], ignore_index=True)
        all_gest_df = all_gest_df.drop_duplicates()

        return all_gest_df

    def _define_gestation_episodes(self, gest_visits_df: pd.DataFrame) -> pd.DataFrame:
        """
        Define pregnancy episodes based on gestational records.
        Implements Script 3 logic (lines 907-970).
        """
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

        df["new_diff"] = df.apply(
            lambda row: 1 if (row["date_diff"] < min_days and row["week_diff"] <= 0) else row["week_diff"],
            axis=1
        )

        df["new_diff2"] = df.apply(
            lambda row: -1 if (row["date_diff"] >= row["day_diff"] and row["week_diff"] > 0) else row["new_diff"],
            axis=1
        )

        df["is_new_episode"] = (df["new_diff2"] <= 0) | (df["prev_week"].isna())
        df["episode"] = df.groupby("person_id")["is_new_episode"].cumsum()

        return df

    def _get_min_max_gestation(self, gest_episodes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get min/max gestational age and corresponding dates per episode.
        Implements Script 3 logic (lines 972-1076).
        """
        df = gest_episodes_df.copy()

        first_gest = df.groupby(["person_id", "episode"]).agg(
            first_gest_date=("visit_date", "min")
        ).reset_index()

        first_gest = first_gest.merge(df, on=["person_id", "episode"])
        first_gest = first_gest[first_gest["visit_date"] == first_gest["first_gest_date"]]
        first_gest = first_gest.groupby(["person_id", "episode"]).agg(
            first_gest_date=("first_gest_date", "first"),
            first_gest_week=("gest_value", "max")
        ).reset_index()

        min_gest = df.groupby(["person_id", "episode"]).agg(
            min_gest_week=("gest_value", "min")
        ).reset_index()
        min_gest = min_gest.merge(df, on=["person_id", "episode"])
        min_gest = min_gest[min_gest["gest_value"] == min_gest["min_gest_week"]]
        min_gest = min_gest.groupby(["person_id", "episode"]).agg(
            min_gest_week=("min_gest_week", "first"),
            min_gest_date=("visit_date", "min"),
            min_gest_date_2=("visit_date", "max")
        ).reset_index()

        end_gest = df.groupby(["person_id", "episode"]).agg(
            end_gest_date=("visit_date", "max")
        ).reset_index()

        end_gest = end_gest.merge(df, on=["person_id", "episode"])
        end_gest = end_gest[end_gest["visit_date"] == end_gest["end_gest_date"]]
        end_gest = end_gest.groupby(["person_id", "episode"]).agg(
            end_gest_date=("end_gest_date", "first"),
            end_gest_week=("gest_value", "max")
        ).reset_index()

        max_gest = df.groupby(["person_id", "episode"]).agg(
            max_gest_week=("gest_value", "max")
        ).reset_index()
        max_gest = max_gest.merge(df, on=["person_id", "episode"])
        max_gest = max_gest[max_gest["gest_value"] == max_gest["max_gest_week"]]
        max_gest = max_gest.groupby(["person_id", "episode"]).agg(
            max_gest_week=("max_gest_week", "first"),
            max_gest_date=("visit_date", "min")
        ).reset_index()

        result = first_gest.merge(end_gest, on=["person_id", "episode"])
        result = result.merge(min_gest, on=["person_id", "episode"])
        result = result.merge(max_gest, on=["person_id", "episode"])

        return result

    def _calculate_start_dates(self, outcomes_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate estimated start dates based on outcome type term duration."""
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

        df["min_start_date"] = df.apply(
            lambda row: pd.to_datetime(row["visit_date"]) - timedelta(days=int(term_durations.get(row["category"], {}).get("min_term", 140)))
            if row["category"] in term_durations else pd.NaT,
            axis=1
        )
        df["max_start_date"] = df.apply(_estimate_start, axis=1)

        return df

    def _merge_gestation_outcomes(self, outcomes_df: pd.DataFrame, gest_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge gestation and outcome-based episodes following Script 3 logic (lines 1096-1310).
        Creates PREG episodes from gestation-only records.
        """
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
        """
        Clean episodes and reclassify outcomes as PREG if inconsistent with gestational age.
        Implements Script 3 logic (lines 1312-1468).
        """
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
        """
        Remove overlapping episodes, keeping later episode if earlier is PREG.
        Implements Script 3 logic (lines 1470-1560).
        """
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

    # -------------------------------------------------------------------------
    # STEP 4: Identify gestational timing-specific concepts (PPS)
    # -------------------------------------------------------------------------
    def step4_identify_gt_concepts(self) -> pd.DataFrame:
        """Persist PPS concepts used for gestational timing."""
        print("\n" + "=" * 60)
        print("STEP 4: Identify gestational timing specific concepts")
        print("=" * 60)

        if self._table_exists("step4_pps_concepts"):
            print("Loading cached result from step4_pps_concepts...")
            pps = self._load_table("step4_pps_concepts")
            print(f"Loaded {len(pps)} PPS concepts")
            return pps

        pps = self.pps_concepts.copy()
        self._save_table(pps, "step4_pps_concepts", "4")
        return pps

    # -------------------------------------------------------------------------
    # STEP 5: Define pregnancy episodes with PPS algorithm
    # -------------------------------------------------------------------------
    def step5_define_pps_episodes(self) -> pd.DataFrame:
        """
        Define PPS episodes using gestational timing concepts.
        """
        print("\n" + "=" * 60)
        print("STEP 5: Define pregnancy episodes with PPS algorithm")
        print("=" * 60)

        if self._table_exists("step5_pps_episodes"):
            print("Loading cached result from step5_pps_episodes...")
            episodes = self._load_table("step5_pps_episodes")
            for col in ["episode_min_date", "episode_max_date", "episode_max_date_plus_two_months"]:
                if col in episodes.columns:
                    episodes[col] = pd.to_datetime(episodes[col])
            print(f"Loaded {len(episodes)} PPS episodes")
            self.pps_episodes = episodes
            return episodes

        print("Querying women of reproductive age...")
        women_query = f"""
            SELECT person_id
            FROM {self.data_catalog_schema}.person
            WHERE gender_concept_id = 8532
              AND year_of_birth >= 1965
              AND year_of_birth <= 2005
        """
        women = self._sql(women_query)
        print(f"Women of reproductive age: {len(women)}")

        women_ids_str = ','.join(map(str, women["person_id"].tolist()))

        pps_ids = self.pps_concepts["domain_concept_id"].tolist()
        pps_ids_str = ','.join(map(str, pps_ids))
        gt_lookup = self.pps_concepts.set_index("domain_concept_id")[["min_month", "max_month"]].to_dict("index")

        print("Querying PPS concept records from clinical tables...")

        queries = {
            'condition': f"""
                SELECT person_id, condition_concept_id as concept_id,
                       condition_start_date as visit_date
                FROM {self.data_catalog_schema}.condition_occurrence
                WHERE condition_concept_id IN ({pps_ids_str})
                  AND person_id IN ({women_ids_str})
            """,
            'procedure': f"""
                SELECT person_id, procedure_concept_id as concept_id,
                       procedure_date as visit_date
                FROM {self.data_catalog_schema}.procedure_occurrence
                WHERE procedure_concept_id IN ({pps_ids_str})
                  AND person_id IN ({women_ids_str})
            """,
            'measurement': f"""
                SELECT person_id, measurement_concept_id as concept_id,
                       measurement_date as visit_date
                FROM {self.data_catalog_schema}.measurement
                WHERE measurement_concept_id IN ({pps_ids_str})
                  AND person_id IN ({women_ids_str})
            """,
            'observation': f"""
                SELECT person_id, observation_concept_id as concept_id,
                       observation_date as visit_date
                FROM {self.data_catalog_schema}.observation
                WHERE observation_concept_id IN ({pps_ids_str})
                  AND person_id IN ({women_ids_str})
            """
        }

        all_records = []
        for domain, query in queries.items():
            print(f"  Querying {domain}...")
            try:
                df = self._sql(query)
                df["concept_name"] = None
                all_records.append(df)
                print(f"    Found {len(df)} records")
            except Exception as e:
                print(f"    Error: {e}")

        pps_records = pd.concat(all_records, ignore_index=True)
        pps_records["visit_date"] = pd.to_datetime(pps_records["visit_date"])
        pps_records = pps_records.dropna(subset=["visit_date"])

        print(f"\n  Total PPS-related records: {len(pps_records)}")

        pps_records["min_month"] = pps_records["concept_id"].map(lambda cid: gt_lookup.get(cid, {}).get("min_month"))
        pps_records["max_month"] = pps_records["concept_id"].map(lambda cid: gt_lookup.get(cid, {}).get("max_month"))

        pps_records = pps_records.sort_values(["person_id", "visit_date"])
        self._save_table(pps_records, "step5a_pps_records", "5")

        episodes = self._define_pps_episodes(pps_records)
        self._save_table(episodes, "step5_pps_episodes", "5")
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

                agreement_t_c = False

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

        eps["episode_length_months"] = (eps["episode_max_date"] - eps["episode_min_date"]).dt.days / 30.0
        eps = eps[eps["episode_length_months"] <= 12].copy()

        cohort_max_date = eps["episode_max_date"].max()
        cohort_min_date = eps["episode_min_date"].min()
        cohort_period_years = (cohort_max_date - cohort_min_date).days / 365.0

        episode_counts = eps.groupby("person_id")["episode"].max().reset_index()
        episode_counts.columns = ["person_id", "max_episode_num"]
        episode_counts["episode_freq"] = episode_counts["max_episode_num"] / cohort_period_years

        valid_patients = episode_counts[episode_counts["episode_freq"] < 5]["person_id"]
        eps = eps[eps["person_id"].isin(valid_patients)].copy()

        print(f"  After frequency filter: {len(eps)} episodes ({len(valid_patients)} patients)")

        eps["episode_max_date_plus_two_months"] = eps["episode_max_date"] + pd.to_timedelta(60, unit="D")

        eps = eps.sort_values(["person_id", "episode_min_date"])
        eps["episode"] = eps.groupby("person_id").cumcount() + 1

        return eps

    # -------------------------------------------------------------------------
    # STEP 6: Merge HIP and PPS episodes
    # -------------------------------------------------------------------------
    def _assign_pps_outcomes(self, pps_episodes: pd.DataFrame) -> pd.DataFrame:
        """
        Assign outcome categories to PPS episodes by searching for HIP outcome concepts
        within a lookback/lookahead window around each episode.

        This implements the logic from original script 6: outcomes_per_episode()
        """
        if pps_episodes.empty:
            return pps_episodes

        print("\n  Assigning outcomes to PPS episodes...")

        outcome_concepts = self.hip_concepts[
            self.hip_concepts["category"].isin(["LB", "SB", "ECT", "AB", "SA", "DELIV"])
        ].copy()
        outcome_concept_ids = outcome_concepts["concept_id"].unique().tolist()

        if not outcome_concept_ids:
            print("    No outcome concepts found, skipping PPS outcome assignment")
            pps_episodes["PPS_outcome_category"] = None
            return pps_episodes

        concept_to_category = outcome_concepts.set_index("concept_id")["category"].to_dict()

        pps_outcomes = []

        for _, episode in pps_episodes.iterrows():
            person_id = episode["person_id"]
            ep_max_date = episode["episode_max_date"]
            ep_max_date_plus = episode.get("episode_max_date_plus_two_months", ep_max_date)

            lookback_date = ep_max_date - timedelta(days=14)

            concept_ids_str = ','.join(map(str, outcome_concept_ids))

            outcome_query = f"""
                SELECT concept_id, concept_date
                FROM (
                    SELECT condition_concept_id as concept_id, condition_start_date as concept_date
                    FROM {self.data_catalog_schema}.condition_occurrence
                    WHERE person_id = {person_id}
                      AND condition_concept_id IN ({concept_ids_str})
                      AND condition_start_date >= '{lookback_date}'
                      AND condition_start_date <= '{ep_max_date_plus}'
                    UNION ALL
                    SELECT procedure_concept_id as concept_id, procedure_date as concept_date
                    FROM {self.data_catalog_schema}.procedure_occurrence
                    WHERE person_id = {person_id}
                      AND procedure_concept_id IN ({concept_ids_str})
                      AND procedure_date >= '{lookback_date}'
                      AND procedure_date <= '{ep_max_date_plus}'
                    UNION ALL
                    SELECT measurement_concept_id as concept_id, measurement_date as concept_date
                    FROM {self.data_catalog_schema}.measurement
                    WHERE person_id = {person_id}
                      AND measurement_concept_id IN ({concept_ids_str})
                      AND measurement_date >= '{lookback_date}'
                      AND measurement_date <= '{ep_max_date_plus}'
                    UNION ALL
                    SELECT observation_concept_id as concept_id, observation_date as concept_date
                    FROM {self.data_catalog_schema}.observation
                    WHERE person_id = {person_id}
                      AND observation_concept_id IN ({concept_ids_str})
                      AND observation_date >= '{lookback_date}'
                      AND observation_date <= '{ep_max_date_plus}'
                ) outcomes
                ORDER BY concept_date
            """

            try:
                outcomes_df = self._sql(outcome_query)

                if outcomes_df.empty:
                    pps_outcomes.append({"person_id": person_id, "episode": episode["episode"], "PPS_outcome_category": None})
                    continue

                outcomes_df["category"] = outcomes_df["concept_id"].map(concept_to_category)

                priority_order = ["LB", "SB", "ECT", "SA", "AB", "DELIV"]
                for priority_cat in priority_order:
                    if priority_cat in outcomes_df["category"].values:
                        pps_outcomes.append({
                            "person_id": person_id,
                            "episode": episode["episode"],
                            "PPS_outcome_category": priority_cat
                        })
                        break
                else:
                    pps_outcomes.append({"person_id": person_id, "episode": episode["episode"], "PPS_outcome_category": None})

            except Exception as e:
                print(f"    Error querying outcomes for person {person_id}: {e}")
                pps_outcomes.append({"person_id": person_id, "episode": episode["episode"], "PPS_outcome_category": None})

        outcomes_df = pd.DataFrame(pps_outcomes)
        pps_with_outcomes = pps_episodes.merge(
            outcomes_df,
            on=["person_id", "episode"],
            how="left"
        )

        outcome_counts = pps_with_outcomes["PPS_outcome_category"].value_counts()
        print(f"    PPS outcome distribution: {outcome_counts.to_dict()}")

        return pps_with_outcomes

    def step6_merge_episodes(self) -> pd.DataFrame:
        """Merge HIP (algo1) and PPS (algo2) episodes by temporal overlap."""
        print("\n" + "=" * 60)
        print("STEP 6: Merge HIP and PPS episodes")
        print("=" * 60)

        if self._table_exists("step6_merged_episodes"):
            print("Loading cached result from step6_merged_episodes...")
            merged = self._load_table("step6_merged_episodes")
            for col in ["pregnancy_start", "pregnancy_end", "pps_start", "pps_end",
                        "recorded_episode_start", "recorded_episode_end", "episode_max_date_plus_two_months"]:
                if col in merged.columns:
                    merged[col] = pd.to_datetime(merged[col])
            print(f"Loaded {len(merged)} merged episodes")
            self.merged_episodes = merged
            return merged

        hip = self.hip_episodes.copy() if self.hip_episodes is not None else pd.DataFrame()
        pps = self.pps_episodes.copy() if self.pps_episodes is not None else pd.DataFrame()

        if not pps.empty:
            pps = self._assign_pps_outcomes(pps)

        if hip.empty and pps.empty:
            print("No episodes to merge.")
            self.merged_episodes = pd.DataFrame()
            return self.merged_episodes

        if not hip.empty:
            hip = hip.rename(columns={"estimated_start_date": "pregnancy_start", "visit_date": "pregnancy_end"})
            hip["algo1_id"] = hip["person_id"].astype(str) + "_" + hip["episode"].astype(str) + "_HIP"
            hip["HIP_flag"] = 1
            hip = hip.drop_duplicates(subset=["person_id", "episode"])

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
                                "PPS_outcome_category": best_pps.get("PPS_outcome_category"),
                            }
                        )
                        pps_merged.add(best_pps["algo2_id"])
                    else:
                        row["PPS_flag"] = 0
                        row["PPS_outcome_category"] = None

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
                                "PPS_outcome_category": p.get("PPS_outcome_category"),
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
                    "PPS_outcome_category": None,
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
                    "PPS_outcome_category": p.get("PPS_outcome_category"),
                }
                for _, p in pps.iterrows()
            ]

        merged = pd.DataFrame(merged_rows)

        if merged.empty:
            print("No merged episodes produced.")
            self.merged_episodes = merged
            return merged

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

        self._save_table(merged, "step6_merged_episodes", "6")
        self.merged_episodes = merged
        return merged

    # -------------------------------------------------------------------------
    # STEP 7: Infer start date and add metadata
    # -------------------------------------------------------------------------
    def step7_add_metadata(self) -> pd.DataFrame:
        """Infer pregnancy start dates and add demographic metadata."""
        print("\n" + "=" * 60)
        print("STEP 7: Infer start dates and add metadata")
        print("=" * 60)

        if self._table_exists("step7_pregnancy_cohort_final"):
            print("Loading cached result from step7_pregnancy_cohort_final...")
            final = self._load_table("step7_pregnancy_cohort_final")
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

        print("Querying person demographics...")
        person_ids = episodes["person_id"].unique().tolist()
        person_ids_str = ','.join(map(str, person_ids))

        person_query = f"""
            SELECT p.person_id, p.year_of_birth, p.month_of_birth, p.day_of_birth,
                   p.gender_concept_id, p.race_concept_id, p.ethnicity_concept_id,
                   p.location_id,
                   r.concept_name as race_concept_name,
                   e.concept_name as ethnicity_concept_name
            FROM {self.data_catalog_schema}.person p
            LEFT JOIN {self.data_catalog_schema}.concept r ON p.race_concept_id = r.concept_id
            LEFT JOIN {self.data_catalog_schema}.concept e ON p.ethnicity_concept_id = e.concept_id
            WHERE p.person_id IN ({person_ids_str})
        """
        person_df = self._sql(person_query)

        person_df = person_df.drop_duplicates(subset=["person_id"])
        episodes = episodes.merge(person_df, on="person_id", how="left")

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

        def _assign_final_outcome(row):
            hip_cat = row.get("HIP_outcome_category")
            pps_cat = row.get("PPS_outcome_category")
            hip_end = row.get("pregnancy_end")
            pps_end = row.get("pps_end")

            outcome_match = False
            if pd.notna(hip_cat) and pd.notna(pps_cat):
                if hip_cat == pps_cat and hip_cat != "PREG":
                    if pd.notna(hip_end) and pd.notna(pps_end):
                        date_diff = abs((hip_end - pps_end).days)
                        if date_diff <= 14:
                            outcome_match = True
                elif hip_cat == "PREG" and pps_cat == "PREG":
                    outcome_match = True

            if outcome_match:
                return hip_cat

            if pd.isna(pps_cat):
                return hip_cat if pd.notna(hip_cat) else "PREG"

            if pd.isna(hip_cat):
                return pps_cat if pd.notna(pps_cat) else "PREG"

            if hip_cat != "PREG" and pps_cat != "PREG":
                if pd.notna(hip_end) and pd.notna(pps_end):
                    if hip_end <= (pps_end - timedelta(days=7)):
                        return pps_cat

            return hip_cat if pd.notna(hip_cat) else "PREG"

        episodes["final_outcome_category"] = episodes.apply(_assign_final_outcome, axis=1)

        episodes["age_at_episode_start"] = (
            episodes["inferred_episode_start"] - episodes["date_of_birth"]
        ).dt.days / 365.25

        episodes["preterm_status"] = (episodes["gestational_age_days"] < 259).astype(int)

        episodes["precision_category"] = "non-specific"
        episodes.loc[episodes["HIP_flag"] == 1, "precision_category"] = "month"
        episodes.loc[
            (episodes["HIP_flag"] == 1) & (episodes["PPS_flag"] == 1),
            "precision_category",
        ] = "two-week"

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

        def _std_eth(x: Optional[str]) -> str:
            if x == "Hispanic or Latino":
                return "Hispanic or Latino"
            if x == "Not Hispanic or Latino":
                return "Not Hispanic or Latino"
            return "Unknown"

        episodes["ethnicity"] = episodes["ethnicity_concept_name"].apply(_std_eth)

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

        self._save_table(final, "step7_pregnancy_cohort_final", "7")
        self.final_cohort = final
        return final

    # -------------------------------------------------------------------------
    # Pipeline orchestration
    # -------------------------------------------------------------------------
    def run_full_pipeline(self) -> pd.DataFrame:
        """Run the entire pregnancy phenotyping pipeline."""
        print("\n" + "=" * 60)
        print("PREGNANCY PHENOTYPING PIPELINE (HIPPS) - Databricks")
        print("=" * 60)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data source:    {self.data_catalog_schema}")
        print(f"Concept tables: {CONCEPT_CATALOG_SCHEMA}")
        print(f"Output tables:  {OUTPUT_CATALOG_SCHEMA}")

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

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nOutputs written to: {OUTPUT_CATALOG_SCHEMA}")
        return final


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
pipeline = PregnancyPhenotypingDatabricks()
pipeline.run_full_pipeline()
