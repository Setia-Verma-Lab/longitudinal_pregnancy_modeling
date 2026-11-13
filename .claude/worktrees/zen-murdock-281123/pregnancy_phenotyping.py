"""
Pregnancy Phenotyping Algorithm (HIPPS) - PostgreSQL Version
Adapted from N3C pregnancy cohort algorithm for OMOP database.

This module implements the Hierarchy and rule-based pregnancy episode Inference
integrated with Pregnancy Progression Signatures (HIPPS) algorithm.

This version is optimized for PostgreSQL databases with schema support.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings

from db_utils import get_connection, query_to_df

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path("pheno_data_sanity_checks")
CONCEPT_DIR = Path("n3c_pregnancy_cohort")


class PregnancyPhenotyping:
    """Main class for pregnancy phenotyping algorithm (PostgreSQL version)."""

    def __init__(self, config_path: str = "db_config.yaml", schema: str = "synpuf5"):
        """Initialize with database connection."""
        self.config_path = config_path
        self.schema = schema
        self.conn = None

        # Load concept files
        print("Loading concept definition files...")
        self.hip_concepts = pd.read_excel(CONCEPT_DIR / "HIP_concepts.xlsx")
        self.matcho_concepts = pd.read_excel(CONCEPT_DIR / "Matcho_concepts.xlsx")
        self.pps_concepts = pd.read_excel(CONCEPT_DIR / "PPS_concepts.xlsx")
        self.matcho_outcome_limits = pd.read_excel(CONCEPT_DIR / "Matcho_outcome_limits.xlsx")
        self.matcho_term_durations = pd.read_excel(CONCEPT_DIR / "Matcho_term_durations.xlsx")

        # Try to load highly specific concepts if available
        try:
            self.highly_specific_concepts = pd.read_excel(CONCEPT_DIR / "Highly_specific_concepts.xlsx")
            print("  - Loaded pre-computed highly specific concepts")
        except FileNotFoundError:
            self.highly_specific_concepts = None
            print("  - Highly specific concepts will be computed in Step 2")

        print(f"  - HIP concepts: {len(self.hip_concepts)} concepts")
        print(f"  - Matcho concepts: {len(self.matcho_concepts)} concepts")
        print(f"  - PPS concepts: {len(self.pps_concepts)} concepts")

    def connect(self):
        """Establish database connection."""
        print("\nConnecting to database...")
        self.conn = get_connection(self.config_path)
        print("Connected successfully.")

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("Database connection closed.")

    def _save_csv(self, df: pd.DataFrame, filename: str, step: str = None):
        """Save DataFrame to CSV in data directory."""
        filepath = DATA_DIR / filename
        df.to_csv(filepath, index=False)
        msg = f"Saved: {filename} ({len(df)} rows)"
        if step:
            msg = f"[Step {step}] {msg}"
        print(msg)

    # =========================================================================
    # STEP 1: Map external concepts to standard OMOP concepts
    # =========================================================================
    def step1_map_concepts(self) -> pd.DataFrame:
        """
        Map Matcho pregnancy concepts to standard OMOP concepts.
        Returns DataFrame with standardized concept_ids.
        """
        print("\n" + "=" * 60)
        print("STEP 1: Map external concepts to standard OMOP concepts")
        print("=" * 60)

        # Get concepts from our concept list that exist in the database
        concept_ids = self.matcho_concepts['concept_id'].tolist()
        concept_ids_str = ','.join(map(str, concept_ids))

        print("Querying OMOP concept table...")
        query = f"""
        SELECT concept_id, concept_name, domain_id, vocabulary_id,
               concept_class_id, standard_concept, concept_code
        FROM {self.schema}.concept
        WHERE concept_id IN ({concept_ids_str})
        """
        concepts_in_db = query_to_df(query, self.conn)

        # Join with our category information
        concepts_df = concepts_in_db.merge(
            self.matcho_concepts[['concept_id', 'category']],
            on='concept_id',
            how='inner'
        )

        print(f"Found {len(concepts_df)} concepts in database")

        # Separate standard and non-standard concepts
        standard_concepts = concepts_df[concepts_df['standard_concept'] == 'S'].copy()
        non_standard_concepts = concepts_df[concepts_df['standard_concept'].isna()].copy()

        print(f"  - Standard concepts: {len(standard_concepts)}")
        print(f"  - Non-standard concepts: {len(non_standard_concepts)}")

        # Map non-standard to standard using concept_relationship
        if len(non_standard_concepts) > 0:
            non_std_ids = non_standard_concepts['concept_id'].tolist()
            non_std_ids_str = ','.join(map(str, non_std_ids))

            print("Mapping non-standard concepts to standard...")
            mapping_query = f"""
            SELECT cr.concept_id_1, cr.concept_id_2, cr.relationship_id,
                   c.concept_id, c.concept_name, c.domain_id, c.vocabulary_id,
                   c.concept_class_id, c.standard_concept, c.concept_code
            FROM {self.schema}.concept_relationship cr
            JOIN {self.schema}.concept c ON cr.concept_id_2 = c.concept_id
            WHERE cr.concept_id_1 IN ({non_std_ids_str})
              AND cr.relationship_id = 'Maps to'
              AND c.standard_concept = 'S'
            """
            mapped = query_to_df(mapping_query, self.conn)

            if len(mapped) > 0:
                # Get category from original non-standard concepts
                mapped = mapped.merge(
                    non_standard_concepts[['concept_id', 'category']].rename(
                        columns={'concept_id': 'concept_id_1'}
                    ),
                    on='concept_id_1',
                    how='inner'
                )
                # Keep only the standard concept columns
                mapped = mapped[['concept_id', 'concept_name', 'domain_id', 'vocabulary_id',
                                 'concept_class_id', 'standard_concept', 'concept_code', 'category']]
                print(f"  - Mapped {len(mapped)} non-standard concepts to standard")
            else:
                mapped = pd.DataFrame()
        else:
            mapped = pd.DataFrame()

        # Combine standard concepts and mapped concepts
        if len(mapped) > 0:
            matcho_concepts_final = pd.concat([standard_concepts, mapped], ignore_index=True)
        else:
            matcho_concepts_final = standard_concepts.copy()

        # Remove duplicates
        matcho_concepts_final = matcho_concepts_final.drop_duplicates(subset=['concept_id', 'category'])

        print(f"\nFinal standardized concepts: {len(matcho_concepts_final)}")
        print(f"Categories: {matcho_concepts_final['category'].value_counts().to_dict()}")

        self._save_csv(matcho_concepts_final, "step1_matcho_concepts_standardized.csv", "1")
        self.matcho_concepts_std = matcho_concepts_final
        return matcho_concepts_final

    # =========================================================================
    # STEP 2: Get concept prevalence and specificity in pregnant women
    # =========================================================================
    def step2_get_specificity(self) -> pd.DataFrame:
        """
        Calculate concept prevalence and specificity to identify
        highly pregnancy-specific concepts.
        """
        print("\n" + "=" * 60)
        print("STEP 2: Get concept prevalence and specificity")
        print("=" * 60)

        if self.highly_specific_concepts is not None:
            print("Using pre-computed highly specific concepts.")
            self._save_csv(self.highly_specific_concepts, "step2_highly_specific_concepts.csv", "2")
            return self.highly_specific_concepts

        concept_ids = self.matcho_concepts_std['concept_id'].tolist()
        concept_ids_str = ','.join(map(str, concept_ids))

        # Step 2a: Find persons with pregnancy concepts
        print("\nFinding persons with pregnancy concepts...")

        # Query each domain table for persons with pregnancy concepts
        domain_queries = {
            'condition': f"""
                SELECT DISTINCT person_id
                FROM {self.schema}.condition_occurrence
                WHERE condition_concept_id IN ({concept_ids_str})
            """,
            'procedure': f"""
                SELECT DISTINCT person_id
                FROM {self.schema}.procedure_occurrence
                WHERE procedure_concept_id IN ({concept_ids_str})
            """,
            'measurement': f"""
                SELECT DISTINCT person_id
                FROM {self.schema}.measurement
                WHERE measurement_concept_id IN ({concept_ids_str})
            """,
            'observation': f"""
                SELECT DISTINCT person_id
                FROM {self.schema}.observation
                WHERE observation_concept_id IN ({concept_ids_str})
            """,
            'drug': f"""
                SELECT DISTINCT person_id
                FROM {self.schema}.drug_exposure
                WHERE drug_concept_id IN ({concept_ids_str})
            """
        }

        persons_with_concepts = []
        for domain, query in domain_queries.items():
            try:
                df = query_to_df(query, self.conn)
                persons_with_concepts.append(df)
                print(f"  - {domain}: {len(df)} persons")
            except Exception as e:
                print(f"  - {domain}: skipped ({e})")

        # Combine and deduplicate
        persons_with_preg = pd.concat(persons_with_concepts, ignore_index=True).drop_duplicates()
        print(f"\nTotal persons with pregnancy concepts: {len(persons_with_preg)}")

        # Step 2b: Get person demographics for pregnant women of reproductive age
        print("\nFiltering to women of reproductive age (15-55)...")
        person_ids_str = ','.join(map(str, persons_with_preg['person_id'].tolist()[:10000]))  # Limit for initial query

        # For large datasets, we need to query in batches
        person_query = f"""
            SELECT person_id, gender_concept_id, year_of_birth, month_of_birth, day_of_birth
            FROM {self.schema}.person
            WHERE gender_concept_id = 8532  -- FEMALE
              AND year_of_birth >= 1965
              AND year_of_birth <= 2005
        """
        person_df = query_to_df(person_query, self.conn)
        print(f"Women of reproductive age in database: {len(person_df)}")

        # Join to get pregnant women
        pregnant_women = persons_with_preg.merge(person_df, on='person_id', how='inner')
        print(f"Pregnant women of reproductive age: {len(pregnant_women)}")

        # Step 2c: Get all concepts for pregnant women and calculate prevalence
        print("\nCalculating concept prevalence in pregnant women...")

        # For efficiency, we'll use a simplified approach:
        # Query concept counts directly from the database
        preg_person_ids = pregnant_women['person_id'].tolist()

        # Create a temp approach - get top concepts by count
        count_queries = {
            'condition': f"""
                SELECT condition_concept_id as concept_id,
                       COUNT(DISTINCT person_id) as count
                FROM {self.schema}.condition_occurrence
                WHERE person_id IN (SELECT person_id FROM {self.schema}.person WHERE gender_concept_id = 8532)
                GROUP BY condition_concept_id
                HAVING COUNT(DISTINCT person_id) > 100
            """,
            'procedure': f"""
                SELECT procedure_concept_id as concept_id,
                       COUNT(DISTINCT person_id) as count
                FROM {self.schema}.procedure_occurrence
                WHERE person_id IN (SELECT person_id FROM {self.schema}.person WHERE gender_concept_id = 8532)
                GROUP BY procedure_concept_id
                HAVING COUNT(DISTINCT person_id) > 100
            """,
            'measurement': f"""
                SELECT measurement_concept_id as concept_id,
                       COUNT(DISTINCT person_id) as count
                FROM {self.schema}.measurement
                WHERE person_id IN (SELECT person_id FROM {self.schema}.person WHERE gender_concept_id = 8532)
                GROUP BY measurement_concept_id
                HAVING COUNT(DISTINCT person_id) > 100
            """,
            'observation': f"""
                SELECT observation_concept_id as concept_id,
                       COUNT(DISTINCT person_id) as count
                FROM {self.schema}.observation
                WHERE person_id IN (SELECT person_id FROM {self.schema}.person WHERE gender_concept_id = 8532)
                GROUP BY observation_concept_id
                HAVING COUNT(DISTINCT person_id) > 100
            """
        }

        # For now, use the pre-defined highly specific concepts from N3C
        # This is appropriate since the specificity calculation was done on a much larger dataset
        print("\nUsing N3C-derived highly specific concepts (ratio > 10)...")
        print("Note: Full specificity calculation requires large case/control populations")

        # Load the pre-computed highly specific concepts from N3C
        highly_specific = pd.read_excel(CONCEPT_DIR / "Highly_specific_concepts.xlsx")
        print(f"Loaded {len(highly_specific)} highly specific concepts")

        self._save_csv(highly_specific, "step2_highly_specific_concepts.csv", "2")
        self.highly_specific_concepts = highly_specific
        return highly_specific

    # =========================================================================
    # STEP 3: Define pregnancy episodes with HIP algorithm
    # =========================================================================
    def step3_define_hip_episodes(self) -> pd.DataFrame:
        """
        Define pregnancy episodes using the Hierarchy-based Inference of Pregnancy (HIP) algorithm.
        This is an extension of the Matcho et al. algorithm.
        """
        print("\n" + "=" * 60)
        print("STEP 3: Define pregnancy episodes with HIP algorithm")
        print("=" * 60)

        # Get HIP concepts
        hip_concept_ids = self.hip_concepts['concept_id'].unique().tolist()
        hip_concept_ids_str = ','.join(map(str, hip_concept_ids))

        # Create HIP concepts lookup with category and gest_value
        # Handle duplicate concept_ids by taking the first occurrence (priority category)
        hip_lookup = (self.hip_concepts.drop_duplicates(subset=['concept_id'])
                      .set_index('concept_id')[['category', 'gest_value']].to_dict('index'))

        print("\nQuerying pregnancy-related records from OMOP tables...")

        # Query each domain table
        queries = {
            'condition': f"""
                SELECT co.person_id, co.condition_concept_id as concept_id,
                       co.condition_start_date as visit_date,
                       c.concept_name
                FROM {self.schema}.condition_occurrence co
                JOIN {self.schema}.concept c ON co.condition_concept_id = c.concept_id
                WHERE co.condition_concept_id IN ({hip_concept_ids_str})
            """,
            'procedure': f"""
                SELECT po.person_id, po.procedure_concept_id as concept_id,
                       po.procedure_date as visit_date,
                       c.concept_name
                FROM {self.schema}.procedure_occurrence po
                JOIN {self.schema}.concept c ON po.procedure_concept_id = c.concept_id
                WHERE po.procedure_concept_id IN ({hip_concept_ids_str})
            """,
            'measurement': f"""
                SELECT m.person_id, m.measurement_concept_id as concept_id,
                       m.measurement_date as visit_date,
                       c.concept_name,
                       m.value_as_number
                FROM {self.schema}.measurement m
                JOIN {self.schema}.concept c ON m.measurement_concept_id = c.concept_id
                WHERE m.measurement_concept_id IN ({hip_concept_ids_str})
            """,
            'observation': f"""
                SELECT o.person_id, o.observation_concept_id as concept_id,
                       o.observation_date as visit_date,
                       c.concept_name,
                       o.value_as_number
                FROM {self.schema}.observation o
                JOIN {self.schema}.concept c ON o.observation_concept_id = c.concept_id
                WHERE o.observation_concept_id IN ({hip_concept_ids_str})
            """
        }

        all_records = []
        for domain, query in queries.items():
            print(f"  Querying {domain}...")
            try:
                df = query_to_df(query, self.conn)
                df['domain'] = domain
                # Add value_as_number if not present
                if 'value_as_number' not in df.columns:
                    df['value_as_number'] = None
                all_records.append(df)
                print(f"    Found {len(df)} records")
            except Exception as e:
                print(f"    Error: {e}")

        # Combine all records
        records_df = pd.concat(all_records, ignore_index=True)
        print(f"\nTotal pregnancy-related records: {len(records_df)}")

        # Add category and gest_value from HIP concepts
        records_df['category'] = records_df['concept_id'].map(lambda x: hip_lookup.get(x, {}).get('category'))
        records_df['gest_value'] = records_df['concept_id'].map(lambda x: hip_lookup.get(x, {}).get('gest_value'))

        # Filter to women of reproductive age
        print("\nFiltering to women of reproductive age...")
        person_query = f"""
            SELECT person_id, year_of_birth, month_of_birth, day_of_birth
            FROM {self.schema}.person
            WHERE gender_concept_id = 8532
              AND year_of_birth >= 1965
              AND year_of_birth <= 2005
        """
        women_df = query_to_df(person_query, self.conn)
        print(f"Women of reproductive age: {len(women_df)}")

        records_df = records_df.merge(women_df, on='person_id', how='inner')
        print(f"Records for women of reproductive age: {len(records_df)}")

        # Convert visit_date to datetime
        records_df['visit_date'] = pd.to_datetime(records_df['visit_date'])

        # Sort by person and date
        records_df = records_df.sort_values(['person_id', 'visit_date'])

        # Save intermediate result
        self._save_csv(records_df, "step3a_hip_records.csv", "3")

        # Define pregnancy episodes using HIP algorithm
        print("\nDefining pregnancy episodes using HIP algorithm...")
        hip_episodes = self._define_hip_episodes(records_df)

        print(f"\nHIP Episodes Summary:")
        print(f"  Total episodes: {len(hip_episodes)}")
        print(f"  Unique persons: {hip_episodes['person_id'].nunique()}")
        if 'category' in hip_episodes.columns:
            print(f"  By outcome: {hip_episodes['category'].value_counts().to_dict()}")

        self._save_csv(hip_episodes, "step3_hip_episodes.csv", "3")
        self.hip_episodes = hip_episodes
        return hip_episodes

    def _define_hip_episodes(self, records_df: pd.DataFrame) -> pd.DataFrame:
        """
        Internal method to define HIP episodes from pregnancy records.
        Implements the Matcho algorithm hierarchy.
        """
        # Outcome hierarchy: LB > SB > DELIV > ECT > AB > SA > PREG
        outcome_priority = {'LB': 1, 'SB': 2, 'DELIV': 3, 'ECT': 4, 'AB': 5, 'SA': 6, 'PREG': 7}

        # Process each outcome type separately, then merge
        outcome_types = ['LB', 'SB', 'ECT', 'AB', 'SA', 'DELIV', 'PREG']
        all_episodes = []

        for outcome in outcome_types:
            outcome_records = records_df[records_df['category'] == outcome].copy()
            if len(outcome_records) == 0:
                continue

            print(f"  Processing {outcome}: {len(outcome_records)} records")

            # Get unique visits per person
            outcome_records = outcome_records.drop_duplicates(subset=['person_id', 'visit_date'])

            # Sort by person and date
            outcome_records = outcome_records.sort_values(['person_id', 'visit_date'])

            # Calculate days between visits for same person
            outcome_records['prev_visit'] = outcome_records.groupby('person_id')['visit_date'].shift(1)
            outcome_records['days_diff'] = (outcome_records['visit_date'] - outcome_records['prev_visit']).dt.days

            # Get minimum days between outcomes from Matcho
            min_days_df = self.matcho_outcome_limits[
                (self.matcho_outcome_limits['first_preg_category'] == outcome) &
                (self.matcho_outcome_limits['outcome_preg_category'] == outcome)
            ]
            min_days = min_days_df['min_days'].iloc[0] if len(min_days_df) > 0 else 168

            # First visit or visits after minimum days are separate episodes
            outcome_records['is_new_episode'] = (
                outcome_records['prev_visit'].isna() |
                (outcome_records['days_diff'] >= min_days)
            )

            # Assign episode numbers within each person
            outcome_records['episode'] = outcome_records.groupby('person_id')['is_new_episode'].cumsum()

            # Keep only the episode end dates (visits marking new episodes)
            episode_visits = outcome_records[outcome_records['is_new_episode']].copy()
            episode_visits = episode_visits[['person_id', 'visit_date', 'category', 'concept_id',
                                             'concept_name', 'gest_value', 'episode']]

            all_episodes.append(episode_visits)

        if not all_episodes:
            return pd.DataFrame()

        # Combine all outcome episodes
        episodes_df = pd.concat(all_episodes, ignore_index=True)

        # Calculate estimated start date based on category-specific term durations
        term_durations = self.matcho_term_durations.set_index('category').to_dict('index')

        def estimate_start_date(row):
            category = row['category']
            visit_date = row['visit_date']
            if category in term_durations:
                max_term = term_durations[category]['max_term']
                return visit_date - timedelta(days=max_term)
            return visit_date - timedelta(days=280)  # Default 40 weeks

        episodes_df['estimated_start_date'] = episodes_df.apply(estimate_start_date, axis=1)

        # Get first gestational concept date for each episode
        preg_records = records_df[records_df['category'] == 'PREG'].copy()
        if len(preg_records) > 0:
            preg_first = preg_records.groupby('person_id')['visit_date'].min().reset_index()
            preg_first.columns = ['person_id', 'gest_date']
            episodes_df = episodes_df.merge(preg_first, on='person_id', how='left')
        else:
            episodes_df['gest_date'] = None

        # Renumber episodes per person
        episodes_df = episodes_df.sort_values(['person_id', 'visit_date'])
        episodes_df['episode'] = episodes_df.groupby('person_id').cumcount() + 1

        return episodes_df

    # =========================================================================
    # STEP 4: Identify gestational timing specific concepts for PPS
    # =========================================================================
    def step4_identify_gt_concepts(self) -> pd.DataFrame:
        """
        Identify gestational timing specific concepts using HIP episodes.
        """
        print("\n" + "=" * 60)
        print("STEP 4: Identify gestational timing specific concepts")
        print("=" * 60)

        # Use pre-computed PPS concepts from N3C
        # These were derived from analyzing gestational timing patterns
        print("Using N3C-derived PPS concepts with gestational timing information")
        print(f"PPS concepts: {len(self.pps_concepts)}")

        self._save_csv(self.pps_concepts, "step4_pps_concepts.csv", "4")
        return self.pps_concepts

    # =========================================================================
    # STEP 5: Define pregnancy episodes with PPS algorithm
    # =========================================================================
    def step5_define_pps_episodes(self) -> pd.DataFrame:
        """
        Define pregnancy episodes using Pregnancy Progression Signatures (PPS) algorithm.
        """
        print("\n" + "=" * 60)
        print("STEP 5: Define pregnancy episodes with PPS algorithm")
        print("=" * 60)

        pps_concept_ids = self.pps_concepts['domain_concept_id'].tolist()
        pps_concept_ids_str = ','.join(map(str, pps_concept_ids))

        # Create lookup for gestational timing
        gt_lookup = self.pps_concepts.set_index('domain_concept_id')[['min_month', 'max_month']].to_dict('index')

        print("\nQuerying PPS concept records from OMOP tables...")

        # Query each domain table for PPS concepts
        queries = {
            'condition': f"""
                SELECT co.person_id, co.condition_concept_id as concept_id,
                       co.condition_start_date as visit_date,
                       c.concept_name
                FROM {self.schema}.condition_occurrence co
                JOIN {self.schema}.concept c ON co.condition_concept_id = c.concept_id
                WHERE co.condition_concept_id IN ({pps_concept_ids_str})
            """,
            'procedure': f"""
                SELECT po.person_id, po.procedure_concept_id as concept_id,
                       po.procedure_date as visit_date,
                       c.concept_name
                FROM {self.schema}.procedure_occurrence po
                JOIN {self.schema}.concept c ON po.procedure_concept_id = c.concept_id
                WHERE po.procedure_concept_id IN ({pps_concept_ids_str})
            """,
            'measurement': f"""
                SELECT m.person_id, m.measurement_concept_id as concept_id,
                       m.measurement_date as visit_date,
                       c.concept_name
                FROM {self.schema}.measurement m
                JOIN {self.schema}.concept c ON m.measurement_concept_id = c.concept_id
                WHERE m.measurement_concept_id IN ({pps_concept_ids_str})
            """,
            'observation': f"""
                SELECT o.person_id, o.observation_concept_id as concept_id,
                       o.observation_date as visit_date,
                       c.concept_name
                FROM {self.schema}.observation o
                JOIN {self.schema}.concept c ON o.observation_concept_id = c.concept_id
                WHERE o.observation_concept_id IN ({pps_concept_ids_str})
            """
        }

        all_records = []
        for domain, query in queries.items():
            print(f"  Querying {domain}...")
            try:
                df = query_to_df(query, self.conn)
                df['domain'] = domain
                all_records.append(df)
                print(f"    Found {len(df)} records")
            except Exception as e:
                print(f"    Error: {e}")

        if not all_records:
            print("No PPS concept records found")
            return pd.DataFrame()

        # Combine all records
        records_df = pd.concat(all_records, ignore_index=True)
        print(f"\nTotal PPS concept records: {len(records_df)}")

        # Filter to women of reproductive age
        print("\nFiltering to women of reproductive age...")
        person_query = f"""
            SELECT person_id, year_of_birth, gender_concept_id
            FROM {self.schema}.person
            WHERE gender_concept_id = 8532
              AND year_of_birth >= 1965
              AND year_of_birth <= 2005
        """
        women_df = query_to_df(person_query, self.conn)

        records_df = records_df.merge(women_df[['person_id']], on='person_id', how='inner')
        records_df = records_df.drop_duplicates()
        print(f"Records for women of reproductive age: {len(records_df)}")

        # Convert visit_date and add gestational timing info
        records_df['visit_date'] = pd.to_datetime(records_df['visit_date'])
        records_df['min_month'] = records_df['concept_id'].map(lambda x: gt_lookup.get(x, {}).get('min_month'))
        records_df['max_month'] = records_df['concept_id'].map(lambda x: gt_lookup.get(x, {}).get('max_month'))

        # Sort by person and date
        records_df = records_df.sort_values(['person_id', 'visit_date'])

        # Save intermediate result
        self._save_csv(records_df, "step5a_pps_records.csv", "5")

        # Define PPS episodes
        print("\nDefining PPS episodes...")
        pps_episodes = self._define_pps_episodes(records_df)

        print(f"\nPPS Episodes Summary:")
        print(f"  Total episodes: {len(pps_episodes)}")
        print(f"  Unique persons: {pps_episodes['person_id'].nunique()}")

        self._save_csv(pps_episodes, "step5_pps_episodes.csv", "5")
        self.pps_episodes = pps_episodes
        return pps_episodes

    def _define_pps_episodes(self, records_df: pd.DataFrame) -> pd.DataFrame:
        """
        Define PPS episodes using temporal sequence analysis.
        Checks if concept timing is consistent with a progressing pregnancy.
        """
        episodes = []

        # Process each person
        for person_id, person_records in records_df.groupby('person_id'):
            person_records = person_records.sort_values('visit_date').reset_index(drop=True)

            if len(person_records) < 1:
                continue

            # Initialize first episode
            episode_num = 1
            episode_start = person_records.iloc[0]['visit_date']
            episode_records = [0]

            for i in range(1, len(person_records)):
                current = person_records.iloc[i]
                prev = person_records.iloc[i - 1]

                # Calculate actual time difference in months
                delta_months = (current['visit_date'] - prev['visit_date']).days / 30.0

                # Check if timing is consistent with pregnancy progression
                # Using the gestational timing ranges from PPS concepts
                expected_max_delta = (current['max_month'] - prev['min_month']) + 2  # Allow 2 month buffer
                expected_min_delta = (current['min_month'] - prev['max_month']) - 2

                is_consistent = (expected_min_delta <= delta_months <= expected_max_delta)

                # Start new episode if not consistent and gap > 2 months
                if not is_consistent and delta_months > 2:
                    # Save current episode
                    ep_records = person_records.iloc[episode_records]
                    episodes.append({
                        'person_id': person_id,
                        'episode': episode_num,
                        'episode_min_date': ep_records['visit_date'].min(),
                        'episode_max_date': ep_records['visit_date'].max(),
                        'num_concepts': len(episode_records)
                    })

                    # Start new episode
                    episode_num += 1
                    episode_records = [i]
                elif delta_months > 10:  # Force new episode if gap > 10 months
                    # Save current episode
                    ep_records = person_records.iloc[episode_records]
                    episodes.append({
                        'person_id': person_id,
                        'episode': episode_num,
                        'episode_min_date': ep_records['visit_date'].min(),
                        'episode_max_date': ep_records['visit_date'].max(),
                        'num_concepts': len(episode_records)
                    })

                    # Start new episode
                    episode_num += 1
                    episode_records = [i]
                else:
                    episode_records.append(i)

            # Save last episode
            if episode_records:
                ep_records = person_records.iloc[episode_records]
                episodes.append({
                    'person_id': person_id,
                    'episode': episode_num,
                    'episode_min_date': ep_records['visit_date'].min(),
                    'episode_max_date': ep_records['visit_date'].max(),
                    'num_concepts': len(episode_records)
                })

        if not episodes:
            return pd.DataFrame()

        pps_df = pd.DataFrame(episodes)

        # Filter episodes longer than 12 months (invalid)
        pps_df['episode_length_months'] = (pps_df['episode_max_date'] - pps_df['episode_min_date']).dt.days / 30.0
        pps_df = pps_df[pps_df['episode_length_months'] <= 12]

        # Add episode_max_date_plus_two_months for merging
        pps_df['episode_max_date_plus_two_months'] = pps_df['episode_max_date'] + timedelta(days=60)

        # Renumber episodes per person
        pps_df = pps_df.sort_values(['person_id', 'episode_min_date'])
        pps_df['episode'] = pps_df.groupby('person_id').cumcount() + 1

        return pps_df

    # =========================================================================
    # STEP 6: Merge pregnancy episodes
    # =========================================================================
    def step6_merge_episodes(self) -> pd.DataFrame:
        """
        Merge HIP and PPS episodes based on temporal overlap.
        """
        print("\n" + "=" * 60)
        print("STEP 6: Merge HIP and PPS pregnancy episodes")
        print("=" * 60)

        hip_df = self.hip_episodes.copy()
        pps_df = self.pps_episodes.copy()

        print(f"HIP episodes: {len(hip_df)}")
        print(f"PPS episodes: {len(pps_df)}")

        if len(hip_df) == 0 and len(pps_df) == 0:
            print("No episodes to merge")
            return pd.DataFrame()

        # Prepare HIP episodes
        hip_df = hip_df.rename(columns={
            'estimated_start_date': 'pregnancy_start',
            'visit_date': 'pregnancy_end'
        })
        hip_df['algo1_id'] = hip_df['person_id'].astype(str) + '_' + hip_df['episode'].astype(str) + '_HIP'
        hip_df['HIP_flag'] = 1
        hip_df = hip_df.drop_duplicates(subset=['person_id', 'episode'])

        # Prepare PPS episodes
        pps_df = pps_df.rename(columns={
            'episode_min_date': 'pps_start',
            'episode_max_date': 'pps_end'
        })
        pps_df['algo2_id'] = pps_df['person_id'].astype(str) + '_' + pps_df['episode'].astype(str) + '_PPS'
        pps_df['PPS_flag'] = 1
        pps_df = pps_df.drop_duplicates(subset=['person_id', 'episode'])

        # Merge based on overlap using a better approach
        if len(hip_df) > 0 and len(pps_df) > 0:
            print("\nMerging episodes based on temporal overlap...")

            merged_rows = []

            # Track which episodes have been merged
            hip_merged = set()
            pps_merged = set()

            # For each person, check for overlapping episodes
            all_persons = set(hip_df['person_id'].tolist()) | set(pps_df['person_id'].tolist())

            for person_id in all_persons:
                person_hip = hip_df[hip_df['person_id'] == person_id]
                person_pps = pps_df[pps_df['person_id'] == person_id]

                for _, hip_row in person_hip.iterrows():
                    hip_start = hip_row['pregnancy_start']
                    hip_end = hip_row['pregnancy_end']

                    # Find overlapping PPS episodes
                    best_pps = None
                    best_overlap = 0

                    for _, pps_row in person_pps.iterrows():
                        pps_start = pps_row['pps_start']
                        pps_end_plus = pps_row['episode_max_date_plus_two_months']

                        # Check overlap
                        overlap_start = max(hip_start, pps_start)
                        overlap_end = min(hip_end, pps_end_plus)

                        if overlap_start <= overlap_end:
                            overlap_days = (overlap_end - overlap_start).days
                            if overlap_days > best_overlap:
                                best_overlap = overlap_days
                                best_pps = pps_row

                    # Create merged row
                    merged_row = {
                        'person_id': person_id,
                        'algo1_id': hip_row['algo1_id'],
                        'pregnancy_start': hip_start,
                        'pregnancy_end': hip_end,
                        'category': hip_row.get('category'),
                        'gest_date': hip_row.get('gest_date'),
                        'HIP_flag': 1
                    }

                    if best_pps is not None:
                        merged_row['algo2_id'] = best_pps['algo2_id']
                        merged_row['pps_start'] = best_pps['pps_start']
                        merged_row['pps_end'] = best_pps['pps_end']
                        merged_row['episode_max_date_plus_two_months'] = best_pps['episode_max_date_plus_two_months']
                        merged_row['PPS_flag'] = 1
                        pps_merged.add(best_pps['algo2_id'])
                    else:
                        merged_row['PPS_flag'] = 0

                    merged_rows.append(merged_row)
                    hip_merged.add(hip_row['algo1_id'])

                # Add PPS-only episodes for this person
                for _, pps_row in person_pps.iterrows():
                    if pps_row['algo2_id'] not in pps_merged:
                        merged_rows.append({
                            'person_id': person_id,
                            'algo2_id': pps_row['algo2_id'],
                            'pps_start': pps_row['pps_start'],
                            'pps_end': pps_row['pps_end'],
                            'episode_max_date_plus_two_months': pps_row['episode_max_date_plus_two_months'],
                            'HIP_flag': 0,
                            'PPS_flag': 1
                        })
                        pps_merged.add(pps_row['algo2_id'])

            merged = pd.DataFrame(merged_rows)

        elif len(hip_df) > 0:
            merged = hip_df.copy()
            merged['PPS_flag'] = 0
        else:
            merged = pps_df.copy()
            merged['HIP_flag'] = 0

        # Calculate merged episode dates
        print("Calculating merged episode dates...")

        def get_merged_start(row):
            dates = []
            if pd.notna(row.get('pregnancy_start')):
                dates.append(row['pregnancy_start'])
            if pd.notna(row.get('pps_start')):
                dates.append(row['pps_start'])
            if pd.notna(row.get('gest_date')):
                dates.append(row['gest_date'])
            return min(dates) if dates else None

        def get_merged_end(row):
            dates = []
            if pd.notna(row.get('pregnancy_end')):
                dates.append(row['pregnancy_end'])
            if pd.notna(row.get('pps_end')):
                dates.append(row['pps_end'])
            return max(dates) if dates else None

        merged['recorded_episode_start'] = merged.apply(get_merged_start, axis=1)
        merged['recorded_episode_end'] = merged.apply(get_merged_end, axis=1)

        # Fill missing flags
        merged['HIP_flag'] = merged['HIP_flag'].fillna(0).astype(int)
        merged['PPS_flag'] = merged['PPS_flag'].fillna(0).astype(int)

        # Get final outcome category (HIP takes priority)
        if 'category' in merged.columns:
            merged['HIP_outcome_category'] = merged['category']
        else:
            merged['HIP_outcome_category'] = None
        merged['PPS_outcome_category'] = None  # PPS doesn't have outcomes in step 5

        # Calculate episode length
        merged['recorded_episode_start'] = pd.to_datetime(merged['recorded_episode_start'])
        merged['recorded_episode_end'] = pd.to_datetime(merged['recorded_episode_end'])
        merged['recorded_episode_length'] = (merged['recorded_episode_end'] - merged['recorded_episode_start']).dt.days / 30.0

        # Remove duplicates and invalid episodes
        merged = merged.drop_duplicates(subset=['person_id', 'recorded_episode_start', 'recorded_episode_end'])
        merged = merged[merged['recorded_episode_start'].notna()]

        # Renumber episodes
        merged = merged.sort_values(['person_id', 'recorded_episode_start'])
        merged['episode_number'] = merged.groupby('person_id').cumcount() + 1

        print(f"\nMerged Episodes Summary:")
        print(f"  Total episodes: {len(merged)}")
        print(f"  Unique persons: {merged['person_id'].nunique()}")
        print(f"  HIP-only: {len(merged[(merged['HIP_flag'] == 1) & (merged['PPS_flag'] == 0)])}")
        print(f"  PPS-only: {len(merged[(merged['HIP_flag'] == 0) & (merged['PPS_flag'] == 1)])}")
        print(f"  Both: {len(merged[(merged['HIP_flag'] == 1) & (merged['PPS_flag'] == 1)])}")

        self._save_csv(merged, "step6_merged_episodes.csv", "6")
        self.merged_episodes = merged
        return merged

    # =========================================================================
    # STEP 7: Infer start date with ESD algorithm and add metadata
    # =========================================================================
    def step7_add_metadata(self) -> pd.DataFrame:
        """
        Infer pregnancy start dates and add demographic metadata.
        """
        print("\n" + "=" * 60)
        print("STEP 7: Infer start dates and add metadata")
        print("=" * 60)

        episodes = self.merged_episodes.copy()

        # Get person demographics
        print("Querying person demographics...")
        person_query = f"""
            SELECT p.person_id, p.year_of_birth, p.month_of_birth, p.day_of_birth,
                   p.gender_concept_id, p.race_concept_id, p.ethnicity_concept_id,
                   p.location_id,
                   r.concept_name as race_concept_name,
                   e.concept_name as ethnicity_concept_name
            FROM {self.schema}.person p
            LEFT JOIN {self.schema}.concept r ON p.race_concept_id = r.concept_id
            LEFT JOIN {self.schema}.concept e ON p.ethnicity_concept_id = e.concept_id
        """
        person_df = query_to_df(person_query, self.conn)

        # Merge demographics - ensure no duplicate person records
        person_df = person_df.drop_duplicates(subset=['person_id'])
        episodes = episodes.merge(person_df, on='person_id', how='left')

        # Calculate date of birth
        episodes['day_of_birth'] = episodes['day_of_birth'].fillna(1).astype(int)
        episodes['month_of_birth'] = episodes['month_of_birth'].fillna(1).astype(int)
        episodes['date_of_birth'] = pd.to_datetime(
            episodes['year_of_birth'].astype(str) + '-' +
            episodes['month_of_birth'].astype(str) + '-' +
            episodes['day_of_birth'].astype(str),
            errors='coerce'
        )

        # Infer episode start date
        print("Inferring episode start dates...")

        term_durations = self.matcho_term_durations.set_index('category').to_dict('index')

        def infer_start_date(row):
            # Use recorded start if available
            if pd.notna(row.get('pregnancy_start')):
                return row['pregnancy_start']

            # Otherwise estimate from outcome and term duration
            end_date = row['recorded_episode_end']
            category = row.get('HIP_outcome_category', 'PREG')

            if pd.isna(end_date):
                return None

            if category in term_durations:
                max_term = term_durations[category]['max_term']
            else:
                max_term = 280  # Default 40 weeks

            return end_date - timedelta(days=max_term)

        episodes['inferred_episode_start'] = episodes.apply(infer_start_date, axis=1)
        episodes['inferred_episode_end'] = episodes['recorded_episode_end']

        # Calculate final outcome category
        episodes['final_outcome_category'] = episodes['HIP_outcome_category'].fillna('PREG')

        # Calculate gestational age
        episodes['inferred_episode_start'] = pd.to_datetime(episodes['inferred_episode_start'])
        episodes['gestational_age_days'] = (episodes['inferred_episode_end'] - episodes['inferred_episode_start']).dt.days

        # Calculate age at episode start
        episodes['age_at_episode_start'] = (episodes['inferred_episode_start'] - episodes['date_of_birth']).dt.days / 365.25

        # Determine preterm status
        episodes['preterm_status'] = (episodes['gestational_age_days'] < 259).astype(int)  # < 37 weeks

        # Add precision category (simplified - would need gestational timing concepts for full implementation)
        episodes['precision_category'] = 'non-specific'  # Default
        episodes.loc[episodes['HIP_flag'] == 1, 'precision_category'] = 'month'
        episodes.loc[(episodes['HIP_flag'] == 1) & (episodes['PPS_flag'] == 1), 'precision_category'] = 'two-week'

        # Standardize race categories
        race_mapping = {
            'Asian': 'Asian',
            'Asian Indian': 'Asian',
            'Chinese': 'Asian',
            'Korean': 'Asian',
            'Vietnamese': 'Asian',
            'Filipino': 'Asian',
            'Japanese': 'Asian',
            'Black': 'Black',
            'Black or African American': 'Black',
            'White': 'White',
            'Native Hawaiian or Other Pacific Islander': 'Native Hawaiian or Other Pacific Islander',
            'Other Pacific Islander': 'Native Hawaiian or Other Pacific Islander',
            'Polynesian': 'Native Hawaiian or Other Pacific Islander',
            'American Indian or Alaska Native': 'American Indian or Alaska Native',
        }
        episodes['race'] = episodes['race_concept_name'].map(race_mapping).fillna('Unknown')

        # Standardize ethnicity
        episodes['ethnicity'] = episodes['ethnicity_concept_name'].apply(
            lambda x: 'Hispanic or Latino' if x == 'Hispanic or Latino'
            else ('Not Hispanic or Latino' if x == 'Not Hispanic or Latino' else 'Unknown')
        )

        # Create race_ethnicity combined field
        episodes['race_ethnicity'] = episodes.apply(
            lambda row: 'Hispanic or Latino Any Race' if row['ethnicity'] == 'Hispanic or Latino'
            else (row['race'] + ' Non-Hispanic' if row['race'] != 'Unknown' else 'Unknown'),
            axis=1
        )

        # Select final columns
        final_columns = [
            'person_id', 'episode_number',
            'inferred_episode_start', 'inferred_episode_end',
            'recorded_episode_start', 'recorded_episode_end',
            'final_outcome_category', 'gestational_age_days',
            'HIP_flag', 'PPS_flag', 'precision_category',
            'year_of_birth', 'date_of_birth', 'age_at_episode_start',
            'race', 'ethnicity', 'race_ethnicity',
            'preterm_status', 'location_id'
        ]
        final_columns = [c for c in final_columns if c in episodes.columns]
        final_episodes = episodes[final_columns].copy()

        print(f"\nFinal Pregnancy Cohort Summary:")
        print(f"  Total episodes: {len(final_episodes)}")
        print(f"  Unique persons: {final_episodes['person_id'].nunique()}")
        print(f"  Outcome categories: {final_episodes['final_outcome_category'].value_counts().to_dict()}")
        print(f"  Preterm births: {final_episodes['preterm_status'].sum()}")

        self._save_csv(final_episodes, "step7_pregnancy_cohort_final.csv", "7")
        self.final_cohort = final_episodes
        return final_episodes

    # =========================================================================
    # Main execution
    # =========================================================================
    def run_full_pipeline(self):
        """Run the complete pregnancy phenotyping pipeline."""
        print("\n" + "=" * 60)
        print("PREGNANCY PHENOTYPING PIPELINE (HIPPS)")
        print("=" * 60)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            self.connect()

            # Step 1: Map concepts
            self.step1_map_concepts()

            # Step 2: Get specificity
            self.step2_get_specificity()

            # Step 3: Define HIP episodes
            self.step3_define_hip_episodes()

            # Step 4: Identify GT concepts
            self.step4_identify_gt_concepts()

            # Step 5: Define PPS episodes
            self.step5_define_pps_episodes()

            # Step 6: Merge episodes
            self.step6_merge_episodes()

            # Step 7: Add metadata
            self.step7_add_metadata()

            print("\n" + "=" * 60)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"\nOutput files saved to: {DATA_DIR.absolute()}")

        except Exception as e:
            print(f"\nERROR: {e}")
            raise

        finally:
            self.close()


def main():
    """Main entry point."""
    pipeline = PregnancyPhenotyping(config_path="db_config.yaml", schema="synpuf5")
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
