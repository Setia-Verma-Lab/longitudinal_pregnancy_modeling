# Pregnancy Phenotyping Pipeline

This folder contains refactored versions of the N3C pregnancy cohort phenotyping algorithm (HIPPS) adapted for local use without Apache Spark.

## Available Versions

### 1. CSV-based version (`pregnancy_phenotyping_csv.py`)
- Reads OMOP tables from CSV files
- Clinical tables from `synpuf_etl_output/`
- Vocabulary tables from `vocab_download_v5/`
- Uses chunked reading for memory efficiency
- No database required

**Usage:**
```bash
python3 pregnancy_phenotyping_csv.py
```

### 2. PostgreSQL-optimized version (`pregnancy_phenotyping_postgres.py`)
- Connects directly to PostgreSQL database
- Uses SQL filtering at database level (much faster)
- Minimal data transfer from database
- Requires database connection configuration

**Setup:**

1. Install required Python packages:
```bash
pip install psycopg2-binary pandas pyyaml openpyxl numpy
```

2. Configure database connection in `db_config.yaml`:
```yaml
database:
  host: localhost
  port: 5432
  dbname: your_omop_database_name
  user: your_username
  password: your_password

schema: public  # or your schema name
```

3. Run the pipeline:
```bash
python3 pregnancy_phenotyping_postgres.py
```

## Pipeline Steps

Both versions implement the same 7-step HIPPS algorithm:

1. **Map external concepts** - Map Matcho concepts to standard OMOP concepts
2. **Get specificity** - Use N3C-derived highly specific pregnancy concepts
3. **HIP episodes** - Define outcome-based episodes (livebirth, stillbirth, etc.)
4. **Identify GT concepts** - Load gestational timing-specific concepts
5. **PPS episodes** - Define episodes based on pregnancy progression signatures
6. **Merge episodes** - Merge HIP and PPS episodes by temporal overlap
7. **Add metadata** - Infer start dates and add demographics

## Outputs

All intermediate and final outputs are saved to `data/`:

- `step1_matcho_concepts_standardized.csv`
- `step2_highly_specific_concepts.csv`
- `step3_hip_episodes.csv`
- `step4_pps_concepts.csv`
- `step5_pps_episodes.csv`
- `step6_merged_episodes.csv`
- `step7_pregnancy_cohort_final.csv` ← **Final pregnancy cohort**

## Caching

Both versions cache intermediate results. If you rerun the pipeline:
- Completed steps load from cache instantly
- Failed/incomplete steps rerun from scratch
- To force rerun a step, delete its output CSV from `data/`

## Performance Comparison

For ~2M patients:

| Version | Memory Usage | Runtime | Best For |
|---------|-------------|---------|----------|
| CSV | Moderate | Slower | Files already extracted |
| PostgreSQL | Low | Faster | Data in database |

## Input Requirements

### OMOP Tables Required:
- **Vocabulary**: `concept`, `concept_relationship`
- **Clinical**: `person`, `condition_occurrence`, `procedure_occurrence`, `measurement`, `observation`

### Concept Definition Files:
From `n3c_pregnancy_cohort/`:
- `HIP_concepts.xlsx`
- `Matcho_concepts.xlsx`
- `PPS_concepts.xlsx`
- `Matcho_outcome_limits.xlsx`
- `Matcho_term_durations.xlsx`
- `Highly_specific_concepts.xlsx` (optional)

## Troubleshooting

### CSV version issues:
- **Out of memory**: The chunked reading should handle large files; if issues persist, try running on a machine with more RAM
- **Missing files**: Ensure OMOP CSVs exist in `synpuf_etl_output/` and `vocab_download_v5/`

### PostgreSQL version issues:
- **Connection failed**: Check `db_config.yaml` credentials and ensure PostgreSQL is running
- **Table not found**: Verify table names match your schema (case-sensitive in some configs)
- **Slow queries**: Ensure indexes exist on `person_id` and `*_concept_id` columns

## Citation

Original N3C pregnancy cohort algorithm:
- Repository: https://github.com/jonessarae/n3c_pregnancy_cohort
- Contact: Sara Jones (sara.jones2@nih.gov), Kate Bradwell (kbradwell@palantir.com)
