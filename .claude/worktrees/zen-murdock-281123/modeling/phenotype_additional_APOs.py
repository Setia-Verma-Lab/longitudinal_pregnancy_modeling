import psycopg2
import pandas as pd
import csv

primary_phenotypes = {
    "pre_eclampsia": """SELECT id FROM public.preeclampsia""",
    "gestational_diabetes": """SELECT id FROM public.gest_diabetes""",
    "gestational_hypertension": """SELECT id FROM public.gest_hypertension""",
    "rpl": """SELECT id FROM public.rpl"""
}

output_file = "patient_phenotypes.csv"
cur = None
conn = None

try:
    # 1. Connect to the database using the correct parameters
    conn = psycopg2.connect(
        database="ohdsi",
        host="localhost",
        port="5432"
    )
    
    # 2. Create a cursor
    cur = conn.cursor()

    # 3. Construct SQL to create a phenotype matrix (0/1 indicators) for all patients
    case_statements = []
    for pheno_name, subquery in primary_phenotypes.items():
        case_statements.append(
            f"MAX(CASE WHEN condition_source_concept_id IN ({subquery}) THEN 1 ELSE 0 END) AS {pheno_name}"
        )

    matrix_query = f"""
    SELECT 
        person_id,
        {", ".join(case_statements)}
    FROM cohort_condition_occurrence
    GROUP BY person_id
    """

    # 4. Fetch the phenotype matrix data
    print("Executing phenotype matrix query...")
    cur.execute(matrix_query)
    rows = cur.fetchall()
    
    # Get column headers
    headers = ['person_id'] + list(primary_phenotypes.keys())

    # 5. Write data to CSV
    print(f"Writing phenotype matrix to {output_file}...")
    with open(output_file, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        writer.writerows(rows)

    # getting uncomplicated full term patient ids
    uncomplicated_query = """
    SELECT DISTINCT person_id
    FROM public.cohort_condition_occurrence
    WHERE person_id NOT IN (
        SELECT DISTINCT co.person_id
        FROM public.cohort_condition_occurrence co
        WHERE co.condition_source_concept_id IN (
            SELECT id FROM public.gest_hypertension UNION
            SELECT id FROM public.gest_diabetes UNION
            SELECT id FROM public.uterine_rupture UNION
            SELECT id FROM public.preterm_birth UNION
            SELECT id FROM public.preeclampsia UNION
            SELECT id FROM public.placenta_previa UNION
            SELECT id FROM public.placenta_accreta UNION
            SELECT id FROM public.hg UNION
            SELECT id FROM public.hellp UNION
            SELECT id FROM public.fetal_growth_restriction UNION
            SELECT id FROM public.ectopic UNION
            SELECT id FROM public.miscarriage UNION
            SELECT id FROM public.stillbirth UNION
            SELECT id FROM public.rpl
        )
    )
    """

    cur.execute(uncomplicated_query)
    uncomplicated_patients = cur.fetchall()
    uncomplicated_patient_ids = set([row[0] for row in uncomplicated_patients])
        
except Exception as e:
    print(f"Error: {e}")
finally:
    if cur is not None:
        cur.close()
    if conn is not None:
        conn.close()

# load initial cohort and merge with these new phenotypes with appropriate columns
initial_cohort = pd.read_csv('../pheno_data/step7_pregnancy_cohort_final.csv')
phenotype_cohort = pd.read_csv(output_file)

# Merge phenotype data with the initial cohort on 'person_id'
# We do a left join on initial_cohort to retain the intended row structure
merged_cohort = initial_cohort.merge(phenotype_cohort,on='person_id',how='left')

# Fill NaN values with 0 for patients who are in the cohort but didn't have conditions in the matrix
merged_cohort[list(primary_phenotypes.keys())] = merged_cohort[list(primary_phenotypes.keys())].fillna(0).astype(int)

# 7. Add ectopic column based on final_outcome_category.
merged_cohort['ectopic'] = (merged_cohort['final_outcome_category'].str.strip().str.upper() == 'ECT').astype(int)

# add full term column based on uncomplicated patient ids. 
merged_cohort['full_term'] = merged_cohort['person_id'].isin(uncomplicated_patient_ids).astype(int)

for col in list(primary_phenotypes.keys()) + ['ectopic', 'full_term']:
    print(merged_cohort.value_counts(col))

df = merged_cohort[merged_cohort['full_term'] == 1]
print(df['person_id'].nunique())

# 8. Save updated phenotype cohort back to the file
merged_cohort.to_csv(output_file, index=False)