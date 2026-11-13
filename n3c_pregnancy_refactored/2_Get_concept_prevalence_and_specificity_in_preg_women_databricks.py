"""
Databricks / Unity Catalog driver for N3C Step 2:
Get concept prevalence and specificity in pregnant women.

All transformation functions below are copied verbatim from
``2_Get_concept_prevalence_and_specificity_in_preg_women.py`` (functional logic unchanged).

This module adds only:
- SparkSession and Unity Catalog table loads
- Input from Step 1 output table (standardized Matcho concepts)
- Optional person prep: if ``person`` lacks ``gender_concept_name``, join ``concept`` on
  ``gender_concept_id`` (same values the original pipeline expects, e.g. FEMALE)
- Casting ``concept_id`` on the Matcho dataframe so joins match OMOP bigint keys
- Writing the final ``highly_specific_table`` result to Delta with ``LastRefreshed``
- When domain tables lack ``*_concept_name`` (standard OMOP CDM), join ``concept`` to add them

Prerequisites:
- Date columns used by the script: ``measurement_date``, ``procedure_date``, ``observation_date``,
  ``condition_start_date``, ``drug_exposure_start_date`` (typical CDM).

Configuration (edit constants):
- ``OMOP_SCHEMA``, ``CONCEPT_REF_SCHEMA``
- ``STEP1_MATCHO_TABLE``: output of Step 1 (e.g. step1_matcho_concepts_standardized)
- ``OUTPUT_TABLE``: highly specific concepts table for downstream steps

Usage:
    %run ./2_Get_concept_prevalence_and_specificity_in_preg_women_databricks.py
"""

from __future__ import annotations

from datetime import datetime

import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType
from pyspark.sql import SparkSession

# ---------------------------------------------------------------------------
# Unity Catalog
# ---------------------------------------------------------------------------
OMOP_SCHEMA = "catalog.omop"
CONCEPT_REF_SCHEMA = "catalog.schema"

PERSON_TABLE = f"{OMOP_SCHEMA}.person"
PROCEDURE_TABLE = f"{OMOP_SCHEMA}.procedure_occurrence"
MEASUREMENT_TABLE = f"{OMOP_SCHEMA}.measurement"
OBSERVATION_TABLE = f"{OMOP_SCHEMA}.observation"
CONDITION_TABLE = f"{OMOP_SCHEMA}.condition_occurrence"
DRUG_TABLE = f"{OMOP_SCHEMA}.drug_exposure"
CONCEPT_TABLE = f"{OMOP_SCHEMA}.concept"

# Standardized Matcho concepts from Step 1 (must include concept_id for joins)
STEP1_MATCHO_TABLE = f"{CONCEPT_REF_SCHEMA}.step1_matcho_concepts_standardized"
OUTPUT_TABLE = f"{CONCEPT_REF_SCHEMA}.step2_highly_specific_concepts"

# If person already has gender_concept_name (N3C-style), prep is skipped.
ADD_GENDER_CONCEPT_NAME_FROM_CONCEPT = True

# Standard OMOP domain tables have only *_concept_id; join concept to add *_concept_name.
ENRICH_DOMAIN_CONCEPT_NAMES = True

AUTO_RUN = True


def _spark() -> SparkSession:
    return SparkSession.builder.getOrCreate()


def _cast_matcho_concept_id(matcho_df):
    if "concept_id" not in matcho_df.columns:
        return matcho_df
    return matcho_df.withColumn("concept_id", F.col("concept_id").cast("long"))


def _prepare_person(person_df, concept_df):
    """IO only: add gender_concept_name when missing (standard OMOP person)."""
    if not ADD_GENDER_CONCEPT_NAME_FROM_CONCEPT or "gender_concept_name" in person_df.columns:
        return person_df
    gender = concept_df.select(
        F.col("concept_id").alias("_g_id"),
        F.col("concept_name").alias("gender_concept_name"),
    )
    return person_df.join(
        gender, person_df.gender_concept_id == gender._g_id, "left"
    ).drop("_g_id")


def _add_concept_name_if_missing(df, concept_id_col: str, concept_name_col: str, concept_df):
    """
    Driver-only: N3C scripts expect denormalized *concept_name on domain rows.
    If the column is absent, left-join OMOP ``concept`` on the domain concept_id.
    """
    if not ENRICH_DOMAIN_CONCEPT_NAMES or concept_name_col in df.columns:
        return df
    lookup = concept_df.select(
        F.col("concept_id").cast("long").alias("_lookup_cid"),
        F.col("concept_name").alias(concept_name_col),
    )
    return df.join(
        lookup,
        F.col(concept_id_col).cast("long") == F.col("_lookup_cid"),
        "left",
    ).drop("_lookup_cid")


def _enrich_domain_tables_with_concept_names(
    procedure_occurrence,
    measurement,
    observation,
    condition_occurrence,
    drug_exposure,
    concept_df,
):
    observation = _add_concept_name_if_missing(
        observation, "observation_concept_id", "observation_concept_name", concept_df
    )
    measurement = _add_concept_name_if_missing(
        measurement, "measurement_concept_id", "measurement_concept_name", concept_df
    )
    procedure_occurrence = _add_concept_name_if_missing(
        procedure_occurrence, "procedure_concept_id", "procedure_concept_name", concept_df
    )
    condition_occurrence = _add_concept_name_if_missing(
        condition_occurrence, "condition_concept_id", "condition_concept_name", concept_df
    )
    drug_exposure = _add_concept_name_if_missing(
        drug_exposure, "drug_concept_id", "drug_concept_name", concept_df
    )
    return (
        procedure_occurrence,
        measurement,
        observation,
        condition_occurrence,
        drug_exposure,
    )


# =============================================================================
# Original N3C Step 2 logic (unchanged)
# =============================================================================


def filter_to_persons_with_concepts(procedure_occurrence, measurement, observation, condition_occurrence, drug_exposure, Matcho_concepts):
    """
    Identify persons with a concept from a table with concepts of interest. 
    Returns a dataframe containing the person_id of each person with a concept. 

    Parameters:
    procedure_occurrence
    measurement
    observation
    condition_occurrence
    drug_exposure
    Matcho_concepts_df - table with concepts of interest 
    """

    preg_df = Matcho_concepts

    # filter observation table
    observation_df = observation.join(preg_df, (preg_df.concept_id == observation.observation_concept_id), "inner")
    # select person_id column
    observation_df = observation_df.select("person_id")

    # filter measurement table
    measurement_df = measurement.join(preg_df, (preg_df.concept_id == measurement.measurement_concept_id), "inner")
    # select person_id column
    measurement_df = measurement_df.select("person_id")

    # filter procedure table
    procedure_df = procedure_occurrence.join(preg_df, (preg_df.concept_id == procedure_occurrence.procedure_concept_id), "inner")
    # select person_id column
    procedure_df = procedure_df.select("person_id")    

    # filter condition table
    condition_df = condition_occurrence.join(preg_df, (preg_df.concept_id == condition_occurrence.condition_concept_id), "inner")
    # select person_id column
    condition_df = condition_df.select("person_id")

    # filter drug table
    drug_df = drug_exposure.join(preg_df, (preg_df.concept_id == drug_exposure.drug_exposure_id), "inner")
    # select person_id column
    drug_df = drug_df.select("person_id")

    # combine tables
    union_df = measurement_df.union(procedure_df)
    union_df = union_df.union(observation_df)
    union_df = union_df.union(condition_df)
    union_df = union_df.union(drug_df)

    return union_df.dropDuplicates()

def persons_with_concepts(person, filter_to_persons_with_concepts):
    """
    Filter person table for persons with a concept of interest.

    Parameters:
    person
    filter_to_persons_with_concepts - dataframe of person_id for each person with a concept of interest
    """

    df = person.join(filter_to_persons_with_concepts, "person_id", "inner")
    df = df.select("person_id", "gender_concept_name", "year_of_birth", "month_of_birth", "day_of_birth")

    # create day column and fill in any missing values with 1
    df = df.withColumn("day", F.when(F.col("day_of_birth").isNull(), 1).otherwise(F.col("day_of_birth")))
    # create date_of_birth column
    df = df.withColumn("date_of_birth",F.concat_ws("-", F.col("year_of_birth"), F.col("month_of_birth"), F.col("day")).cast("date"))

    df = df.select("person_id", "gender_concept_name", "date_of_birth")
    
    return df
    
def get_all_concepts(persons_with_concepts, observation, measurement, procedure_occurrence, condition_occurrence, drug_exposure):
    """
    Get all records from the domain tables listed below for every person with a concept of interest.

    Parameters:
    procedure_occurrence
    measurement
    observation
    condition_occurrence
    drug_exposure
    persons_with_concepts
    """

    person_df = persons_with_concepts

    # filter observation table
    observation_df = observation.select("person_id","observation_concept_id","observation_concept_name","observation_date")
    observation_df = observation_df.join(person_df, "person_id", "inner")
    observation_df = observation_df.withColumnRenamed("observation_concept_id","concept_id")
    observation_df = observation_df.withColumnRenamed("observation_concept_name","concept_name")
    observation_df = observation_df.withColumnRenamed("observation_date","visit_date")
    observation_df = observation_df.withColumn("domain", F.lit("observation"))

    # filter measurement table
    measurement_df = measurement.select("person_id","measurement_concept_id","measurement_concept_name","measurement_date")
    measurement_df = measurement_df.join(person_df, "person_id", "inner")
    measurement_df = measurement_df.withColumnRenamed("measurement_concept_id","concept_id")
    measurement_df = measurement_df.withColumnRenamed("measurement_concept_name","concept_name")
    measurement_df = measurement_df.withColumnRenamed("measurement_date","visit_date")
    measurement_df = measurement_df.withColumn("domain", F.lit("measurement"))

    # filter procedure table
    procedure_df = procedure_occurrence.select("person_id","procedure_concept_id","procedure_concept_name","procedure_date")
    procedure_df = procedure_df.join(person_df, "person_id", "inner")
    procedure_df = procedure_df.withColumnRenamed("procedure_concept_id","concept_id")
    procedure_df = procedure_df.withColumnRenamed("procedure_concept_name","concept_name")
    procedure_df = procedure_df.withColumnRenamed("procedure_date","visit_date")
    procedure_df = procedure_df.withColumn("domain", F.lit("procedure"))  

    # filter condition table
    condition_df = condition_occurrence.select("person_id","condition_concept_id","condition_concept_name","condition_start_date")
    condition_df = condition_df.join(person_df, "person_id", "inner")
    condition_df = condition_df.withColumnRenamed("condition_concept_id","concept_id")
    condition_df = condition_df.withColumnRenamed("condition_concept_name","concept_name")
    condition_df = condition_df.withColumnRenamed("condition_start_date","visit_date")
    condition_df = condition_df.withColumn("domain", F.lit("condition")) 

    # filter drug table
    drug_df = drug_exposure.select("person_id","drug_concept_id","drug_concept_name","drug_exposure_start_date")
    drug_df = drug_df.join(person_df, "person_id", "inner")
    drug_df = drug_df.withColumnRenamed("drug_concept_id","concept_id")
    drug_df = drug_df.withColumnRenamed("drug_concept_name","concept_name")
    drug_df = drug_df.withColumnRenamed("drug_exposure_start_date","visit_date")
    drug_df = drug_df.withColumn("domain", F.lit("drug")) 

    # combine tables
    union_df = measurement_df.union(procedure_df)
    union_df = union_df.union(observation_df)
    union_df = union_df.union(condition_df)
    union_df = union_df.union(drug_df)

    # drop any duplicate person_ids and concept_ids
    union_df = union_df.dropDuplicates(["person_id","concept_id"])

    # get difference in days between visit date and date of birth
    union_df = union_df.withColumn("date_diff", F.datediff(union_df.visit_date, union_df.date_of_birth))

    # calculate age from date_diff
    union_df = union_df.withColumn("age", (F.col("date_diff")/365))

    return union_df
    
def filter_to_women_of_reproductive_age(get_all_concepts):
    """
    Filter to records belonging to women of reproductive age (15-55).

    Parameters:
    get_all_concepts - all records belonging to persons with a concept of interest
    """
    df = get_all_concepts
    df = df.filter(df.gender_concept_name == "FEMALE")
    df = df.filter((df.age >= 15) & (df.age < 56))
    return df
    
def persons_without_concepts(person, filter_to_persons_with_concepts):
    """
    Filter person table for persons without a concept of interest.

    Parameters:
    person
    filter_to_persons_with_concepts - dataframe of person_id for each person with a concept of interest

    Note: The published N3C script applies ``filter(has_concept is null)`` after a ``select`` that
    drops ``has_concept``, which fails in Spark. Filter is applied before that ``select`` here so
    the intended left-anti-join semantics match runnable Spark code.
    """ 
    
    preg_df = filter_to_persons_with_concepts.select("person_id")
    preg_df = preg_df.withColumn("has_concept", F.lit(1))
    df = person.join(preg_df, "person_id", "left")
    df = df.filter(F.col("has_concept").isNull())
    df = df.select("person_id", "gender_concept_name", "year_of_birth", "month_of_birth", "day_of_birth")
    
    # create day column and fill in any missing values with 1
    df = df.withColumn("day", F.when(F.col("day_of_birth").isNull(), 1).otherwise(F.col("day_of_birth")))
    # create date_of_birth column
    df = df.withColumn("date_of_birth",F.concat_ws("-", F.col("year_of_birth"), F.col("month_of_birth"), F.col("day")).cast("date"))

    df = df.select("person_id", "gender_concept_name", "date_of_birth")

    return df

def get_count_and_proportion(filter_to_women_of_reproductive_age):
    """
    Calculate count and proportion of unique persons by concept.

    Parameters:
    filter_to_women_of_reproductive_age - all records belonging to women of reproductive age with indication of pregnancy
    """
    df = filter_to_women_of_reproductive_age

    # get total number of unique persons
    person_num = df.select(F.countDistinct("person_id").alias("number_ids"))
    total = person_num.collect()[0]["number_ids"]
    print(total)

    # get count of unique persons by concept
    count_df = df.groupBy("concept_id","concept_name").agg(F.countDistinct("person_id").alias("count"))

    # add column for proportion
    count_df = count_df.withColumn("proportion", F.col("count").cast(IntegerType())/total)

    # remove concepts without concept names or ids
    count_df = count_df.filter(~(F.col("concept_id") == 0))
    count_df = count_df.filter(~F.col("concept_name").isNull())

    return count_df

def filter_to_frequent_concepts(get_count_and_proportion):
    """
    Filter to frequent concepts (count > 1000).

    Parameters:
    get_count_and_proportion - concepts with count and proportion
    """
    df = get_count_and_proportion
    df = df.filter(F.col("count") > 1000)
    return df

def get_frequent_concepts(filter_to_frequent_concepts, persons_without_concepts, observation, measurement, procedure_occurrence, condition_occurrence, drug_exposure):
    """
    Get all records of frequent concepts from the domain tables listed below for every person without a concept of interest.

    Parameters:
    procedure_occurrence
    measurement
    observation
    condition_occurrence
    drug_exposure
    filter_to_frequent_concepts - frequent concepts in persons with concept of interest
    persons_without_concepts - persons without concept of interest
    """
    concept_df = filter_to_frequent_concepts.select("concept_id")
    person_df = persons_without_concepts

    # filter observation table
    observation_df = observation.select("person_id","observation_concept_id","observation_concept_name","observation_date")
    observation_df = observation_df.withColumnRenamed("observation_concept_id","concept_id")
    observation_df = observation_df.join(person_df, "person_id", "inner")
    observation_df = observation_df.withColumnRenamed("observation_concept_name","concept_name")
    observation_df = observation_df.withColumnRenamed("observation_date","visit_date")

    # filter measurement table
    measurement_df = measurement.select("person_id","measurement_concept_id","measurement_concept_name","measurement_date")
    measurement_df = measurement_df.withColumnRenamed("measurement_concept_id","concept_id")
    measurement_df = measurement_df.join(person_df, "person_id", "inner")
    measurement_df = measurement_df.withColumnRenamed("measurement_concept_name","concept_name")
    measurement_df = measurement_df.withColumnRenamed("measurement_date","visit_date")

    # filter procedure table
    procedure_df = procedure_occurrence.select("person_id","procedure_concept_id","procedure_concept_name","procedure_date")
    procedure_df = procedure_df.withColumnRenamed("procedure_concept_id","concept_id")
    procedure_df = procedure_df.join(person_df, "person_id", "inner")
    procedure_df = procedure_df.withColumnRenamed("procedure_concept_name","concept_name")
    procedure_df = procedure_df.withColumnRenamed("procedure_date","visit_date") 

    # filter condition table
    condition_df = condition_occurrence.select("person_id","condition_concept_id","condition_concept_name","condition_start_date")
    condition_df = condition_df.withColumnRenamed("condition_concept_id","concept_id")
    condition_df = condition_df.join(person_df, "person_id", "inner")
    condition_df = condition_df.withColumnRenamed("condition_concept_id","concept_id")
    condition_df = condition_df.withColumnRenamed("condition_concept_name","concept_name")
    condition_df = condition_df.withColumnRenamed("condition_start_date","visit_date")

    # filter drug table
    drug_df = drug_exposure.select("person_id","drug_concept_id","drug_concept_name","drug_exposure_start_date")
    drug_df = drug_df.withColumnRenamed("drug_concept_id","concept_id")
    drug_df = drug_df.join(person_df, "person_id", "inner")
    drug_df = drug_df.withColumnRenamed("drug_concept_name","concept_name")
    drug_df = drug_df.withColumnRenamed("drug_exposure_start_date","visit_date")

    # combine tables
    union_df = measurement_df.union(procedure_df)
    union_df = union_df.union(observation_df)
    union_df = union_df.union(condition_df)
    union_df = union_df.union(drug_df)

    # drop any duplicate person_ids and concept_ids
    union_df = union_df.dropDuplicates(["person_id","concept_id"])

    # get difference in days between visit date and date of birth
    union_df = union_df.withColumn("date_diff", F.datediff(union_df.visit_date, union_df.date_of_birth))

    # calculate age from date_diff
    union_df = union_df.withColumn("age", (F.col("date_diff")/365))

    return union_df    

def get_count_and_proportion_for_controls(get_frequent_concepts):
    """
    Calculate count and proportion of unique persons by concept.

    Parameters:
    get_frequent_concepts - all records of frequent concepts from controls (persons without concept of interest)
    """
    df = get_frequent_concepts

    # get total number of unique persons
    person_num = df.select(F.countDistinct("person_id").alias("number_ids"))
    total = person_num.collect()[0]["number_ids"]
    print(total)

    # get count of unique persons by concept
    count_df = df.groupBy("concept_id","concept_name").agg(F.countDistinct("person_id").alias("count_controls"))

    return count_df

def count_and_proportion_all(get_count_and_proportion_for_controls, filter_to_frequent_concepts, get_frequent_concepts):
    """
    Combine tables of count and proportion by concept for cases and controls and calculate the ratio of proportions.

    Parameters:
    get_count_and_proportion_for_controls - count and proportion of controls (persons without concept of interest) by concept
    filter_to_frequent_concepts - count and proportion of cases (persons with concept of interest) by concept
    """
    cases = filter_to_frequent_concepts
    controls = get_count_and_proportion_for_controls.drop("concept_name")
    
    # join tables
    df = cases.join(controls, "concept_id", "left")

    # impute 1 for any missing counts
    df = df.withColumn("count_controls", F.when(F.col("count_controls").isNull(), F.lit(1)).otherwise(F.lit(F.col("count_controls"))))

    # get total number of unique persons in control group 
    person_num = get_frequent_concepts.select(F.countDistinct("person_id").alias("number_ids"))
    control_total = person_num.collect()[0]["number_ids"]

    # add column for proportion in control group
    df = df.withColumn("proportion_controls", F.col("count_controls").cast(IntegerType())/control_total)
    
    # add column for ratio in control group
    df = df.withColumn("ratio", F.col("proportion")/F.col("proportion_controls"))
    
    return df
    
def highly_specific_table(count_and_proportion_all):
    """
    Get highly specific concepts with a ratio greater than 10.
    """
    df = count_and_proportion_all.filter(F.col('ratio') > 10)
    return df


def main(
    write_output: bool = True,
    output_table: str = OUTPUT_TABLE,
):
    spark = _spark()

    person = spark.table(PERSON_TABLE)
    procedure_occurrence = spark.table(PROCEDURE_TABLE)
    measurement = spark.table(MEASUREMENT_TABLE)
    observation = spark.table(OBSERVATION_TABLE)
    condition_occurrence = spark.table(CONDITION_TABLE)
    drug_exposure = spark.table(DRUG_TABLE)
    concept = spark.table(CONCEPT_TABLE)

    procedure_occurrence, measurement, observation, condition_occurrence, drug_exposure = (
        _enrich_domain_tables_with_concept_names(
            procedure_occurrence,
            measurement,
            observation,
            condition_occurrence,
            drug_exposure,
            concept,
        )
    )

    Matcho_concepts_df = _cast_matcho_concept_id(spark.table(STEP1_MATCHO_TABLE))

    person = _prepare_person(person, concept)

    filter_to_persons_with_concepts_df = filter_to_persons_with_concepts(
        procedure_occurrence,
        measurement,
        observation,
        condition_occurrence,
        drug_exposure,
        Matcho_concepts_df,
    )
    persons_with_concepts_df = persons_with_concepts(person, filter_to_persons_with_concepts_df)
    get_all_concepts_df = get_all_concepts(
        persons_with_concepts_df,
        observation,
        measurement,
        procedure_occurrence,
        condition_occurrence,
        drug_exposure,
    )
    filter_to_women_of_reproductive_age_df = filter_to_women_of_reproductive_age(get_all_concepts_df)
    persons_without_concepts_df = persons_without_concepts(person, filter_to_persons_with_concepts_df)
    get_count_and_proportion_df = get_count_and_proportion(filter_to_women_of_reproductive_age_df)
    filter_to_frequent_concepts_df = filter_to_frequent_concepts(get_count_and_proportion_df)
    get_frequent_concepts_df = get_frequent_concepts(
        filter_to_frequent_concepts_df,
        persons_without_concepts_df,
        observation,
        measurement,
        procedure_occurrence,
        condition_occurrence,
        drug_exposure,
    )
    get_count_and_proportion_for_controls_df = get_count_and_proportion_for_controls(get_frequent_concepts_df)
    count_and_proportion_all_df = count_and_proportion_all(
        get_count_and_proportion_for_controls_df,
        filter_to_frequent_concepts_df,
        get_frequent_concepts_df,
    )
    highly_specific_table_df = highly_specific_table(count_and_proportion_all_df)

    out = highly_specific_table_df.withColumn("LastRefreshed", F.lit(datetime.now()))
    if write_output:
        (
            out.write.mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(output_table)
        )
        print(f"Wrote {out.count()} rows to {output_table}")

    return highly_specific_table_df


if __name__ == "__main__" or AUTO_RUN:
    main()
