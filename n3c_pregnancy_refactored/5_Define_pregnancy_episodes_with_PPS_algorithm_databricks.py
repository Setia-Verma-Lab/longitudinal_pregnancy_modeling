"""
Databricks / Unity Catalog driver for N3C Step 5:
Define pregnancy episodes with the PPS algorithm.

All functions below are unchanged from
``5_Define_pregnancy_episodes_with_PPS_algorithm.py``.

This module adds only:
- ``SparkSession`` and Unity Catalog loads (``catalog.omop``, ``catalog.schema``)
- ``pps_concepts`` reference table (``domain_concept_id``, ``min_month``, ``max_month``, etc.)
- Join ``concept`` for ``*_concept_name`` on domain / visit tables when missing (standard OMOP)
- ``gender_concept_name`` on ``person`` when missing
- Module-level ``spark`` for ``get_PPS_episodes`` (``spark.createDataFrame``)
- Delta outputs: ``step5_pps_episodes``, ``step5_pps_episode_concepts`` (for Step 6), with ``LastRefreshed``

Edit constants at the top if your table names differ.
"""

from __future__ import annotations

from datetime import datetime

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DateType
from pyspark.sql.functions import expr
from itertools import chain
from pyspark.sql.types import ArrayType, IntegerType, DateType, StructType
import pandas as pd
import operator
from pyspark.sql.functions import pandas_udf,PandasUDFType
import collections
from pyspark.sql import Row

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
VISIT_TABLE = f"{OMOP_SCHEMA}.visit_occurrence"
CONCEPT_TABLE = f"{OMOP_SCHEMA}.concept"

PPS_CONCEPTS_TABLE = f"{CONCEPT_REF_SCHEMA}.pps_concepts"
OUTPUT_TABLE = f"{CONCEPT_REF_SCHEMA}.step5_pps_episodes"
PPS_EPISODE_CONCEPTS_TABLE = f"{CONCEPT_REF_SCHEMA}.step5_pps_episode_concepts"

ADD_GENDER_CONCEPT_NAME_FROM_CONCEPT = True
ENRICH_DOMAIN_CONCEPT_NAMES = True
AUTO_RUN = True

# Used by get_PPS_episodes (spark.createDataFrame); set in main().
spark = None


def _get_spark() -> SparkSession:
    return SparkSession.builder.getOrCreate()


def _prepare_person(person_df, concept_df):
    if not ADD_GENDER_CONCEPT_NAME_FROM_CONCEPT or "gender_concept_name" in person_df.columns:
        return person_df
    gender = concept_df.select(
        F.col("concept_id").cast("long").alias("_g_id"),
        F.col("concept_name").alias("gender_concept_name"),
    )
    return person_df.join(
        gender, F.col("gender_concept_id").cast("long") == F.col("_g_id"), "left"
    ).drop("_g_id")


def _cast_pps_concept_ids(pps_df):
    if "domain_concept_id" in pps_df.columns:
        pps_df = pps_df.withColumn("domain_concept_id", F.col("domain_concept_id").cast("long"))
    return pps_df


def _pps_collect_list_fields(item):
    """Read (start_date, concept_id) from collect_list(struct) after Spark -> pandas."""
    if isinstance(item, (list, tuple)):
        return item[0], int(item[1])
    if isinstance(item, dict):
        return item["domain_concept_start_date"], int(item["domain_concept_id"])
    try:
        return item["domain_concept_start_date"], int(item["domain_concept_id"])
    except (KeyError, TypeError):
        pass
    try:
        return item.domain_concept_start_date, int(item.domain_concept_id)
    except AttributeError:
        pass
    return item[0], int(item[1])


def _add_concept_name_if_missing(df, concept_id_col: str, concept_name_col: str, concept_df):
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


def _enrich_domain_tables_for_step5(
    procedure_occurrence,
    measurement,
    observation,
    condition_occurrence,
    visit_occurrence,
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
    visit_occurrence = _add_concept_name_if_missing(
        visit_occurrence, "visit_concept_id", "visit_concept_name", concept_df
    )
    return (
        procedure_occurrence,
        measurement,
        observation,
        condition_occurrence,
        visit_occurrence,
    )


def input_GT_concepts(condition_occurrence, procedure_occurrence, observation, measurement, visit_occurrence, PPS_concepts):
    c_o = condition_occurrence
    p_o = procedure_occurrence
    o_df = observation
    m_df = measurement
    v_o = visit_occurrence
    top_concepts = PPS_concepts
    
    def rename_cols(top_concepts,df,start_date_col,id_col,name_col):
        df = df.withColumnRenamed(start_date_col, 'domain_concept_start_date')
        df = df.withColumnRenamed(id_col, 'domain_concept_id')
        df = df.withColumnRenamed(name_col, 'domain_concept_name')
        # Join on PPS ids only; pps_concepts also has domain_concept_name -> ambiguous if joined whole table
        pps_ids = top_concepts.select(
            F.col("domain_concept_id").cast("long").alias("domain_concept_id")
        ).distinct()
        top_concepts_df = df.join(pps_ids, 'domain_concept_id', 'inner')
        top_concepts_df = top_concepts_df.select('person_id','domain_concept_start_date','domain_concept_id','domain_concept_name')
        return top_concepts_df.distinct()

    c_o = rename_cols(top_concepts,c_o,'condition_start_date','condition_concept_id','condition_concept_name')
    p_o = rename_cols(top_concepts,p_o,'procedure_date','procedure_concept_id','procedure_concept_name')
    o_df = rename_cols(top_concepts,o_df,'observation_date','observation_concept_id','observation_concept_name')
    m_df = rename_cols(top_concepts,m_df,'measurement_date','measurement_concept_id','measurement_concept_name')
    v_o = rename_cols(top_concepts,v_o,'visit_start_date','visit_concept_id','visit_concept_name')
    top_preg_related_concepts = (((c_o.unionByName(p_o)).unionByName(o_df)).unionByName(m_df)).unionByName(v_o)

    # enclave_data_start_date = "2018-01-01"
    # date_from = F.to_date(F.lit(enclave_data_start_date)).cast(DateType())
    # enclave_data_end_date = "2022-04-07"
    # date_to = F.to_date(F.lit(enclave_data_end_date)).cast(DateType())

    # top_preg_related_concepts = top_preg_related_concepts.filter(F.col('domain_concept_start_date') > date_from)

    # top_preg_related_concepts = top_preg_related_concepts.filter(F.col('domain_concept_start_date') <= date_to)

    return top_preg_related_concepts

def get_PPS_episodes(PPS_concepts, input_GT_concepts, person):
    '''
    OBTAIN ALL RELEVANT INPUT PATIENTS AND SAVE GT INFORMATION PER CONCEPT TO A LOOKUP DICTIONARY 
    First we save the women that have gestational timing concepts, and save the gestational timing information for each concept.
    We add the concepts and their gestational timing months ([min,max]) during pregnancy to a dictionary (hash) in 
    concept key: month value list format e.g. {2211756: [4,8], 2101830: [2,2]...} 
    '''
    df = PPS_concepts
    personDF = person
    patients_with_preg_concepts = input_GT_concepts
    concepts_list = patients_with_preg_concepts.select('person_id').distinct()

    # get all the unique people that have any of the gestational timing concepts along with their demographics
    personDF = personDF.join(concepts_list,'person_id','inner')
    personDF = personDF.toPandas()
    personDF = personDF[['person_id','gender_concept_name', 'year_of_birth']].drop_duplicates()
    personDF = personDF.rename(columns={"person_id": "personID"})

    # preserve the concept name information (used as a look up later on)
    id2name = df.select('domain_concept_id','domain_concept_name').distinct()
    id2name = id2name.withColumnRenamed('domain_concept_id','concept_id')

    # save the gestational time ranges per concept and convert it to a dictionary for ease of lookup
    df = df.select('domain_concept_id','min_month','max_month')
    df = df.withColumn('month_range', F.concat(F.col('min_month'),F.lit(','), F.col('max_month')))
    map_df = df.select('domain_concept_id','month_range')
    map_df = map_df.toPandas()
    value_map = dict(zip(map_df.domain_concept_id, map_df.month_range))

    # reformat the dictionary values to be a list of months
    concept2monthRangeList = {}
    for k in value_map:
        kid = int(k)
        if kid not in concept2monthRangeList:
            concept2monthRangeList[kid] = []
        keyval = value_map[k]
        new_keyval_list = keyval.strip().split(",")
        concept2monthRangeList[kid].append(float(new_keyval_list[0]))
        concept2monthRangeList[kid].append(float(new_keyval_list[1]))

    ''' 
    SAVE EXPECTED GESTATIONAL TIMING MONTH INFORMATION FOR EACH OF THE PATIENT RECORDS
    Looping over each person with pregancy concepts, order their concepts by date of each record, and for each concept ID in order, loop through the 
    keys of the dictionary and compare to the concept ID, if there’s a match, save the month value(s) to a list for the record date. You’ll end up with 
    record date: list of matching months, save this to a new dictionary with record dates as the keys. Where no match occurs, put NA
    '''

    # filter out cases where date is null
    patients_with_preg_concepts = patients_with_preg_concepts.filter(F.col('domain_concept_start_date').isNotNull())

    # create list column
    grouped_df = patients_with_preg_concepts.groupby("person_id").agg(F.collect_list(F.struct("domain_concept_start_date", "domain_concept_id")).alias("list_col"))

    person_dates_df = grouped_df.toPandas()
    person_dates_dict = dict(zip(person_dates_df.person_id, person_dates_df.list_col))

    new_person_dict = {}
    newDFrows = []

    for k in person_dates_dict:
        personlist = []
        for listitem in person_dates_dict[k]:
            rowdate, rowconcept = _pps_collect_list_fields(listitem)
            rowConceptMonthList = concept2monthRangeList[rowconcept]
            newlist = [rowdate,rowconcept,rowConceptMonthList,k]
            personlist.append(newlist)
        personlist.sort(key=lambda x: x[0])

        '''
        INITIAL EPISODE DEFINITION
        Filter to plausible pregnancy timelines and concept month sequences, and number by episode to get person_episode_number
        '''
        person_episode_number = 1
        person_episodes = []
        person_episode_dates = {}

        # treat the first record as belonging to the first episode
        person_episodes.append(person_episode_number)
        if person_episode_number not in person_episode_dates:
            person_episode_dates[person_episode_number] = []
        person_episode_dates[person_episode_number].append(personlist[0][0])

        # the below loop automatically only runs for patients that have multiple records for the GT concepts
        for i in range(1,len(personlist)):
            delta_t = ((personlist[i][0] - personlist[i-1][0]).days/30)
            if (k == '<person_id to check>'):
                print(delta_t)
            def records_comparison(personlist,i,k):
                # for the below: t = time (actual), c = concept (expected)
                # first do the comparisons to the records PREVIOUS to record i
                if (k == '<person_id to check>'):
                    print('made it to function')
                return_agreement_t_c = False
                for j in range(1,(i+1)):
                    # obtain difference in actual dates of the consecutive patient records
                    delta_t = ((personlist[i][0] - personlist[i-j][0]).days/30)
                    # obtain the max expected month difference based on clinician knowledge of the two concepts (allow two extra months for leniency)
                    adjConceptMonths_MaxExpectedDelta = (personlist[i][2][1] - personlist[i-j][2][0]) + 2
                    # obtain the min expected month difference based on clinician knowledge of the two concepts (allow two extra months for leniency)
                    adjConceptMonths_MinExpectedDelta = (personlist[i][2][0] - personlist[i-j][2][1]) - 2
                    # save a boolean. TEST = whether the actual date diff falls within the max and min expected date diffs for the consecutive concepts based on their 
                    # clinician-assigned ranges. 
                    agreement_t_c = ((adjConceptMonths_MaxExpectedDelta >= delta_t) and (delta_t >= adjConceptMonths_MinExpectedDelta))
                    if (k == '<person_id to check>'):
                        print('prev record checks for person')
                        print(delta_t)
                        print(adjConceptMonths_MaxExpectedDelta)
                        print(adjConceptMonths_MinExpectedDelta)
                        print(agreement_t_c)
                    if agreement_t_c == True:
                        return_agreement_t_c = True
                # next do the comparisons to the records SOURROUNDING record i
                len_to_start = i
                len_to_end = (len(personlist) - 1) - i
                bridge_len = min(len_to_start,len_to_end)
                for b in range(1,(bridge_len+1)):
                    # bridge check: take bridge records around record i, in case record i was an outlier
                    bridge_delta_t = ((personlist[i+b][0] - personlist[i-b][0]).days/30)
                    bridge_adjConceptMonths_MaxExpectedDelta = (personlist[i+b][2][1] - personlist[i-b][2][0]) + 2
                    bridge_adjConceptMonths_MinExpectedDelta = (personlist[i+b][2][0] - personlist[i-b][2][1]) - 2
                    bridge_agreement_t_c = ((bridge_adjConceptMonths_MaxExpectedDelta >= bridge_delta_t) and (bridge_delta_t >= bridge_adjConceptMonths_MinExpectedDelta))
                    if bridge_agreement_t_c == True:
                        return_agreement_t_c = True
                return return_agreement_t_c
            # perform the checks to determine whether this is a continuation of an episode or the start of a new episode
            agreement_t_c = records_comparison(personlist,i,k)
            if (k == '<person_id to check>'):
                print(agreement_t_c)
            if ((agreement_t_c == False) and (delta_t > 2)): 
                # none of the matches on any iteration were true, and the concepts aren't within 2 months of each other (min retry period for any outcome)
                person_episode_number +=1
            elif delta_t > 10:
                person_episode_number +=1
            if (k == '<person_id to check>'):
                print('i')
                print(i)
                print('person_episode_number')
                print(person_episode_number)
                print('delta_t')
                print(delta_t)
                print('agreement_t_c')
                print(agreement_t_c)
            person_episodes.append(person_episode_number)
            if person_episode_number not in person_episode_dates:
                person_episode_dates[person_episode_number] = []
            person_episode_dates[person_episode_number].append(personlist[i][0])

        ''' 
        EPISODE REFINEMENT
         - Check that all the episodes are < 12 mo in length (the 9-10 mo of pregnancy plus the few months of delivery concept ramblings). In the case that
        any have to be removed, loop through the episodes of the patient again and renumber the remaining episodes
        - Remove males (newborns) and females that are not of reproductive age
        '''

        episodes_to_remove = []
        for episode in person_episode_dates:
            len_of_episode = ((person_episode_dates[episode][-1] - person_episode_dates[episode][0]).days / 30)
            if len_of_episode > 12:
                episodes_to_remove.append(episode)
        new_person_episodes = [0 if x in episodes_to_remove else x for x in person_episodes]
        numUniqueNonZero = sum(1 for i in set(new_person_episodes) if i != 0)
        nonZeroNewList = [i for i in range(1,numUniqueNonZero+1)]
        nonZeroOrigList = [i for i in (set(new_person_episodes)) if i != 0]
        for j in range(0,len(nonZeroNewList)):
            new_person_episodes = [nonZeroNewList[j] if item == nonZeroOrigList[j] else item for item in new_person_episodes]
        for m in range(len(personlist)):
            personlist[m].append(new_person_episodes[m])
        for person_record in personlist:
            newDFrows.append(person_record)
        new_person_dict[k] = personlist

    preg_df=pd.DataFrame(newDFrows,columns=['conceptDate','concept_id','conceptMonth','personID','person_episode_number'])

    # add in the concept names
    id2name = id2name.toPandas()
    preg_df = preg_df.join(id2name.set_index('concept_id'), on='concept_id',how='left')

    # filter out non-typical episode length episodes (assigned zeros)
    preg_df = preg_df.loc[preg_df['person_episode_number'] > 0]

    # remove all men (mostly male babies)
    preg_df = preg_df.join(personDF.set_index('personID'), on='personID',how='left')
    preg_df = preg_df.loc[preg_df['gender_concept_name'] == 'FEMALE']
    preg_df = preg_df.loc[(preg_df['year_of_birth'] >= 1965) & (preg_df['year_of_birth'] <= 2005)]

    preg_df = preg_df.sort_values(["personID", "conceptDate"], ascending = (True, True))

    preg_df=spark.createDataFrame(preg_df) 
    # remove patients (and all their episodes) in cases where the patient is assigned > 5 episodes per year out of the total cohort time range
    frequency_dataframe = preg_df.withColumn('partitionAll',F.lit('1'))
    windowSpecDates = Window.partitionBy("partitionAll")
    windowSpecEpisodesPerPerson = Window.partitionBy("personID")
    frequency_dataframe = frequency_dataframe.withColumn("maxDate", F.max(F.col("conceptDate")).over(windowSpecDates)) \
                                             .withColumn("minDate", F.min(F.col("conceptDate")).over(windowSpecDates))
    frequency_dataframe = frequency_dataframe.withColumn("maxEpisodeNum", F.max(F.col("person_episode_number")).over(windowSpecEpisodesPerPerson)) \
                                             .withColumn('cohort_period', (F.datediff(F.col("maxDate"),F.col("minDate")))/365) \
                                             .withColumn('episode_freq', F.col('maxEpisodeNum') / F.col('cohort_period'))
    frequency_dataframe = frequency_dataframe.filter(F.col('episode_freq') < 5).select('personID').distinct()
    preg_df = preg_df.join(frequency_dataframe,'personID','inner')

    return preg_df

def get_episode_max_min_dates(get_PPS_episodes):
    df = get_PPS_episodes
    df = df.groupby('personID','person_episode_number').agg(F.min(F.col('conceptDate')).alias('episode_min_date'),F.max(F.col('conceptDate')).alias('episode_max_date'))
    df = df.withColumn('episode_max_date_plus_two_months', expr("add_months(episode_max_date, 2)"))
    return df

def main(write_output: bool = True, output_table: str = OUTPUT_TABLE):
    global spark
    spark = _get_spark()

    person = spark.table(PERSON_TABLE)
    procedure_occurrence = spark.table(PROCEDURE_TABLE)
    measurement = spark.table(MEASUREMENT_TABLE)
    observation = spark.table(OBSERVATION_TABLE)
    condition_occurrence = spark.table(CONDITION_TABLE)
    visit_occurrence = spark.table(VISIT_TABLE)
    concept = spark.table(CONCEPT_TABLE)

    procedure_occurrence, measurement, observation, condition_occurrence, visit_occurrence = (
        _enrich_domain_tables_for_step5(
            procedure_occurrence,
            measurement,
            observation,
            condition_occurrence,
            visit_occurrence,
            concept,
        )
    )
    person = _prepare_person(person, concept)

    PPS_concepts = _cast_pps_concept_ids(spark.table(PPS_CONCEPTS_TABLE))

    input_GT_concepts_df = input_GT_concepts(
        condition_occurrence,
        procedure_occurrence,
        observation,
        measurement,
        visit_occurrence,
        PPS_concepts,
    )
    get_PPS_episodes_df = get_PPS_episodes(PPS_concepts, input_GT_concepts_df, person)
    PPS_episodes_df = get_episode_max_min_dates(get_PPS_episodes_df)

    refreshed = F.lit(datetime.now())
    if write_output:
        (
            get_PPS_episodes_df.withColumn("LastRefreshed", refreshed)
            .write.mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(PPS_EPISODE_CONCEPTS_TABLE)
        )
        print(f"Wrote {get_PPS_episodes_df.count()} rows to {PPS_EPISODE_CONCEPTS_TABLE}")
        (
            PPS_episodes_df.withColumn("LastRefreshed", refreshed)
            .write.mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(output_table)
        )
        print(f"Wrote {PPS_episodes_df.count()} rows to {output_table}")

    return PPS_episodes_df


if __name__ == "__main__" or AUTO_RUN:
    main()
