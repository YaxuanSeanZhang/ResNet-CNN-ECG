import os
import pandas as pd


def check_user():
    """
    Check if a user can run this analysis.py and provide an error if not
    Note, this is a hacky convenience function, not an actual security check, that is handled through
    actual measures, and no one has access to files they should not have.
    Changing the environment variable will not allow you to do anything useful.
    :return:
    """
    if os.getenv("CSCI5525_EXTERNAL") != "True":
        raise ValueError("You are not authorized to run this analysis.py. Please contact the author for more information.")



def fetch_cohort(schema):
    """
    Return a dataframe containing the cohort.
    :return:
    """
    from external_validation.connection import connect_db

    query = "select * from {schema}.qhs_ecg_volume_cohort".format(schema=schema)
    con = connect_db()
    cohort = pd.read_sql_query(query, con)
    return cohort


def fetch_diagnoses(schema):
    """
    Fetch diagnoses for selected cohort
    Cohort was previously defined and stored in the database
    :return:
    """

    from external_validation.connection import connect_db


    query = """ select
        patient.patient_clinic_number as clinic,
        fact_dx.diagnosis_method_code as dx_method,
        fact_dx.diagnosis_code as dx_code,
        fact_dx.diagnosis_dtm as dx_dtm,
        dim_dx_code.diagnosis_name as dx_name,
        dim_dx_code.diagnosis_description as dx_description
    from edtwh.fact_diagnosis as fact_dx
      inner join {schema}.dim_patient as patient
        on patient.patient_dk = fact_dx.patient_dk
      inner join {schema}.qhs_ecg_volume_cohort as cohort
          on cohort.mayo_clinic_nbr = patient.patient_clinic_number
      inner join {schema}.dim_diagnosis_code as dim_dx_code
        on fact_dx.diagnosis_code_dk = dim_dx_code.diagnosis_code_dk
    where
      fact_dx.diagnosis_code in ('426.9', 'I45.9')""".format(schema=schema)

    con = connect_db()
    diagnoses = pd.read_sql_query(query, con)
    return diagnoses