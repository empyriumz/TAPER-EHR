import numpy as np
import pandas as pd
import os
import argparse
from utils.data_utils import *


if __name__ == "__main__":

    """
    Generate dataframe where each row represents patient admission
    """

    parser = argparse.ArgumentParser(description="Process Mimic-iii CSV Files")
    parser.add_argument(
        "-p", "--path", default=None, type=str, help="path to mimic-iii csvs"
    )
    parser.add_argument(
        "-s", "--save", default=None, type=str, help="path to dump output"
    )
    args = parser.parse_args()

    patients = read_patients_table(args.path)

    # format date time
    df_adm = pd.read_csv(os.path.join(args.path, "ADMISSIONS.csv.gz"))
    df_adm.ADMITTIME = pd.to_datetime(
        df_adm.ADMITTIME, format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    df_adm.DISCHTIME = pd.to_datetime(
        df_adm.DISCHTIME, format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    df_adm.DEATHTIME = pd.to_datetime(
        df_adm.DEATHTIME, format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )

    df_adm = df_adm.sort_values(["SUBJECT_ID", "ADMITTIME"])
    df_adm = df_adm.reset_index(drop=True)
    # one task in the paper is to predict re-admission within 30 days
    df_adm["NEXT_ADMITTIME"] = df_adm.groupby("SUBJECT_ID").ADMITTIME.shift(periods=-1)
    df_adm["NEXT_ADMISSION_TYPE"] = df_adm.groupby("SUBJECT_ID").ADMISSION_TYPE.shift(
        periods=-1
    )

    rows = df_adm.NEXT_ADMISSION_TYPE == "ELECTIVE"
    df_adm.loc[rows, "NEXT_ADMITTIME"] = pd.NaT
    df_adm.loc[rows, "NEXT_ADMISSION_TYPE"] = np.NaN

    df_adm = df_adm.sort_values(["SUBJECT_ID", "ADMITTIME"])

    # When we filter out the "ELECTIVE",
    # we need to correct the next admit time
    # for these admissions since there might
    # be 'emergency' next admit after "ELECTIVE"
    df_adm[["NEXT_ADMITTIME", "NEXT_ADMISSION_TYPE"]] = df_adm.groupby(["SUBJECT_ID"])[
        ["NEXT_ADMITTIME", "NEXT_ADMISSION_TYPE"]
    ].fillna(method="bfill")
    df_adm["DAYS_NEXT_ADMIT"] = (
        df_adm.NEXT_ADMITTIME - df_adm.DISCHTIME
    ).dt.total_seconds() / (24 * 60 * 60)
    df_adm["readmission_label"] = (df_adm.DAYS_NEXT_ADMIT < 30).astype("int")
    ### filter out newborn and death
    df_adm = df_adm[df_adm["ADMISSION_TYPE"] != "NEWBORN"]
    df_adm["DURATION"] = (
        df_adm["DISCHTIME"] - df_adm["ADMITTIME"]
    ).dt.total_seconds() / (24 * 60 * 60)

    # Adding clinical codes to dataset
    # add diagnoses
    code = "ICD9_CODE"
    diagnoses = read_icd_diagnoses_table(args.path)
    diagnoses = filter_codes(diagnoses, code=code)
    diagnoses = group_by_return_col_list(diagnoses, ["SUBJECT_ID", "HADM_ID"], code)

    # add procedures
    procedures = read_icd_procedures_table(args.path)
    procedures = filter_codes(procedures, code=code)
    procedures = group_by_return_col_list(
        procedures, ["SUBJECT_ID", "HADM_ID"], code, "ICD9_CODE_PROCEDURE"
    )

    # add cptevents
    code = "CPT_CD"
    cptevents = read_cptevents_table(args.path)
    cptevents = filter_codes(cptevents, code=code)
    cptevents = group_by_return_col_list(cptevents, ["SUBJECT_ID", "HADM_ID"], code)

    # add prescriptions
    code = "NDC"
    prescriptions = read_prescriptions_table(args.path)
    prescriptions = filter_codes(prescriptions, code=code)
    prescriptions = group_by_return_col_list(
        prescriptions, ["SUBJECT_ID", "HADM_ID"], code
    )

    stays = read_icustays_table(args.path)
    stays = stays.merge(patients, how='inner', left_on=['SUBJECT_ID'], right_on=["SUBJECT_ID"])
    cols = ["SUBJECT_ID", "HADM_ID"]
    stays = stays.merge(diagnoses, how="left", left_on=cols, right_on=cols)
    stays = stays.merge(cptevents, how="left", left_on=cols, right_on=cols)
    stays = stays.merge(prescriptions, how="left", left_on=cols, right_on=cols)
    stays = stays.merge(procedures, how="left", left_on=cols, right_on=cols)

    stays = add_age_to_icustays(stays)

    df_adm = pd.merge(
        df_adm, stays, on=["SUBJECT_ID", "HADM_ID"], how="left"
    )
    filt = df_adm["ICD9_CODE"].isna() & df_adm["CPT_CD"].isna()
    df_adm = df_adm[~filt]

    df_adm["ADMITTIME_C"] = df_adm.ADMITTIME.apply(
        lambda x: str(x).split(" ")[0]
    )
    df_adm["ADMITTIME_C"] = pd.to_datetime(
        df_adm.ADMITTIME_C, format="%Y-%m-%d", errors="coerce"
    )
    
    df_adm = compute_time_delta(df_adm)
    # only retain the first row for each HADM ID
    df_partial = df_adm.groupby("HADM_ID").first()
    if not os.path.isdir(args.save):
        os.makedirs(args.save)
    
    
    df_partial.to_pickle(os.path.join(args.save, "no_text_data.pkl"))
    df_adm.to_pickle(os.path.join(args.save, "no_text_data_all.pkl"))