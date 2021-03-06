import numpy as np
import pandas as pd
import os
import argparse
from utils.data_utils import *
from tqdm import tqdm
import itertools

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
    parser.add_argument(
        "-min-adm",
        "--min_admission",
        default=1,
        type=int,
        help="minimum number of admissions for each patient",
    )
    args = parser.parse_args()

    # format date time
    df_adm = pd.read_csv(os.path.join(args.path, "ADMISSIONS.csv"))
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
    
    # remove patients with admissions < min_adm
    if args.min_admission > 1:
        df_adm = remove_min_admissions(df_adm, min_admits=args.min_admission)
    
    # Adding clinical codes to dataset
    # add diagnoses
    code = "ICD9_CODE"
    diagnoses = read_icd_diagnoses_table(args.path)
    diagnoses = filter_codes(diagnoses, code=code, min_=10)
    
    # add procedures
    procedures = read_icd_procedures_table(args.path)
    procedures = filter_codes(procedures, code=code, min_=10)
   
    with open("vocab/icd-map.pkl", "rb") as f:
        dic_icd = pickle.load(f)
    with open("vocab/proc-map.pkl", "rb") as f:
        dic_proc = pickle.load(f)
    
    diagnoses['ICD9_SHORT'] = diagnoses['ICD9_CODE'].apply(lambda x: dic_icd[x])
    # adding a constant to procedure code mapping to avoid conflicts
    mapping_shift = max(dic_icd.values())
    procedures['ICD9_CODE'] = procedures['ICD9_CODE'].astype(str)
    procedures['PROC_SHORT'] = procedures['ICD9_CODE'].apply(lambda x: dic_proc[x]+mapping_shift 
                                                             if x in dic_proc.keys() else None)
    procedures.dropna(inplace=True)
    
    diagnoses = group_by_return_col_list(
        diagnoses, ["SUBJECT_ID", "HADM_ID"], 'ICD9_SHORT')
    procedures = group_by_return_col_list(
        procedures, ["SUBJECT_ID", "HADM_ID"], 'PROC_SHORT'
    )
    
    # ICU info
    patients = read_patients_table(args.path)
    stays = read_icustays_table(args.path)
    stays = stays.merge(patients, how='inner', left_on=['SUBJECT_ID'], right_on=["SUBJECT_ID"])
    cols = ["SUBJECT_ID", "HADM_ID"]
    stays = stays.merge(diagnoses, how="inner", left_on=cols, right_on=cols)
    stays = stays.merge(procedures, how="inner", left_on=cols, right_on=cols)
    stays = add_age_to_icustays(stays)
    
    df_adm = pd.merge(
        df_adm, stays, on=["SUBJECT_ID", "HADM_ID"], how="inner"
    )

    df_adm["ADMITTIME_C"] = df_adm.ADMITTIME.apply(
        lambda x: str(x).split(" ")[0]
    )
    df_adm["ADMITTIME_C"] = pd.to_datetime(
        df_adm.ADMITTIME_C, format="%Y-%m-%d", errors="coerce"
    )
    
    df_adm = compute_time_delta(df_adm)
    # only retain the first row for each HADM ID
    df = df_adm.groupby("HADM_ID").first()
    # remove organ donor admissions
    if "DIAGNOSIS" in df.columns:
        REMOVE_DIAGNOSIS = ~(
            (df["DIAGNOSIS"] == "ORGAN DONOR ACCOUNT")
            | (df["DIAGNOSIS"] == "ORGAN DONOR")
            | (df["DIAGNOSIS"] == "DONOR ACCOUNT")
        )
        df = df[REMOVE_DIAGNOSIS]
        
    # begin demographic info processing
    demographic_cols = {
        "AGE": [],
        "GENDER": [],
        "LAST_CAREUNIT": [],
        "MARITAL_STATUS": [],
        "ETHNICITY": [],
        "DISCHARGE_LOCATION": [],
    }

    df["MARITAL_STATUS"], demographic_cols["MARITAL_STATUS"] = pd.factorize(
        df["MARITAL_STATUS"]
    )
    df["ETHNICITY"], demographic_cols["ETHNICITY"] = pd.factorize(df["ETHNICITY"])

    df["DISCHARGE_LOCATION"], demographic_cols["DISCHARGE_LOCATION"] = pd.factorize(
        df["DISCHARGE_LOCATION"]
    )
    df["LAST_CAREUNIT"], demographic_cols["LAST_CAREUNIT"] = pd.factorize(
        df["LAST_CAREUNIT"]
    )
    df["GENDER"], demographic_cols["GENDER"] = pd.factorize(df["GENDER"])
    los_bins = [1, 2, 3, 4, 5, 6, 7, 8, 14, float("inf")]
    los_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # LOS: Length of stay
    df["LOS"] = pd.cut(df["LOS"], bins=los_bins, labels=los_labels)
    
    data = {}
    pids = list(set(df["SUBJECT_ID"]))
    for i, pid in enumerate(tqdm(pids)):
        pid_df = df[df["SUBJECT_ID"] == pid]
        pid_df = pid_df.sort_values("ADMITTIME").reset_index()
        # change key from subject ID to the integer number
        data[i] = []
        time = 0
        tmp = pid_df.iloc[0]
        # first part of admit data, which is same regardless of visits 
        admit_data = {}
        # one-hot encoding for MARITAL, LAST_CAREUNIT and ETHNICITY   
        demographics = [tmp["AGE"], tmp["GENDER"]]
        marital_status = np.zeros(
                    (demographic_cols["MARITAL_STATUS"].size,), dtype=int
                )
        marital_status[tmp["MARITAL_STATUS"]] = 1
        demographics += list(marital_status)
        icu_unit = np.zeros(
                    (demographic_cols["LAST_CAREUNIT"].size,), dtype=int
                )
        icu_unit[tmp["LAST_CAREUNIT"]] = 1
        demographics += list(icu_unit)

        ethnicity = np.zeros((demographic_cols["ETHNICITY"].size,), dtype=int)
        ethnicity[tmp["ETHNICITY"]] = 1
        demographics += list(ethnicity)
        admit_data["demographics"] = demographics
        admit_data["readmission"] = tmp["readmission_label"]
        admit_data["mortality"] = tmp["DEATHTIME"] == tmp["DEATHTIME"]
        data[i].append(admit_data)
        for _, r in pid_df.iterrows():
            admit_data = {}
            # gather the medical codes for each visit              
            admit_data["diagnoses"] = r['ICD9_SHORT']
            admit_data["procedures"] = r['PROC_SHORT'] 
            admit_data["los"] = r["LOS"]           
            time += r["TIMEDELTA"]
            admit_data["timedelta"] = time          
            data[i].append(admit_data)

    pids = list(data.keys())    
    data_info = {}
    data_info["num_patients"] = len(pids)
    num_icd9_codes = max(dic_icd.values()) + 1 # mapping start with 0
    num_proc_codes = max(dic_proc.values()) + 1
    data_info["num_icd9_codes"] = num_icd9_codes
    data_info["num_proc_codes"] = num_proc_codes
    data_info["num_med_codes"] = 0
    data_info["num_cpt_codes"] = 0
    data_info["demographics_shape"] = len(data[pids[0]][0]["demographics"])
    data_info["demographic_cols"] = demographic_cols
    
    if not os.path.isdir(args.save):
        os.makedirs(args.save)
    
    with open(os.path.join(args.save, "data_no_grouped.pkl"), "wb") as handle:
        data_dict = {}
        data_dict["info"] = data_info
        data_dict["data"] = data
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)