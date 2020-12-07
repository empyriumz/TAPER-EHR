import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from data_loader.utils.vocab import Vocab
import pickle
import argparse


def main():
    """

    Will generate a dictionary as follows:
        <key> patientid : <value> list of dicts, where each dict contains admission data
                                  [
                                  {<key> feature/label name : <value> feature/label value}
                                  ]

    """
    parser = argparse.ArgumentParser(description="Generate Text+Code dataset")
    parser.add_argument(
        "-p",
        "--path",
        default=None,
        type=str,
        help="path to pandas dataframe where rows are admissions",
    )
    parser.add_argument(
        "-vp",
        "--vocab_path",
        default="",
        type=str,
        help="path to where code vocabulary are stored assumes diagnoses vocab file named as diag.vocab and cpt vocab as cpt.vocab",
    )
    parser.add_argument(
        "-s", "--save", default="./", type=str, help="path to save pkl files"
    )
    parser.add_argument(
        "-sc",
        "--short_code",
        default=False,
        action="store_true",
        help="flag for using short codes ",
    )
    parser.add_argument(
        "-diag",
        "--diagnoses",
        default=False,
        action="store_true",
        help="flag for including diagnoses codes",
    )
    parser.add_argument(
        "-proc",
        "--procedures",
        default=False,
        action="store_true",
        help="flag for including procedures codes",
    )
    parser.add_argument(
        "-med",
        "--medications",
        default=False,
        action="store_true",
        help="flag for including medication codes",
    )
    parser.add_argument(
        "-cpt",
        "--cpts",
        default=False,
        action="store_true",
        help="flag for including cpt codes",
    )
    args = parser.parse_args()
    df = pd.read_pickle(args.path)
    df_orig = df
    # remove organ donor admissions
    if "DIAGNOSIS" in df.columns:
        REMOVE_DIAGNOSIS = ~(
            (df["DIAGNOSIS"] == "ORGAN DONOR ACCOUNT")
            | (df["DIAGNOSIS"] == "ORGAN DONOR")
            | (df["DIAGNOSIS"] == "DONOR ACCOUNT")
        )
        df = df[REMOVE_DIAGNOSIS]

    df = df[~df["ICD9_CODE"].isna()]  # drop patients with no icd9 code

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
    df["AGE"] = df["AGE"].astype(int)
    los_bins = [1, 2, 3, 4, 5, 6, 7, 8, 14, float("inf")]
    los_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # LOS: Length of stay
    df["LOS"] = pd.cut(df["LOS"], bins=los_bins, labels=los_labels)

    diag_vocab = Vocab()
    cpt_vocab = Vocab()
    med_vocab = Vocab()
    proc_vocab = Vocab()

    if args.vocab_path != "":
        # to use below checkout https://github.com/sajaddarabi/HCUP-US-EHR
        if args.diagnoses:
            diag_vocab._build_from_file(os.path.join(args.vocab_path, "diag.vocab"))
        if args.cpts:
            cpt_vocab._build_from_file(os.path.join(args.vocab_path, "cpt.vocab"))
        # if (args.procedures):
        #    proc_vocab._build_from_file(os.path.join(args.vocab_path, 'proc.vocab'))
        # if (args.med):
        # med_vocab._build_from_file(os.path.join(args.vocab_path, 'med.vocab'))

    data = {}
        
    pids = list(set(df["SUBJECT_ID"]))
    try:
        for pid in tqdm(pids):
            pid_df = df[df["SUBJECT_ID"] == pid]
            pid_df = pid_df.sort_values("ADMITTIME").reset_index()
            data[pid] = []
            time = 0
            for _, r in pid_df.iterrows():
                # filt notes prior to n days and concatenate them
                # leave discharge summary seperate
                admit_data = {}
                demographics = [
                    r["AGE"],
                    r["GENDER"]
                ]
                # one-hot encoding for MARITAL, LAST_CAREUNIT and ETHNICITY
                marital_status = np.zeros(
                    (demographic_cols["MARITAL_STATUS"].size,), dtype=int
                )
                marital_status[r["MARITAL_STATUS"]] = 1
                demographics += list(marital_status)
                
                icu_unit = np.zeros(
                    (demographic_cols["LAST_CAREUNIT"].size,), dtype=int
                )
                icu_unit[r["LAST_CAREUNIT"]] = 1
                demographics += list(icu_unit)
                
                ethnicity = np.zeros((demographic_cols["ETHNICITY"].size,), dtype=int)
                ethnicity[r["ETHNICITY"]] = 1
                demographics += list(ethnicity)
                
                admit_data["demographics"] = demographics
                if args.diagnoses:
                    diagnosis_codes = r["ICD9_CODE"]
                    try:
                        dtok = diag_vocab.convert_to_ids(diagnosis_codes, "D", True)
                    except:
                        dtok = [0]

                if args.procedures:
                    proc_codes = r["ICD9_CODE_PROCEDURE"]
                    try:
                        ptok = proc_vocab.convert_to_ids(
                            proc_codes, "P", short_icd9=True
                        )
                    except:
                        ptok = [0]

                if args.medications:
                    med_codes = r["NDC"]
                    try:
                        mtok = med_vocab.convert_to_ids(med_codes, "M", short_icd9=True)
                    except:
                        mtok = [0]

                if args.cpts:
                    cpt_codes = r["CPT_CD"]
                    try:
                        ctok = cpt_vocab.convert_to_ids(cpt_codes, "C", short_icd9=True)
                    except:
                        ctok = [0]

                admit_data["diagnoses"] = dtok
                admit_data["procedures"] = ptok
                admit_data["medications"] = mtok
                admit_data["cptproc"] = ctok
                
                time += r["TIMEDELTA"]
                admit_data["timedelta"] = time
                admit_data["los"] = r["LOS"]
                admit_data["readmission"] = r["readmission_label"]
                admit_data["mortality"] = r["DEATHTIME"] == r["DEATHTIME"]
                data[pid].append(admit_data)

    except Exception as error:
        print(error)
        import pdb

        pdb.set_trace()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    pids = list(data.keys())
    flatten = lambda x: [item for sublist in x for item in sublist]

    data_info = {}
    data_info["num_patients"] = len(pids)
 
    if args.diagnoses:
        num_icd9_codes = len(set(flatten(df_orig["ICD9_CODE"].dropna())))
        data_info["num_icd9_codes"] = num_icd9_codes

    if args.procedures:
        num_proc_codes = len(set(flatten(df_orig["ICD9_CODE_PROCEDURE"].dropna())))
        data_info["num_proc_codes"] = num_proc_codes

    if args.medications:
        num_med_codes = len(set(flatten(df_orig["NDC"].dropna())))
        data_info["num_med_codes"] = num_med_codes
    data_info["demographics_shape"] = len(data[pids[0]][0]["demographics"])
    data_info["demographic_cols"] = demographic_cols
    data_info["total_codes"] = (
        data_info["num_icd9_codes"]
        + data_info["num_proc_codes"]
        + data_info["num_med_codes"]
    )

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    with open(os.path.join(args.save, "data.pkl"), "wb") as handle:
        data_dict = {}
        data_dict["info"] = data_info
        data_dict["data"] = data
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(args.save, "cpt_vocab.pkl"), "wb") as handle:
        pickle.dump(cpt_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.save, "diag_vocab.pkl"), "wb") as handle:
        pickle.dump(diag_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.save, "med_vocab.pkl"), "wb") as handle:
        pickle.dump(med_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.save, "proc_vocab.pkl"), "wb") as handle:
        pickle.dump(proc_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()