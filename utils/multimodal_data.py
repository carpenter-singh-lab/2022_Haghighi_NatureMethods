import pandas as pd
import numpy as np


def load_l1000(profiles_file):
    l1k = pd.read_csv(profiles_file)
    l1k = l1k[ ["allele"] + list(str(x) for x in range(0,l1k.shape[1]-1)) ]
    l1k = l1k.rename(index=str, columns={"allele": "Allele"})
    return l1k


def load_cell_painting(morphology_file, profiles_file, aggregate_replicates=False):
    morphology = pd.read_csv(morphology_file)
    treatments = morphology["Metadata_broad_sample"].unique()
    print("Treatments:", len(treatments))
    FF = 23 # First feature in data frame
    profiles = pd.read_csv(profiles_file)

    mdata = pd.merge(morphology[morphology.columns[0:FF]], profiles, on=["Metadata_Plate", "Metadata_Well"])
    mdata = mdata[~mdata["Training"]]
    mdata = mdata.reset_index(drop=True)
    mdata.loc[mdata["Allele"].isnull(), "Allele"] = "EMPTY"
    cp = mdata[ ["Allele"] + [str(x) for x in range(0,255)] ]

    if aggregate_replicates:
        cp = cp.groupby("Allele").mean().reset_index()
    return cp


def align_profiles(l1k, cp, sample=0):
    print("From:", l1k.shape, cp.shape)
    common_alleles = set(cp["Allele"].unique()).intersection( l1k["Allele"].unique() )
    l1k = l1k[l1k["Allele"].isin(common_alleles)]
    l1k = l1k.sort_values(by="Allele")

    cp = cp[cp["Allele"].isin(common_alleles)]
    cp = cp.sort_values(by="Allele")

    if sample > 0:
        grouped = l1k.groupby("Allele")
        l1k_sample = grouped.apply(lambda x: x.sample(n=sample))
        l1k_sample = l1k_sample.reset_index(drop=True)
        l1k_sample = l1k_sample.sort_values(by="Allele")

        grouped = cp.groupby("Allele")
        cp_sample = grouped.apply(lambda x: x.sample(n=sample))
        cp_sample = cp_sample.reset_index(drop=True)
        cp_sample = cp_sample.sort_values(by="Allele")
        l1k, cp = l1k_sample, cp_sample

    print("To:", l1k.shape, cp.shape)
    return l1k, cp

