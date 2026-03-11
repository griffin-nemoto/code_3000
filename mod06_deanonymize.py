import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    identifiers = ['zip3','age','gender']
    merge = pd.merge(anon_df,aux_df,on=identifiers,how='inner')
    
    match_anon = merge.groupby('anon_id')['anon_id'].transform('count')
    match_aux = merge.groupby('name')['name'].transform('count')
    
    unique = merge[(match_anon ==1) and (match_aux ==1)].copy()
    return unique[['anon_id','name']].rename(columns={'name':'match'})


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    if len(anon_df) == 0:
        return 0
    return len(matches_df) / len(matches_df)
