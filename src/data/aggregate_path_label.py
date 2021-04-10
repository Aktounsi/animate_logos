# import packages
import pandas as pd

# import label data
firestore_data_label_path = pd.read_csv("../../data/label_path/export_table_path_label_4.4.2021.csv",
                                        skiprows=[1],
                                        dtype={"logo": object, "path_0": bool, "path_1": bool, "path_2": bool,
                                               "path_3": bool, "path_4": bool,
                                               "path_5": bool, "path_6": bool, "path_7": bool, "alias": object})

# extract id
data_label_path_id = firestore_data_label_path.assign(
    logo_id=firestore_data_label_path.logo.str.extract("(\d+)")).astype({"logo_id": 'int64'})
data_label_path_id_melt = pd.melt(data_label_path_id, id_vars=["logo", "alias", "logo_id"],
                                  value_vars=["path_0", "path_1", "path_2", "path_3", "path_4", "path_5", "path_6",
                                              "path_7"],
                                  var_name="path", value_name="animate")
data_label_path_id_melt_type = data_label_path_id_melt.assign(
    order_id=data_label_path_id_melt.path.str.extract("(\d+)")).astype({"order_id": 'int64'})

# create matching dataframe
data_matching_filename_new = pd.read_csv("../../data/label_path/label_matching_scraped_svgs.csv")
data_matching_filename_old = pd.read_csv("../../data/label_path/label_matching.csv")

data_matching_filename_new_clean = data_matching_filename_new.assign(
    filename=data_matching_filename_new.logo_id,
    logo_id=data_matching_filename_new.logo_id.str.extract("(\d+)"))

data_matching_filename_id = pd.concat([data_matching_filename_old,
                                       data_matching_filename_new_clean]).astype({"logo_id": 'int64'})

# merge data
label_path_merged = pd.merge(data_label_path_id_melt_type,
                             data_matching_filename_id, how='left', on=["logo_id", "order_id"])

# filter data
label_path_merged_filter = label_path_merged[
    label_path_merged.alias.isin(["Jani", "Jakob", "Jonathan", "Julia", "Kikipu", "Lena",
                                  "Niklas", "Ramo", "rebecca", "Sarah_240321", "Tim_Harry",
                                  "Jan B.", "Sarah", "Tim"])].dropna()

# aggregate data
path_label_final = label_path_merged_filter.groupby(by=["filename", "logo_id", "order_id", "animation_id"]).mean(
    "animate")

# export data
path_label_final.to_csv("../../data/label_path/animation_path_label.csv")