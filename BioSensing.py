# Top level document
import os as os


from Features.Features import feat_head

# Current location
dir_path = os.path.dirname(os.path.realpath(__file__))
# Import wesad data
all_data = feat_head.load_dict(os.path.join(dir_path, "Features", "Raw_data", "raw_data.pkl"))
# Determine features
features = feat_head.features_db(all_data)
# Intermediate save
feat_head.save_features(features, os.path.join(dir_path, "Features", "Features_out", "features"))
# Pass over to classification
