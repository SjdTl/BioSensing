import os as os
import feat_head as feat_head

all_data = feat_head.load_dict(os.path.join(os.dir_path, "Raw_data", "raw_data.pkl"))
features = feat_head.features_db(all_data)
feat_head.save_features(features, os.path.join(os.dir_path, "Features_out", "features"))