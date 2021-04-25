from src.pipeline import *

def create_animation(svg_file_path):
    logo = Logo(data_dir=svg_file_path)

    # Create input for model 1
    df = logo.create_df(pca_model="../models/pca_path_embedding.sav")