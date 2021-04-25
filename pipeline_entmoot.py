from src.pipeline import *
from src.models.train_animation_predictor import *
from src.models import config
from src.models.entmoot_functions import *


def create_animation_entmoot(svg_file_path):
    logo = Logo(data_dir=svg_file_path)

    # Create input for model 1
    df = logo.create_df(pca_model="models/pca_path_embedding.sav")

    # Retrieve model 1 prediction and extract relative position to midpoint of animated paths of SVG
    df = retrieve_m1_predictions(df)
    df = retrieve_animation_midpoints(df, drop=True)

    # Scale features
    scaler = pickle.load(open(config.scaler_path, 'rb'))
    df[config.sm_features] = scaler.transform(df[config.sm_features])

    # Extract path vectors as list
    path_vectors = df[config.sm_features].values.tolist()

    # Load ENTMOOT optimizer to data
    with open("models/entmoot_optimizer.pkl", "rb") as f:
        optimizer = pickle.load(f)

    # Load surrogate model for function evaluations
    func = SurrogateModelFNN()

    # Predict animation vectors
    an_vec_preds = []
    score_preds = []
    for i in range(len(path_vectors)):
        opt_x, opt_y = entmoot_predict(optimizer, func, path_vectors[i])
        an_vec_preds.append(opt_x)
        score_preds.append(-opt_y)

    df['animation_vector'] = an_vec_preds
    df['reward'] = score_preds


if __name__ == '__main__':
    import os
    os.chdir('..')
    create_animation_entmoot('data/svgs/logo_0.svg')


