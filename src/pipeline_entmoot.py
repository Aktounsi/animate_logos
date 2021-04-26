from src.pipeline import *
from src.models.train_animation_predictor import *
from src.models import config
from xml.dom import minidom
import os

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

    # Predict and store animation vectors
    an_vec_preds = []
    score_preds = []
    for i in range(len(path_vectors)):
        opt_x, opt_y = entmoot_predict(optimizer, func, path_vectors[i])
        an_vec_preds.append(opt_x)
        score_preds.append(-opt_y)

    df['animation_vector'] = an_vec_preds
    df['reward'] = score_preds

    gb = [df.groupby('filename')[column].apply(list) for column in
          'animation_id animation_vector reward'.split()]
    svg_animations = pd.concat(gb, axis=1).reset_index()

    for i, row in svg_animations.iterrows():
        try:
            _, doc = create_animated_svg(f"data/svgs/{row['filename']}.svg", row['animation_id'], row['animation_vector'],
                                "", save=False)
        except FileNotFoundError:
            print(f"File not found: {row['filename']}")
            pass

    return doc.toprettyxml(encoding="iso-8859-1")


if __name__ == '__main__':
    os.chdir('..')
    animated_svg = create_animation_entmoot('data/svgs/logo_0.svg')



