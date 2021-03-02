import os
import pandas as pd
from src.models.train_model_head import *

os.chdir('..')

combined_embedding = pd.read_pickle('./data/combined_embedding.pkl')

train_model_head(svg_dataset=combined_embedding, num_agents=5, top_parent_limit=3, generations=2)
