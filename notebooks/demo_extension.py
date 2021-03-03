from src.models.train_model_head import *
import os
import pandas as pd

os.chdir('..')

combined_embedding = pd.read_pickle('./data/combined_embedding.pkl')
combined_embedding['animation_id'] = pd.to_numeric(combined_embedding['animation_id'])

# reduce to easy SVGs (<= 5 paths) in order to speed up training
easy_logos = set(np.unique(combined_embedding['filename'])[combined_embedding.groupby(['filename']).max()['animation_id']<=0])
emb_red = combined_embedding[combined_embedding['filename'].isin(easy_logos)].sort_values(['filename', 'animation_id'])
emb_red.reset_index(drop=True, inplace=True)

top_model = train_model_head(svg_dataset=emb_red, num_agents=2, top_parent_limit=2, generations=1)
torch.save(top_model, './models/best_model_head.pkl')
torch.save(top_model.state_dict(), './models/best_model_head_state_dict.pth')


