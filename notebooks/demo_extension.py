import os, sys
import pandas as pd

os.chdir('..')

from src.models.train_model_head import *


def main():
    combined_embedding = pd.read_pickle('./data/combined_embedding.pkl')
    combined_embedding['animation_id'] = pd.to_numeric(combined_embedding['animation_id'])

    svg_embedding = pd.read_pickle('./data/svgs_embedding.pkl')

    # reduce to easy SVGs (<= 8 paths) in order to speed up training
    easy_logos = set(np.unique(combined_embedding['filename'])[combined_embedding.groupby(['filename']).max()['animation_id']<=7])
    emb_red = combined_embedding[combined_embedding['filename'].isin(easy_logos)].sort_values(['filename', 'animation_id'])
    emb_red.reset_index(drop=True, inplace=True)

    top_model = train_model_head(path_level_dataset=emb_red, svg_level_dataset=svg_embedding, num_agents=5, top_parent_limit=3, generations=2, out_size=13)
    torch.save(top_model, './models/best_model_head.pkl')
    torch.save(top_model.state_dict(), './models/best_model_head_state_dict.pth')


if __name__ == '__main__':
    main()
