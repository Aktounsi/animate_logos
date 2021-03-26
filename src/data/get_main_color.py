def get_main_color(df, columns):
    # Todo: Get average of all paths, not relevant paths only
    for col in columns:
        df[f'avg_{col}'] = df.groupby('filename')[col].transform('mean')
    return df
