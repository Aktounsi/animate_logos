import importlib, os, glob, pickle, torch, logging
from datetime import datetime
from concurrent import futures
from tqdm import tqdm
import pandas as pd

from src.preprocessing.deepsvg.svglib.svg import SVG
from src.preprocessing.deepsvg.difflib.tensor import SVGTensor
from src.preprocessing.deepsvg.train import train
from src.preprocessing.deepsvg import utils
from src.preprocessing.deepsvg.svglib.geom import Bbox
from src.preprocessing.deepsvg.svgtensor_dataset import SVGTensorDataset, load_dataset
from src.preprocessing.deepsvg.utils.utils import batchify, linear


def save_tensor_as_pkl(svg, file_path):
    t_data = svg.copy().numericalize().to_tensor(concat_groups=False)
    t_filling = svg.to_fillings()
    tensor_dict = {'tensors': [[tensor] for tensor in t_data], 'fillings': t_filling}
    output = open(file_path, 'wb')
    pickle.dump(tensor_dict, output)
    output.close()


def preprocess_svg(svg_file, output_folder, tensor_folder, meta_data):
    filename = os.path.splitext(os.path.basename(svg_file))[0]

    svg = SVG.load_svg(svg_file)
    svg.fill_(False)
    svg.normalize()
    svg.zoom(0.9)
    svg.canonicalize()
    svg = svg.simplify_heuristic()

    svg.save_svg(os.path.join(output_folder, f"{filename}.svg"))

    save_tensor_as_pkl(svg, os.path.join(tensor_folder, f"{filename}.pkl"))

    len_groups = [path_group.total_len() for path_group in svg.svg_path_groups]

    meta_data[filename] = {
        "id": filename,
        "total_len": sum(len_groups),
        "nb_groups": len(len_groups),
        "len_groups": len_groups,
        "max_len_group": max(len_groups)
    }


def preprocessing_main(data_folder="data/svgs", workers=4):
    output_folder = f'{data_folder}_simplified'
    tensor_folder = f'{data_folder}_tensors'
    output_meta_file = f'{data_folder}_meta.csv'

    if not os.path.exists(output_folder): os.makedirs(output_folder)
    if not os.path.exists(tensor_folder): os.makedirs(tensor_folder)

    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        svg_files = glob.glob(os.path.join(data_folder, "*.svg"))
        meta_data = {}

        with tqdm(total=len(svg_files)) as pbar:
            preprocess_requests = [
                executor.submit(preprocess_svg, svg_file, output_folder, tensor_folder, meta_data)
                for svg_file in svg_files]

            for _ in futures.as_completed(preprocess_requests):
                pbar.update(1)

    df = pd.DataFrame(meta_data.values())
    df.to_csv(output_meta_file, index=False)

    logging.info("SVG Preprocessing complete.")


def save_model(model_dir, model_name, model, cfg=None, optimizer=None, scheduler_lr=None, scheduler_warmup=None,
               stats=None, train_vars=None):
    state = {
        "model": model.state_dict()
    }

    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler_lr is not None:
        state["scheduler_lr"] = scheduler_lr.state_dict()
    if scheduler_warmup is not None:
        state["scheduler_warmup"] = scheduler_warmup.state_dict()
    if cfg is not None:
        state["cfg"] = cfg.to_dict()
    if stats is not None:
        state["stats"] = stats.to_dict()
    if train_vars is not None:
        state["train_vars"] = train_vars.to_dict()

    model_path = os.path.join(model_dir, model_name)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(state, model_path)


def train_embedding_model(data_folder="./data/svgs",
                          num_epochs=50,
                          batch_size=100,
                          cfg_module="configs.deepsvg.hierarchical_ordered",
                          log_dir="./logs",
                          debug=False,
                          resume=False,
                          save=True):

    os.chdir("src/preprocessing")
    cfg = importlib.import_module(cfg_module).Config()  # _fonts
    cfg.dataloader_module = "deepsvg.svgtensor_dataset"
    os.chdir("../..")

    cfg.data_dir = f"{data_folder}_tensors/"
    cfg.meta_filepath = f"{data_folder}_meta.csv"
    cfg.batch_size = batch_size
    cfg.num_epochs = num_epochs

    model_name, experiment_name = cfg_module.split(".")[-2:]
    model = train(cfg, model_name, experiment_name, log_dir=log_dir, debug=debug, resume=resume)

    if save:
        save_model("models",
                   f"{datetime.now().strftime('%Y%m%d_%H%M')}_model_batch{batch_size}_epoch{num_epochs}_{data_folder.split('/')[-1]}.pth.tar",
                   model)

    return model


def encode_tensor(dataset, idx, model, device, cfg):
    data = dataset.get(id=idx, random_aug=False)
    model_args = batchify((data[key] for key in cfg.model_args), device)
    with torch.no_grad():
        z = model(*model_args, encode_mode=True)
        return z


def decode(z, model, do_display=True, return_svg=False, return_png=False):
    commands_y, args_y = model.greedy_sample(z=z)
    tensor_pred = SVGTensor.from_cmd_args(commands_y[0].cpu(), args_y[0].cpu())
    svg_path_sample = SVG.from_tensor(tensor_pred.data, viewbox=Bbox(256),
                                      allow_empty=True).normalize().split_paths().set_color("random")

    if return_svg:
        return svg_path_sample

    return svg_path_sample.draw(do_display=do_display, return_png=return_png)


def apply_embedding(dataset, pkl_file, pkl_list, model, device, cfg):
    filename = os.path.splitext(os.path.basename(pkl_file))[0]

    z = encode_tensor(dataset, filename, model, device, cfg).numpy()[0][0][0]

    dict_data = {"filename": filename,
                 "embedding": z}

    pkl_list.append(dict_data)


def apply_embedding_model(model_path="models/hierarchical_ordered.pth.tar",
                          cfg_module="configs.deepsvg.hierarchical_ordered",
                          data_folder="data/svgs",
                          workers=4):

    os.chdir("src/preprocessing")
    cfg = importlib.import_module(cfg_module).Config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = cfg.make_model().to(device)
    cfg.dataloader_module = "deepsvg.svgtensor_dataset"
    os.chdir("../..")

    utils.load_model(model_path, model)
    model.eval();

    cfg.data_dir = f"{data_folder}_tensors/"
    cfg.meta_filepath = f"{data_folder}_meta.csv"

    dataset = load_dataset(cfg)

    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        pkl_files = glob.glob(os.path.join(cfg.data_dir , "*.pkl"))
        pkl_list = []
        with tqdm(total=len(pkl_files)) as pbar:
            embedding_requests = [executor.submit(apply_embedding, dataset, pkl_file, pkl_list, model, device, cfg)
                                  for pkl_file in pkl_files]

            for _ in futures.as_completed(embedding_requests):
                pbar.update(1)

    df = pd.DataFrame.from_records(pkl_list, index='filename')['embedding'].apply(pd.Series)
    df.reset_index(level=0, inplace=True)

    if data_folder == "data/decomposed_svgs":
        df['animation_id'] = df['filename'].apply(lambda row: row.split('_')[-1])
        cols = list(df.columns)
        cols = [cols[0], cols[-1]] + cols[1:-1]
        df = df.reindex(columns=cols)
        df['filename'] = df['filename'].apply(lambda row: row.split('_')[0])

    output = open(f'{data_folder}_embedding.pkl', 'wb')
    pickle.dump(df, output)
    output.close()

    logging.info("Embedding complete.")

    return df


def combine_embeddings(df_svg_embedding, df_decomposed_svg_embedding):
    merged_embeddings = df_decomposed_svg_embedding.merge(df_svg_embedding, how='left', on='filename')
    combined_embedding = merged_embeddings[["filename", "animation_id"]]
    for col in range(256):
        combined_embedding[col] = merged_embeddings[str(col) + "_x"] + merged_embeddings[str(col) + "_y"]

    output = open("data/combined_embedding.pkl", 'wb')
    pickle.dump(combined_embedding, output)
    output.close()

    return combined_embedding


