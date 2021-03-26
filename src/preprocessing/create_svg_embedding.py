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
from src.preprocessing.deepsvg.svg_dataset import SVGDataset, load_dataset  # SVG Dataset
from src.preprocessing.deepsvg.utils.utils import batchify, linear

# Reproducibility
utils.set_seed(42)


def _load_model(model_path="models/hierarchical_ordered.pth.tar",
                cfg_module="configs.deepsvg.hierarchical_ordered",
                data_folder="data/svgs"):
    os.chdir("src/preprocessing")
    cfg = importlib.import_module(cfg_module).Config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = cfg.make_model().to(device)
    cfg.dataloader_module = "deepsvg.svg_dataset"
    os.chdir("../..")

    utils.load_model(model_path, model)
    model.eval();

    cfg.data_dir = f"{data_folder}/"
    cfg.meta_filepath = f"{data_folder}_meta.csv"

    dataset = load_dataset(cfg)

    return dataset, model, device, cfg


def _save_model(model_dir, model_name, model, cfg=None, optimizer=None, scheduler_lr=None, scheduler_warmup=None,
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
    cfg.dataloader_module = "deepsvg.svgdataset"
    os.chdir("../..")

    cfg.data_dir = f"{data_folder}/"
    cfg.meta_filepath = f"{data_folder}_meta.csv"
    cfg.batch_size = batch_size
    cfg.num_epochs = num_epochs

    model_name, experiment_name = cfg_module.split(".")[-2:]
    model = train(cfg, model_name, experiment_name, log_dir=log_dir, debug=debug, resume=resume)

    if save:
        date_time = datetime.now().strftime('%Y%m%d_%H%M')
        data = data_folder.split('/')[-1]
        _save_model("models",
                    f"{date_time}_model_batch{batch_size}_epoch{num_epochs}_{data}.pth.tar",
                    model)

    return model


def encode_svg(filename,
               model_path="models/hierarchical_ordered.pth.tar",
               cfg_module="configs.deepsvg.hierarchical_ordered",
               data_folder="data/svgs"):
    dataset, model, device, cfg = _load_model(model_path=model_path, cfg_module=cfg_module, data_folder=data_folder)

    return _encode_svg(dataset, filename, model, device, cfg)


def apply_embedding_model_to_svgs(model_path="models/hierarchical_ordered.pth.tar",
                                  cfg_module="configs.deepsvg.hierarchical_ordered",
                                  data_folder="data/svgs",
                                  workers=4,
                                  save=True):
    dataset, model, device, cfg = _load_model(model_path=model_path, cfg_module=cfg_module, data_folder=data_folder)

    # TODO: Change data_dir in _load_model (new parameter)
    cfg.data_dir = data_folder

    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        svg_files = glob.glob(os.path.join(cfg.data_dir, "*.svg"))
        svg_list = []
        with tqdm(total=len(svg_files)) as pbar:
            embedding_requests = [executor.submit(_apply_embedding_to_svg, dataset, svg_file, svg_list, model, device, cfg)
                                  for svg_file in svg_files]

            for _ in futures.as_completed(embedding_requests):
                pbar.update(1)

    df = pd.DataFrame.from_records(svg_list, index='filename')['embedding'].apply(pd.Series)
    df.reset_index(level=0, inplace=True)

    if data_folder == "data/decomposed_svgs":
        df['animation_id'] = df['filename'].apply(lambda row: row.split('_')[-1])
        cols = list(df.columns)
        cols = [cols[0], cols[-1]] + cols[1:-1]
        df = df.reindex(columns=cols)
        df['filename'] = df['filename'].apply(lambda row: "_".join(row.split('_')[0:-1]))

    if save:
        model = model_path.split("/")[-1].replace("_svgs.pth.tar", "").replace("_decomposed", "")
        data = data_folder.split("/")[-1]
        output = open(f'data/{model}_{data}_embedding.pkl', 'wb')
        pickle.dump(df, output)
        output.close()

    logging.info("Embedding complete.")

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def _apply_embedding_to_svg(dataset, svg_file, svg_list, model, device, cfg):
    z = _encode_svg(dataset, svg_file, model, device, cfg).numpy()[0][0][0]
    filename = os.path.splitext(os.path.basename(svg_file))[0]

    dict_data = {"filename": filename,
                 "embedding": z}

    svg_list.append(dict_data)


def _encode_svg(dataset, filename, model, device, cfg):
    svg = SVG.load_svg(filename)
    try:
        svg = dataset.simplify(svg)
        svg = dataset.preprocess(svg)
        data = dataset.get(svg=svg)
    except Exception as e:
        svg = svg.canonicalize(normalize=True)
        svg = dataset.preprocess(svg)
        data = dataset.get(svg=svg)
        #print(f"{filename}: Simplify failed {e}")
    model_args = batchify((data[key] for key in cfg.model_args), device)
    with torch.no_grad():
        z = model(*model_args, encode_mode=True)
        return z


def combine_embeddings(df_svg_embedding,
                       df_decomposed_svg_embedding,
                       model_path="models/hierarchical_ordered.pth.tar",
                       save=True):
    # TODO: model_path is only needed for naming. Not very nice. Change later?
    merged_embeddings = df_decomposed_svg_embedding.merge(df_svg_embedding, how='left', on='filename')
    combined_embedding = merged_embeddings[["filename", "animation_id"]]
    for col in range(256):
        combined_embedding[col] = merged_embeddings[str(col) + "_x"] + merged_embeddings[str(col) + "_y"]

    if save:
        model = model_path.split("/")[-1].replace("_svgs.pth.tar", "").replace("_decomposed", "")
        output = open(f"data/{model}_combined_embedding.pkl", 'wb')
        pickle.dump(combined_embedding, output)
        output.close()

    return combined_embedding


def decode_z(z,
             model_path="models/hierarchical_ordered.pth.tar",
             cfg_module="configs.deepsvg.hierarchical_ordered",
             data_folder="data/svgs",
             do_display=True,
             return_svg=False,
             return_png=False):
    dataset, model, device, cfg = _load_model(model_path=model_path, cfg_module=cfg_module, data_folder=data_folder)

    commands_y, args_y = model.greedy_sample(z=z)
    tensor_pred = SVGTensor.from_cmd_args(commands_y[0].cpu(), args_y[0].cpu())
    svg_path_sample = SVG.from_tensor(tensor_pred.data, viewbox=Bbox(256),
                                      allow_empty=True).normalize().split_paths().set_color("random")

    if return_svg:
        return svg_path_sample

    return svg_path_sample.draw(do_display=do_display, return_png=return_png)
