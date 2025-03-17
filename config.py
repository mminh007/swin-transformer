import yaml
import os
import argparse

def setup_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config-file", type=str, help="path to config file")
    parser.add_argument("--mode", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--name", type=str, help="using save model")
    parser.add_argument("--data", type=str, help="data using train model")
    parser.add_argument("--in-chans", type=int, help="dimension of input")
    parser.add_argument("--imgsz", type=int, help="size of input")
    parser.add_argument("--labels", type=int, help="number of labels")
    
    parser.add_argument("--patch-size", type=int)
    parser.add_argument("--nheads", nargs='+', type=int)
    parser.add_argument("--embed-dim", type=int)
    parser.add_argument("--depths", nargs='+', type=int)
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--mlp-ratio", type=int)
    parser.add_argument("--qkv-bias", type=bool)
    parser.add_argument("--qk-scale", type=bool)
    parser.add_argument("--ape", type=bool)
    parser.add_argument("--use-rel-pos", type=bool)
    parser.add_argument("--drop-out", type=float)
    parser.add_argument("--norm-eps", type=float)

    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--dtype", type=str)
    parser.add_argument("--devices", type=str)
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--beta1", type=float)
    parser.add_argument("--beta2", type=float)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--outputs-dir", type=str)

    return parser


def update_config(args: argparse.Namespace):
    if not args.config_file:
        return args
    
    cfg_path = args.config_file + ".yaml" if not args.config_file.endswith(".yaml") else args.config_file

    with open(cfg_path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    
    for key, value in data.items():
        if getattr(args, key) is None:
            if key in ["nheads", "depths"] and isinstance(value, str):
                value = list(map(int, value.split()))
            setattr(args, key, value)

    # config_args = argparse.Namespace(**data)
    # args = parser.parse_args(namespace=config_args)
    

    return args

