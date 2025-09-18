import argparse
from scripts.core_config import AppConfig
from scripts.predict_off import PredictorApp

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=None, help="path to features parquet/csv")
    p.add_argument("--output", default=None, help="path to predictions parquet/csv")
    p.add_argument("--model", default=None, help="path to model file")
    p.add_argument("--log-level", default=None, help="logging level")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = AppConfig.from_env()
    # переопределения из аргументов командной строки
    if args.input: cfg = type(cfg)(args.input, cfg.output_path, cfg.model_path, cfg.log_level)
    if args.output: cfg = type(cfg)(cfg.input_path, args.output, cfg.model_path, cfg.log_level)
    if args.model: cfg = type(cfg)(cfg.input_path, cfg.output_path, args.model, cfg.log_level)
    if args.log_level: cfg = type(cfg)(cfg.input_path, cfg.output_path, cfg.model_path, args.log_level)
    PredictorApp(cfg).run()

