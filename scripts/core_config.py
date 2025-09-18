#!/usr/bin/env python3
from dataclasses import dataclass
import os
from typing import List

@dataclass(frozen=True)
class AppConfig:
    data_candidates: List[str]
    model_dirs: List[str]
    artifacts_root: str
    env_model_dir_var: str

    out_dir: str
    out_file: str

    app_name: str
    spark_shuffle_partitions: str
    spark_log_level: str

    @staticmethod
    def from_env() -> "AppConfig":
        data_candidates = [
            "data/clean/features.parquet",
            "data/clean/products_features.parquet",
            "data/clean/products_sample.parquet",
        ]
        model_dirs = [
            "data/out/kmeans_model_fast",
            "data/out/kmeans_model",
        ]
        return AppConfig(
            data_candidates=data_candidates,
            model_dirs=model_dirs,
            artifacts_root="artifacts/kmeans",
            env_model_dir_var="LAB7_MODEL_DIR",
            out_dir="data/out",
            out_file="predictions.parquet",
            app_name="Lab7Predict",
            spark_shuffle_partitions=os.getenv("SPARK_SHUFFLE_PARTITIONS", "16"),
            spark_log_level=os.getenv("SPARK_LOG_LEVEL", "WARN"),
        )

