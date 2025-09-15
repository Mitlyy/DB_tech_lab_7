#!/usr/bin/env python3
import os
import sys
from typing import Optional

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

try:
    from pyspark.ml.clustering import KMeansModel
    from pyspark.ml import PipelineModel
except Exception as e:
    print("[FATAL] Требуется PySpark. Установи pyspark и запусти снова.")
    raise

DATA_CANDIDATES = [
    "data/clean/features.parquet",            
    "data/clean/products_features.parquet",   
    "data/clean/products_sample.parquet",     
]

MODEL_DIRS = [
    "data/out/kmeans_model_fast",             
    "data/out/kmeans_model",                  
]

RAW_COLS = [
    "energy_100g",
    "proteins_100g",
    "fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "fiber_100g",
    "salt_100g",
]


def pick_first_existing(paths) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def find_data() -> str:
    p = pick_first_existing(DATA_CANDIDATES)
    if not p:
        raise SystemExit(
            "[FATAL] Не найден входной parquet. Ожидал один из:\n  - "
            + "\n  - ".join(DATA_CANDIDATES)

        )
    return p


def find_model_dir() -> str:
    env = os.getenv("LAB7_MODEL_DIR", "").strip()
    if env and os.path.isdir(env):
        return env

    p = pick_first_existing(MODEL_DIRS)
    if p:
        return p

    root = "artifacts/kmeans"
    if os.path.isdir(root):
        cands = [
            os.path.join(root, d)
            for d in os.listdir(root)
            if d.startswith("model_k")
        ]
        cands = [d for d in cands if os.path.isdir(d)]
        if cands:
            return sorted(cands)[-1]

    raise SystemExit(
        "[FATAL] Не найдена модель. Ожидал одну из:\n  - "
        + "\n  - ".join(MODEL_DIRS)
        
    )


def main():
    spark = (
        SparkSession.builder.appName("Lab7Predict")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    in_path = find_data()
    model_dir = find_model_dir()

    print(f"[INFO] Using data:  {in_path}")
    print(f"[INFO] Using model: {model_dir}")

    df = spark.read.parquet(in_path)
    cols = set(df.columns)

    out_dir = "data/out"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "predictions.parquet")

    has_features = "features" in cols
    has_raw = any(c in cols for c in RAW_COLS)

    model = None
    used_mode = None

    if has_features:
        try:
            model = KMeansModel.load(model_dir)
            used_mode = "kmeans_model"
        except Exception:
            model = None

    if model is None and has_raw:
        try:
            model = PipelineModel.load(model_dir)
            used_mode = "pipeline_model"
        except Exception:
            model = None

    if model is None:
        msg = "[FATAL] Не удалось загрузить ни KMeansModel, ни PipelineModel.\n"
        if not has_features and not has_raw:
            msg += (
                "В parquet нет ни 'features', ни сырых колонок "
                f"({', '.join(RAW_COLS)}). "
                
            )
        raise SystemExit(msg)

    preds = model.transform(df)

    id_col = "id" if "id" in preds.columns else None
    cluster_col = "cluster" if "cluster" in preds.columns else "prediction"
    if id_col is None:
        preds = preds.withColumn("id", F.monotonically_increasing_id())
        id_col = "id"

    out = preds.select(F.col(id_col).cast("long").alias("id"),
                       F.col(cluster_col).cast("int").alias("cluster"))
    out.write.mode("overwrite").parquet(out_path)

    print(f"[INFO] Mode: {used_mode}")
    print(f"[INFO] Saved predictions -> {out_path}")

    spark.stop()


if __name__ == "__main__":
    main()

