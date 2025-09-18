#!/usr/bin/env python3
import os
from typing import Optional, List

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

try:
    from pyspark.ml.clustering import KMeansModel
    from pyspark.ml import PipelineModel
except Exception:
    print("[FATAL] Требуется PySpark. Установи pyspark и запусти снова.")
    raise

from scripts.core_config import AppConfig
from scripts.core_logging import get_logger


RAW_COLS = [
    "energy_100g",
    "proteins_100g",
    "fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "fiber_100g",
    "salt_100g",
]


class PredictorApp:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.log = get_logger(self.__class__.__name__, "INFO")
        self.spark: Optional[SparkSession] = None

    @staticmethod
    def pick_first_existing(paths: List[str]) -> Optional[str]:
        for p in paths:
            if os.path.exists(p):
                return p
        return None

    def find_data(self) -> str:
        p = self.pick_first_existing(self.cfg.data_candidates)
        if not p:
            raise SystemExit(
                "[FATAL] Не найден входной parquet. Ожидал один из:\n  - "
                + "\n  - ".join(self.cfg.data_candidates)
            )
        return p

    def find_model_dir(self) -> str:
        env = os.getenv(self.cfg.env_model_dir_var, "").strip()
        if env and os.path.isdir(env):
            return env

        p = self.pick_first_existing(self.cfg.model_dirs)
        if p:
            return p

        root = self.cfg.artifacts_root
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
            + "\n  - ".join(self.cfg.model_dirs)
        )

    def build_spark(self) -> SparkSession:
        spark = (
            SparkSession.builder.appName(self.cfg.app_name)
            .config("spark.sql.shuffle.partitions", self.cfg.spark_shuffle_partitions)
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel(self.cfg.spark_log_level)
        return spark

    def run(self) -> None:
        # Spark
        self.spark = self.build_spark()

        # Paths
        in_path = self.find_data()
        model_dir = self.find_model_dir()
        print(f"[INFO] Using data:  {in_path}")
        print(f"[INFO] Using model: {model_dir}")

        # IO
        df = self.spark.read.parquet(in_path)
        cols = set(df.columns)

        os.makedirs(self.cfg.out_dir, exist_ok=True)
        out_path = os.path.join(self.cfg.out_dir, self.cfg.out_file)

        # Model load (как и раньше)
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

        out = preds.select(
            F.col(id_col).cast("long").alias("id"),
            F.col(cluster_col).cast("int").alias("cluster"),
        )
        out.write.mode("overwrite").parquet(out_path)

        print(f"[INFO] Mode: {used_mode}")
        print(f"[INFO] Saved predictions -> {out_path}")

        self.spark.stop()


def main():
    cfg = AppConfig.from_env()
    app = PredictorApp(cfg)
    app.run()


if __name__ == "__main__":
    main()

