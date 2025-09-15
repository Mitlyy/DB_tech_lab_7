## Run (local)
# 1) DB
```
docker compose up -d

```
# 2) Build parquet (raw features)
```
sbt run

```
# 3) Inference (uses PipelineModel/KMeansModel auto-detect)
```
python3 scripts/predict_off.py

```
# 4) Write predictions to MySQL
```
sbt run
```
## Check
```
docker exec -it lab7-mysql mysql -ulab7 -plab7pass -e "
SELECT COUNT(*) AS n FROM lab7.predictions;
SELECT cluster, COUNT(*) AS cnt FROM lab7.predictions GROUP BY cluster ORDER BY cluster;"
```
```
