import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import com.typesafe.config.ConfigFactory
import java.util.Properties
import java.nio.file.{Files, Paths}

object Main {
  def main(args: Array[String]): Unit = {
    val cfg = ConfigFactory.load()
    val db  = cfg.getConfig("app.mysql")
    val pth = cfg.getConfig("app.paths")
    val tbl = cfg.getConfig("app.tables")

    val url = s"jdbc:mysql://${db.getString("host")}:${db.getInt("port")}/${db.getString("db")}?serverTimezone=UTC"

    val props = new Properties()
    props.setProperty("user", db.getString("user"))
    props.setProperty("password", db.getString("pass"))
    props.setProperty("driver", "com.mysql.cj.jdbc.Driver")

    val featuresPath    = pth.getString("features")
    val predictionsPath = pth.getString("predictions")
    val rawTable        = tbl.getString("raw")
    val predTable       = tbl.getString("predictions")

    val spark = SparkSession.builder()
      .appName("lab7")
      .master(sys.env.getOrElse("SPARK_MASTER", "local[*]"))
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    val raw: DataFrame = spark.read.jdbc(url, rawTable, props)

    val feats: DataFrame = raw
      .withColumn("energy_100g", col("energy_100g").cast("double"))
      .withColumn("fat_100g",    col("fat_100g").cast("double"))
      .withColumn("sugars_100g", col("sugars_100g").cast("double"))
      .filter(col("energy_100g").isNotNull && col("fat_100g").isNotNull && col("sugars_100g").isNotNull)
      .select(col("id").cast("long").as("id"), col("energy_100g"), col("fat_100g"), col("sugars_100g"))

    feats.write.mode("overwrite").parquet(featuresPath)

    val predsExists = Files.exists(Paths.get(predictionsPath))
    if (predsExists) {
      val preds = spark.read.parquet(predictionsPath)
      preds
        .select(
          col("id").cast("long").as("id"),
          col("prediction").cast("int").as("cluster"),
          current_timestamp().as("processed_at")
        )
        .write
        .mode("append")
        .jdbc(url, predTable, props)
    } else {
      println(s"[lab7] predictions file not found: $predictionsPath — шаг записи в MySQL пропущен.")
    }

    spark.stop()
  }
}

