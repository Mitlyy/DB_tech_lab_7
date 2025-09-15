import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import com.typesafe.config.ConfigFactory
import java.util.Properties
import java.nio.file.{Files, Paths}

object Main {
  def main(args: Array[String]): Unit = {
    val cfg = ConfigFactory.load()
    val db  = cfg.getConfig("app.mysql")
    val pth = cfg.getConfig("app.paths")
    val tbl = cfg.getConfig("app.tables")

    val jdbcUrl = s"jdbc:mysql://${db.getString("host")}:${db.getInt("port")}/${db.getString("db")}?serverTimezone=UTC"

    val props = new Properties()
    props.setProperty("user", db.getString("user"))
    props.setProperty("password", db.getString("pass"))
    props.setProperty("driver", "com.mysql.cj.jdbc.Driver")

    val featuresPath    = pth.getString("features")        // data/clean/features.parquet
    val predictionsPath = pth.getString("predictions")     // data/out/predictions.parquet
    val rawTable        = tbl.getString("raw")             // raw_products
    val predTable       = tbl.getString("predictions")     // predictions

    val spark = SparkSession.builder()
      .appName("lab7")
      .master(sys.env.getOrElse("SPARK_MASTER", "local[*]"))
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    val required = Seq(
      "energy_100g",
      "proteins_100g",
      "fat_100g",
      "carbohydrates_100g",
      "sugars_100g",
      "fiber_100g",
      "salt_100g"
    )

    var df = spark.read.jdbc(jdbcUrl, rawTable, props)
      .withColumn("id", col("id").cast(LongType))

    required.foreach { c =>
      if (df.columns.contains(c)) {
        df = df.withColumn(c, regexp_replace(col(c).cast(StringType), ",", ".").cast(DoubleType))
      } else {
        df = df.withColumn(c, lit(null).cast(DoubleType))
      }
    }

    val limits = Map(
      "energy_100g"        -> (0.0, 4000.0),
      "proteins_100g"      -> (0.0, 100.0),
      "fat_100g"           -> (0.0, 100.0),
      "carbohydrates_100g" -> (0.0, 100.0),
      "sugars_100g"        -> (0.0, 100.0),
      "fiber_100g"         -> (0.0, 100.0),
      "salt_100g"          -> (0.0, 100.0)
    )
    limits.foreach { case (c, (lo, hi)) =>
      df = df.filter(col(c).isNull || (col(c) >= lit(lo) && col(c) <= lit(hi)))
    }

    val nRaw = df.count()
    if (nRaw == 0) {
      println("[lab7] WARNING: raw_products пустая — данных нет. Заполни хотя бы 1–2 строки и перезапусти.")
      df.select(Seq("id") ++ required map col: _*)
        .limit(0)
        .write.mode("overwrite").parquet(featuresPath)
      spark.stop()
      return
    }

    val outDf = df.select(Seq("id") ++ required map col: _*)
    outDf.write.mode("overwrite").parquet(featuresPath)
    println(s"[lab7] Wrote raw features parquet to: " + featuresPath)
    println(s"[lab7] Rows written: " + outDf.count())

    if (Files.exists(Paths.get(predictionsPath))) {
      val preds = spark.read.parquet(predictionsPath)
      val cols = preds.columns.toSet

      val normalized =
        if (!cols.contains("cluster") && cols.contains("prediction")) {
          preds.withColumn("cluster", col("prediction"))
        } else {
          preds
        }

      val withId =
        if (!normalized.columns.contains("id"))
          normalized.withColumn("id", monotonically_increasing_id())
        else
          normalized

      val out = withId
        .withColumn("id", col("id").cast(LongType))
        .withColumn("cluster", col("cluster").cast(IntegerType))
        .withColumn("processed_at", current_timestamp())
        .select("id", "cluster", "processed_at")

      val n = out.count()
      println(s"[lab7] Loaded predictions rows: " + n)
      out.show(5, truncate=false)

      out.write.mode("append").jdbc(jdbcUrl, predTable, props)
      println(s"[lab7] Wrote " + n + " rows to MySQL table: " + predTable)
    } else {
      println(s"[lab7] predictions file not found: " + predictionsPath + " — шаг записи в MySQL пропущен.")
    }

    spark.stop()
  }
}

