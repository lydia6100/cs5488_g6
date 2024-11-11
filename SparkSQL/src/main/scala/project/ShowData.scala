package project

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * Show Basic Information about Arxiv Data
 * Use 11 CPU cores, Cost 1min20s
 */
object ShowData  {
  def main(args: Array[String]): Unit = {
    val conf: SparkConf = new
        SparkConf().setMaster("local[*]").setAppName("ShowData")
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    import spark.implicits._

    // Load JSON file to DataFrame
    val df: DataFrame = spark.read
      .json("/Users/sarek/Downloads/SparkWorkspace/sparkPro1/data/arxiv-metadata-oai-snapshot.json")

    // Basic information about the DataFrame
    df.printSchema()

    val rowCount = df.count()
    println(s"Row count: $rowCount")

    val columnCount = df.columns.length
    println(s"Column count: $columnCount")

    df.summary().show()
    println("Summary of arxiv data")

    spark.stop()
  }
}
