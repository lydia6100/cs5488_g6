package project

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object PaperYearlyCount {
  def main(args: Array[String]): Unit = {
    val conf: SparkConf = new SparkConf().setMaster("local[*]").setAppName("YearCount")
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    import spark.implicits._

    val df: DataFrame = spark.read
      .json("/Users/sarek/Downloads/SparkWorkspace/sparkPro1/data/arxiv-metadata-oai-snapshot.json")

    df.createOrReplaceTempView("arxivData")

    // SQL query to count the number of papers published(v1) each year
    val yearCountDF = spark.sql(
      """
      SELECT split(versions[0].created, ' ')[3] AS year,
        COUNT(*) AS count
        FROM arxivData
        GROUP BY split(versions[0].created, ' ')[3]
      """
    )
    // Save the result to a JSON file
    yearCountDF.coalesce(1).write
      .mode("overwrite")
      .json("/Users/sarek/Downloads/SparkWorkspace/sparkPro1/output/yearlyPaperCount")
    // Release resources
    spark.stop()
  }
}
