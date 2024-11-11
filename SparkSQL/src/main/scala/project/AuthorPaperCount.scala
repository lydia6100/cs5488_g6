package project

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object AuthorPaperCount {
  def main(args: Array[String]): Unit = {
    val conf: SparkConf = new SparkConf().setMaster("local[*]").setAppName("AuthorPaperCount")

    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    import spark.implicits._

    val df: DataFrame = spark.read
      .json("/Users/sarek/Downloads/SparkWorkspace/sparkPro1/data/arxiv-metadata-oai-snapshot.json")
    // You need to use real parth to your data

    // Explode the authors_parsed array
    val explodedDF = df.withColumn("author", explode($"authors_parsed"))

    // Create a temporary view from the exploded DataFrame
    explodedDF.createOrReplaceTempView("explodedArxivData")

    // SQL query to count the number of papers published by each author
    val authorCountDF = spark.sql(
      """
      SELECT CONCAT(author[0], ' ', author[1]) AS author,
        COUNT(*) AS count
        FROM explodedArxivData
        GROUP BY CONCAT(author[0], ' ', author[1])
      """
    )

    // Save the result to a JSON file
    authorCountDF.coalesce(1).write
      .mode("overwrite")
      .json("/Users/sarek/Downloads/SparkWorkspace/sparkPro1/output/authorPaperCount")

    // Release resources
    spark.stop()
  }
}
