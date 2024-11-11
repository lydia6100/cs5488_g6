package project

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object AuthorYearlyPaperCount {
  def main(args: Array[String]): Unit = {
    val conf: SparkConf = new SparkConf().setMaster("local[*]").setAppName("AuthorYearCount")
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    import spark.implicits._

    val df: DataFrame = spark.read
      .json("/Users/sarek/Downloads/SparkWorkspace/sparkPro1/data/arxiv-metadata-oai-snapshot.json")

    // Explode the authors_parsed array
    val explodedDF = df.withColumn("author", explode($"authors_parsed"))

    // Create a temporary view from the exploded DataFrame
    explodedDF.createOrReplaceTempView("explodedArxivData")

    // SQL query to count the number of papers published by each author each year
    val authorYearCountDF = spark.sql(
      """
      SELECT CONCAT(author[0], ' ', author[1]) AS author,
        split(versions[0].created, ' ')[3] AS year,
        COUNT(*) AS count
        FROM explodedArxivData
        GROUP BY CONCAT(author[0], ' ', author[1]), split(versions[0].created, ' ')[3]
      """
    )

    // Save the result to a JSON file
    authorYearCountDF.coalesce(1).write
      .mode("overwrite")
      .json("/Users/sarek/Downloads/SparkWorkspace/sparkPro1/output/authorYearlyPaperCount")

    // Release resources
    spark.stop()
  }
}
