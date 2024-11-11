package project

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object CateYearlyPaperCount {
  def main(args: Array[String]): Unit = {
    val conf: SparkConf = new SparkConf().setMaster("local[*]").setAppName("CateYearlyPaperCount")
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    import spark.implicits._

    // Load the JSON data from the file to get the list of categories
    val categoryCountDF: DataFrame = spark.read
      .json("/Users/sarek/Downloads/SparkWorkspace/sparkPro1/data/category_count.json")
    val categoryList = categoryCountDF
      .select("main_category").distinct().collect().map(_.getString(0)).toList

    // Read the entire dataset
    val df: DataFrame = spark.read
      .json("/Users/sarek/Downloads/SparkWorkspace/sparkPro1/data/arxiv-metadata-oai-snapshot.json")

    // Register the DataFrame as a temporary view
    df.createOrReplaceTempView("arxivData")

    val explodedDF = spark.sql("""
      SELECT
        id,
        explode(split(categories, ' ')) AS category,
        split(versions[0].created, ' ')[3] AS year
      FROM arxivData
    """)

    // Register the exploded DataFrame as a temporary view
    explodedDF.createOrReplaceTempView("exploded_arxivData")

    val categoryYearCountDF = spark.sql(s"""
      SELECT
        CASE
          ${categoryList.map(category => s"WHEN category LIKE '$category%' THEN '$category'").mkString(" ")}
          ELSE 'other'
        END AS main_category,
        year,
        COUNT(*) AS count
      FROM exploded_arxivData
      WHERE ${categoryList.map(category => s"category LIKE '$category%'").mkString(" OR ")}
      GROUP BY main_category, year
      ORDER BY main_category, year
    """)

    // Save the result to a JSON file
    categoryYearCountDF.coalesce(1).write
      .mode("overwrite")
      .json("/Users/sarek/Downloads/SparkWorkspace/sparkPro1/output/cateYearlyPaperCount")

    // Release resources
    spark.stop()
  }
}
