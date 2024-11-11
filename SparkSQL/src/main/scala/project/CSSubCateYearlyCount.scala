package project

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}

object CSSubCateYearlyCount {
  def main(args: Array[String]): Unit = {
    val conf: SparkConf = new SparkConf().setMaster("local[*]").setAppName("CateYearlyPaperCount")
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    import spark.implicits._

    // Load the JSON data from the file to get the list of categories
    val csCategoryCountDF: DataFrame = spark.read
      .json("/Users/sarek/Downloads/SparkWorkspace/sparkPro1/data/cs_category_count.json")
    val csCategoryList = csCategoryCountDF
      .select("category").distinct().collect().map(_.getString(0)).toList

    // Read the entire dataset
    val df: DataFrame = spark.read
      .json("/Users/sarek/Downloads/SparkWorkspace/sparkPro1/data/arxiv-metadata-oai-snapshot.json")

    // Register the DataFrame as a temporary view
    df.createOrReplaceTempView("arxivData")

    // Use Spark SQL to split the categories string into an array and explode it, then extract the year from versions[0].created
    val explodedDF = spark.sql(
    """
      SELECT
        id,
        explode(split(categories, ' ')) AS category,
        split(versions[0].created, ' ')[3] AS year
      FROM arxivData
    """
    )

    // Register the exploded DataFrame as a temporary view
    explodedDF.createOrReplaceTempView("exploded_arxivData")

    // Use Spark SQL to filter by categories that are in the csCategoryList and group by category and year, and count the number of papers
    val categoryYearCountDF = spark.sql(s"""
      SELECT
        category,
        year,
        COUNT(*) AS count
      FROM exploded_arxivData
      WHERE category IN (${csCategoryList.map(category => s"'$category'").mkString(", ")})
      GROUP BY category, year
      ORDER BY category, year
    """)

    // Save the result to a JSON file
    categoryYearCountDF.coalesce(1).write
      .mode("overwrite")
      .json("/Users/sarek/Downloads/SparkWorkspace/sparkPro1/output/csCateYearlyPaperCount")

    // Release resources
    spark.stop()
  }
}
