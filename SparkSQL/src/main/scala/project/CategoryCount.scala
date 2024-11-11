package project

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}


object CategoryCount {
  def main(args: Array[String]): Unit = {

    val conf: SparkConf = new
        SparkConf().setMaster("local[*]").setAppName("CountCategory")
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    import spark.implicits._

    val df: DataFrame = spark.read
      //      .option("multiLine", value = true)
      .json("/Users/sarek/Downloads/SparkWorkspace/sparkPro1/data/arxiv-metadata-oai-snapshot.json")

    // Create a temporary view for arxivData
    df.createOrReplaceTempView("arxivData")

    // Split the categories labels into separate rows
    val explodedDF = spark.sql(
      """
        SELECT EXPLODE(SPLIT(categories, ' ')) AS category
        FROM arxivData
      """
    )

    explodedDF.createOrReplaceTempView("explodedCategories")

    // Extract the main category then count for each category
    val categoryCount = spark.sql(
      """
        SELECT SUBSTRING_INDEX(category, '.', 1) AS main_category,
          COUNT(*) AS count
          FROM explodedCategories
          GROUP BY main_category
          ORDER BY count DESC
      """
    )
    categoryCount.show()
    // Save to JSON
    categoryCount.coalesce(1)
      .write.mode("overwrite")
      .json("/Users/sarek/Downloads/SparkWorkspace/sparkPro1/output/categoryCount")

    spark.stop()
  }
}
