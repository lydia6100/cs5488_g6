package project

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * Filter sub category for CS
 */
object CSCategoriesFilter {
  def main(args: Array[String]): Unit = {
    val conf: SparkConf = new
        SparkConf().setMaster("local[*]").setAppName("FilterCSCategories")
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    import spark.implicits._

    // Read the JSON file and create a DataFrame
    val df: DataFrame = spark.read
      //      .option("multiLine", value = true)
      .json("/Users/sarek/Downloads/SparkWorkspace/sparkPro1/data/arxiv-metadata-oai-snapshot.json")

    // Create a temporary view
    df.createOrReplaceTempView("arxivData")

    // Use SparkSQL to filter out data with the main category as CS
    val csData = spark.sql(
      " SELECT * FROM arxivData WHERE categories LIKE 'cs.%' OR categories LIKE '% cs.%'")

    // Get all category labels containing `cs.`
    val csCategories = csData.select("categories")
      .distinct().collect()
      .flatMap(_.getString(0).split(" "))
      .filter(_.startsWith("cs.")).distinct

    println("CS Categories: " + csCategories.mkString(", "))

    // Create a DataFrame to store the count of each subcategory
    val csCategoryCount = csCategories.map { category =>
      val count = csData.filter($"categories".contains(category)).count()
      (category, count)
    }.toSeq.toDF("category", "count")

    // Process each subcategory in parallel, query data by subcategory and save
    csCategories.par
      .foreach { category =>
        val filteredData = csData.filter($"categories".contains(category))
        filteredData.coalesce(1).write
          .mode("overwrite")
          .json(s"/Users/sarek/Downloads/SparkWorkspace/sparkPro1/output/csSubCategoryData/$category")
        println(s"Saved data for category: $category")
      }

    // Print the DataFrame content for debugging
    csCategoryCount.show()
    // Save the category counts as a JSON file
    csCategoryCount.coalesce(1).write
      .mode("overwrite")
      .json("/Users/sarek/Downloads/SparkWorkspace/sparkPro1/output/csSubCategoryCount")

    spark.stop()
  }
}
