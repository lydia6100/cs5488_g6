package project

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * Filter main category for CS
 */
object CSAllFilter {
  def main(args: Array[String]): Unit = {
    //创建上下文环境配置对象
    val conf: SparkConf = new
        SparkConf().setMaster("local[*]").setAppName("FilterCSAll")
    //创建 SparkSession 对象
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    // 引入隐式转换规则
    import spark.implicits._

    // 读取 JSON 文件 创建 DataFrame
    val df: DataFrame = spark.read
      .json("/Users/sarek/Downloads/SparkWorkspace/sparkPro1/data/arxiv-metadata-oai-snapshot.json")

    // 显示DataFrame的schema
    df.printSchema()
    // 统计行数
    val rowCount = df.count()
    println(s"Row count: $rowCount")
    // 统计列数
    val columnCount = df.columns.length
    println(s"Column count: $columnCount")

    // 创建临时视图
    df.createOrReplaceTempView("arxivData")

    // 使用 SparkSQL 筛选出大分类为 CS 的数据
    val csData = spark.sql(
      " SELECT * FROM arxivData WHERE categories LIKE 'cs.%' OR categories LIKE '% cs.%'")

    //显示筛选后的数据
    csData.show()

    // 将筛选后的数据保存为 JSON 文件
    csData.coalesce(1).write
      .mode("overwrite")
      .json("/Users/sarek/Downloads/SparkWorkspace/sparkPro1/output/csData")

    // 统计行数
    val csRowCount = csData.count()
    println(s"CS category row count: $csRowCount")

    //释放资源
    spark.stop()
  }
}
