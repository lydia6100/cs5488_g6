package com.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

import java.io.File

object Config {
  val inputPath: String = "D:\\Code\\OtherProject\\cs5488_g6\\SparkDataPipeline\\src\\main\\resources\\test_input"
  val outputPath: String = "D:\\Code\\OtherProject\\cs5488_g6\\SparkDataPipeline\\src\\main\\resources\\test_output"
  val clusterResultsPath: String = "_cluster_results.csv"
}

object MainApp {
  def main(args: Array[String]): Unit = {
    // Step 1: 初始化 Spark 上下文和会话
    val conf = new SparkConf()
      .setAppName("TF-IDF-KMeans")
      .setMaster("local[*]")
      .set("spark.driver.memory", "20g")
      .set("spark.executor.memory", "20g")
      .set("spark.default.parallelism", "8") // 增加并行度

    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    val spark = SparkSession
      .builder()
      .appName("TF-IDF-KMeans")
      .getOrCreate()

    // Step 2: 获取所有 JSON 文件的路径 (使用绝对路径)
    val resourcePath = new File(Config.inputPath)
    val jsonFiles = resourcePath.listFiles.filter(_.getName.endsWith(".json"))

    // Step 3: 将处理每个 JSON 文件的操作并行执行
    jsonFiles.par.foreach { file =>
      JsonProcessor.processJsonFile(file, sc, spark)
    }
    // 停止 Spark 上下文
    sc.stop()
  }
}