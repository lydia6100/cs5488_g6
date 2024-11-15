package com.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

import java.io.File

object MainApp {
  def main(args: Array[String]): Unit = {
    // Step 1: 初始化 Spark 上下文和会话
    val conf = new SparkConf().setAppName("TF-IDF-KMeans")
      .setMaster("local[*]")
      .set("spark.driver.memory", "16g")
      .set("spark.executor.memory", "16g")
      .set("spark.default.parallelism", "8") // 增加并行度

    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")
    val spark = SparkSession.builder().appName("TF-IDF-KMeans").getOrCreate()

    // Step 2: 获取所有 JSON 文件的路径 (使用绝对路径)
    val resourcePath = new File("D:\\Code\\OtherProject\\cs5488_g6\\SparkDataPipeline\\src\\main\\resources\\input")
    val jsonFiles = resourcePath.listFiles.filter(_.getName.endsWith(".json"))

    // Step 3: 将处理每个 JSON 文件的操作顺序执行
    jsonFiles.foreach { file =>
      JsonProcessor.processJsonFile(file, sc, spark)
    }
    // 停止 Spark 上下文
    sc.stop()
  }
}