package com.spark

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}

import java.io.File

object JsonProcessor {
  /**
   * Processes a single JSON file.
   * Reads the file, performs text preprocessing, computes TF-IDF values,
   * determines the optimal K value for clustering, performs clustering, and outputs the results.
   *
   * @param file  The JSON file to process.
   * @param sc    The SparkContext.
   * @param spark The SparkSession.
   */
  def processJsonFile(file: File, sc: SparkContext, spark: SparkSession): Unit = {
    println(s"Processing file: ${file.getName}")

    // Step 4: 读取 JSON 文件，将其转换为 RDD
    val filePath = file.getAbsolutePath
    val fileName = file.getName.split("\\.").dropRight(1).mkString(".")
    val jsonDataFrame: DataFrame = spark.read.json(filePath)
    // 选择需要的 abstract 字段
    val abstractRDD: RDD[String] = jsonDataFrame
      .select("abstract")
      .rdd.map(row => row.getString(0))

    // Step 5: 文本预处理（分词、去除停用词、词形还原）
    val filteredRDD: RDD[Seq[String]] = TextPreprocessor.preprocess(abstractRDD)

    // Step 6: 计算 TF-IDF
    val (tfIdf: RDD[org.apache.spark.mllib.linalg.Vector], vocabDict: Map[Int, String]) = TfidfCalculator
      .computeTFIDF(filteredRDD)

    // Step7: 使用采样数据确定最佳 K 值
    val sampledTfIdf = tfIdf.sample(withReplacement = false, fraction = 0.5)
    val optimalK = ClusteringEvaluator.determineOptimalK(sampledTfIdf)

    // Step 8: 使用最佳 K 值对完整数据进行聚类
    val model = Clustering.performKMeans(tfIdf, optimalK)

    // Step 9: 预测并输出结果
    Clustering.outputClusterPredictions(vocabDict, tfIdf, model, fileName)
  }
}