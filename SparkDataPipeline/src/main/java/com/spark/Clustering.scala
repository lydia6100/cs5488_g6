package com.spark

import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.json4s.DefaultFormats
import org.json4s.jackson.Json

import java.io.{File, PrintWriter}


object Clustering {
  /**
   * Performs KMeans clustering on the given TF-IDF vectors.
   *
   * @param tfIdf The RDD of TF-IDF vectors.
   * @param k     The number of clusters.
   * @return The trained KMeans model.
   */
  def performKMeans(tfIdf: RDD[org.apache.spark.mllib.linalg.Vector], k: Int): KMeansModel = {
    println("[performKMeans] Performing KMeans clustering")
    println(s"Performing KMeans clustering with K = $k")
    val kmeans = new KMeans()
    kmeans.setK(k)
    kmeans.setSeed(1L)
    val model = kmeans.run(tfIdf)
    model
  }

  /**
   * Outputs the cluster predictions for each document.
   * Each document is mapped to its respective cluster, along with the keywords and their scores.
   *
   * @param filteredRDD The filtered RDD containing sequences of words.
   * @param tfIdf       The RDD of TF-IDF vectors.
   * @param model       The trained KMeans model.
   * @param fileName    The name of the JSON file being processed (used for output file naming).
   */
  def outputClusterPredictions(filteredRDD: RDD[Seq[String]], tfIdf: RDD[org.apache.spark.mllib.linalg.Vector], model: KMeansModel, fileName: String): Unit = {
    println("[outputClusterPredictions] Outputting cluster predictions")
    val keywordsWithScoresRDD = tfIdf.zip(filteredRDD).map { case (vector, keywords) =>
      val scores = vector.toArray
      (keywords, scores)
    }

    // 预测每篇文章的簇并生成 JSON
    val predictions = keywordsWithScoresRDD.map { case (keywords, scores) =>
      val cluster = model.predict(Vectors.dense(scores))
      Map(
        "keywords" -> keywords,
        "scores" -> scores,
        "cluster" -> cluster
      )
    }

    // 将 JSON 结果保存到文件
    val outputPath = s"D:\\Code\\OtherProject\\cs5488_g6\\SparkDataPipeline\\src\\main\\resources\\output\\${fileName}.json"
    val jsonStrings = predictions.collect().map { prediction =>
      implicit val formats: DefaultFormats.type = DefaultFormats
      Json(DefaultFormats).write(prediction)
    }

    val writer = new PrintWriter(new File(outputPath))
    try {
      jsonStrings.foreach(writer.println)
    } finally {
      writer.close()
    }
  }
}