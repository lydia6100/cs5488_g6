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
   * Outputs the cluster predictions along with the keywords, scores, and assigned cluster.
   *
   * @param vocabDict The vocabulary dictionary (index -> term).
   * @param tfIdf     The RDD of TF-IDF vectors.
   * @param model     The trained KMeans model.
   * @param fileName  The output file name.
   */
  def outputClusterPredictions(vocabDict: Map[Int, String], tfIdf: RDD[org.apache.spark.mllib.linalg.Vector], model: KMeansModel, fileName: String): Unit = {
    println("[outputClusterPredictions] Outputting cluster predictions")

    val predictions = tfIdf.map { vector =>
      val cluster = model.predict(vector)
      val keywordsWithScores = vector.toArray.zipWithIndex
        .filter { case (score, _) => score > 0 }
        .map { case (score, index) => (vocabDict.getOrElse(index, ""), score) }

      Map(
        "keywords" -> keywordsWithScores.map(_._1),
        "scores" -> keywordsWithScores.map(_._2),
        "cluster" -> cluster
      )
    }

    // Save JSON results to file
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