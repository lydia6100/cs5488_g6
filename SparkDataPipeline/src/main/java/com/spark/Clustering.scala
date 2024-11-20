package com.spark

import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.rdd.RDD


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
}
