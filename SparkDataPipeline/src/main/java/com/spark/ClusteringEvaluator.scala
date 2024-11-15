package com.spark

import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.rdd.RDD

object ClusteringEvaluator {
  /**
   * Determines the optimal number of clusters (K) for KMeans using the Elbow Method.
   * Uses WSSSE (Within Set Sum of Squared Errors) to evaluate different values of K.
   *
   * @param tfIdf The RDD of TF-IDF vectors.
   * @param sc    The SparkContext.
   * @return The optimal number of clusters (K).
   */
  def determineOptimalK(tfIdf: RDD[org.apache.spark.mllib.linalg.Vector], sc: SparkContext): Int = {
    println("[determineOptimalK] Determining the optimal K value")

    println("Determining the optimal K value...")

    // 将 RDD 持久化以提升多次操作的性能
    tfIdf.persist(org.apache.spark.storage.StorageLevel.MEMORY_AND_DISK)

    val ks = (2 to 5)
    var optimalK = 2
    var minWSSSE = Double.MaxValue

    for (k <- ks) {
      val kmeans = new KMeans()
        .setK(k)
        .setSeed(1L)
        .setMaxIterations(10)

      val model = kmeans.run(tfIdf)
      val wssse = model.computeCost(tfIdf)
      println(s"K = $k, WSSSE = $wssse")

      if (wssse < minWSSSE) {
        minWSSSE = wssse
        optimalK = k
      }
    }

    println(s"Optimal K value determined: $optimalK")

    // 释放 RDD 缓存
    tfIdf.unpersist()
    optimalK
  }
}