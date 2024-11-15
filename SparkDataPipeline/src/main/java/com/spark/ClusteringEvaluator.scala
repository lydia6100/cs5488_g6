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
   * @return The optimal number of clusters (K).
   */
  def determineOptimalK(tfIdf: RDD[org.apache.spark.mllib.linalg.Vector]): Int = {
    println("[determineOptimalK] Determining the optimal K value")

    // 创建 K 值的集合并使用 Scala 并行集合来进行并行处理
    val ks = (2 to 20).par

    // 并行计算每个 K 值的 WSSSE
    val wssseValues = ks.map { k =>
      val kmeans = new KMeans().setK(k).setSeed(1L)
      val model = kmeans.run(tfIdf)
      val wssse = model.computeCost(tfIdf)
      println(s"K = $k, WSSSE = $wssse")
      (k, wssse)
    }.toList

    // 找到 WSSSE 最小的 K
    val (optimalK, minWSSSE) = wssseValues.minBy(_._2)
    println(s"Optimal K value determined: $optimalK")

    // 释放 RDD 缓存
    tfIdf.unpersist()
    optimalK
  }
}