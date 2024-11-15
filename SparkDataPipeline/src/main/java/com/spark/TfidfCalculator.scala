package com.spark

import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD

object TfidfCalculator {
  /**
   * Computes the TF-IDF values for the given filtered RDD.
   * The TF-IDF values are filtered based on minimum score and top percentage.
   *
   * @param filteredRDD The filtered RDD containing sequences of words.
   * @param minScore    The minimum score threshold for filtering TF-IDF values.
   * @param topPercent  The percentage of top scores to retain.
   * @return An RDD of TF-IDF vectors.
   */
  def computeTFIDF(filteredRDD: RDD[Seq[String]], minScore: Double = 0.05, topPercent: Double = 0.8): RDD[org.apache.spark.mllib.linalg.Vector] = {
    println("[computeTFIDF] Start computing TF-IDF")
    val hashingTF = new HashingTF(Math.pow(2, 18).toInt)
    val tf: RDD[org.apache.spark.mllib.linalg.Vector] = hashingTF.transform(filteredRDD)

    val idf = new IDF().fit(tf)
    val tfIdf: RDD[org.apache.spark.mllib.linalg.Vector] = idf.transform(tf)
    val filteredTfIdf = tfIdf.map { vector =>
      val filteredValuesWithIndices = vector.toArray.zipWithIndex
        .filter { case (score, _) => score >= minScore }
        .sortBy(-_._1)
        .take((vector.size * topPercent).toInt)

      val filteredArray = Array.ofDim[Double](vector.size)
      filteredValuesWithIndices.foreach { case (score, index) =>
        filteredArray(index) = score
      }
      Vectors.dense(filteredArray)
    }
    filteredTfIdf
  }
}