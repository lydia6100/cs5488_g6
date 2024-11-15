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
   * @return An RDD of filtered TF-IDF vectors and the vocabulary dictionary.
   */
  def computeTFIDF(filteredRDD: RDD[Seq[String]], minScore: Double = 0.05, topPercent: Double = 0.8): (RDD[org.apache.spark.mllib.linalg.Vector], Map[Int, String]) = {
    println("[computeTFIDF] Start computing TF-IDF")
    val hashingTF = new HashingTF(Math.pow(2, 18).toInt)
    val tf: RDD[org.apache.spark.mllib.linalg.Vector] = hashingTF.transform(filteredRDD)

    val idf = new IDF().fit(tf)
    val tfIdf: RDD[org.apache.spark.mllib.linalg.Vector] = idf.transform(tf)

    // Build the vocabulary dictionary based on HashingTF indices (index -> term)
    val vocab = filteredRDD.flatMap(seq => seq).distinct().collect()
    val vocabDict = vocab.map(term => (hashingTF.indexOf(term), term)).toMap

    // Filter TF-IDF values based on minScore and topPercent
    val filteredTfIdf = tfIdf.map { vector =>
      val filteredValuesWithIndices = vector.toArray.zipWithIndex
        .filter { case (score, _) => score >= minScore }
        .sortBy(-_._1)
        .take((vector.toArray.count(_ > 0) * topPercent).toInt)

      val filteredIndices = filteredValuesWithIndices.map(_._2)
      val filteredValues = filteredValuesWithIndices.map(_._1)

      Vectors.sparse(vector.size, filteredIndices, filteredValues)
    }
    (filteredTfIdf, vocabDict)
  }

  /**
   * Converts the TF-IDF vectors back to RDD[Seq[(String, Double)]] format.
   *
   * @param vectors The RDD of TF-IDF vectors.
   * @param vocab   The global vocabulary used to build the vectors.
   * @return An RDD of sequences of (keyword, tf-idf value) pairs for each document.
   */
  def vectorsToSeq(vectors: RDD[org.apache.spark.mllib.linalg.Vector], vocab: Array[String]): RDD[Seq[(String, Double)]] = {
    vectors.map { vector =>
      vector.toArray.zipWithIndex
        .filter { case (tfidfValue, _) => tfidfValue > 0 }
        .map { case (tfidfValue, index) => (vocab(index), tfidfValue) }
        .toSeq
    }
  }
}