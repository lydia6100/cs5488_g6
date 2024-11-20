package com.spark

import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vectors}
import org.apache.spark.rdd.RDD
import org.json4s.DefaultFormats
import org.json4s.jackson.Json

import java.io.{BufferedWriter, File, FileWriter, PrintWriter}

object OutputHandler {
  /**
   * Performs PCA to reduce the dimensionality of the data to 2D and saves the clustering results.
   *
   * @param tfIdf    The RDD of TF-IDF vectors.
   * @param model    The trained KMeans model.
   * @param fileName The name of the file being processed, used for output.
   */
  def performPCAAndSaveResults(tfIdf: RDD[org.apache.spark.mllib.linalg.Vector], model: KMeansModel, fileName: String): Unit = {
    // 将稀疏向量转换为密集向量以进行 PCA 降维
    val tfIdfDense = tfIdf.map {
      case sparse: SparseVector => Vectors.dense(sparse.toArray)
      case dense: DenseVector => dense
    }

    // 使用 RowMatrix 进行 PCA 降维到二维
    val rowMatrix = new RowMatrix(tfIdfDense)
    val pcaModel = rowMatrix.computePrincipalComponents(2)
    val tfIdf2D = rowMatrix.multiply(pcaModel).rows

    // 保存聚类结果到文件（只保存每篇文章的二维坐标和聚类标签）
    val outputFilePath = s"${Config.outputPath}\\${fileName}${Config.clusterResultsPath}"
    val writer = new BufferedWriter(new FileWriter(outputFilePath))
    try {
      writer.write("X1,X2,Cluster\n")
      tfIdf2D.zip(model.predict(tfIdf)).collect().foreach { case (vector, cluster) =>
        writer.write(s"${vector(0)},${vector(1)},$cluster\n")
      }
    } finally {
      writer.close()
    }
  }

  /**
   * Outputs the cluster predictions, including keywords, scores, and cluster labels for each document.
   *
   * @param vocabDict The vocabulary dictionary mapping indices to terms.
   * @param tfIdf     The RDD of TF-IDF vectors.
   * @param model     The trained KMeans model.
   * @param fileName  The name of the file being processed, used for output.
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
    val outputPath = s"${Config.outputPath}\\${fileName}.json"
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