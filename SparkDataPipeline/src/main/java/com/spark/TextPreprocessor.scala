package com.spark

import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.rdd.RDD

object TextPreprocessor {
  /**
   * Preprocesses the text data by performing several steps:
   * 1. Remove newline characters.
   * 2. Tokenize the text into words.
   * 3. Remove stop words.
   * 4. Perform simple lemmatization by removing common suffixes.
   *
   * @param abstractsRDD The RDD of document abstracts.
   * @return An RDD containing sequences of preprocessed words for each document.
   */
  def preprocess(abstractsRDD: RDD[String]): RDD[Seq[String]] = {
    println("[preprocess] Start text preprocessing")
    abstractsRDD.map { text =>
      // 去除换行符
      val cleanedText = text.replaceAll("\n", " ")
      // 分词
      val words = cleanedText.split("\\s+").toSeq
      // 去除停用词
      val remover = new StopWordsRemover()
      val stopWords = remover.getStopWords
      val filteredWords = words.filter(word => !stopWords.contains(word.toLowerCase))
      // 简单词形还原 (将常见后缀去除，例如 -ing, -ed, -s)
      val lemmatizedWords = filteredWords.map { word =>
        word.toLowerCase.replaceAll("[^a-zA-Z0-9]", "")
          .replaceAll("ing$", "")
          .replaceAll("ed$", "")
          .replaceAll("s$", "")
      }

      lemmatizedWords.filter(_.nonEmpty).toList
    }
  }
}
