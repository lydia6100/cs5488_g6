package com.spark

import edu.stanford.nlp.pipeline._
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.rdd.RDD

import java.util.Properties
import scala.collection.JavaConverters._

object TextPreprocessor {
  /**
   * Preprocesses the text data by performing several steps:
   * 1. Remove newline characters.
   * 2. Tokenize the text into words using Stanford NLP.
   * 3. Remove stop words.
   * 4. Perform lemmatization using Stanford NLP.
   *
   * @param abstractsRDD The RDD of document abstracts.
   * @return An RDD containing sequences of preprocessed words for each document.
   */
  def preprocess(abstractsRDD: RDD[String]): RDD[Seq[String]] = {
    println("[preprocess] Start text preprocessing")

    abstractsRDD.mapPartitions { partition =>
      // 配置Stanford CoreNLP
      val props = new Properties()
      props.setProperty("annotators", "tokenize, ssplit, pos, lemma")
      val pipeline = new StanfordCoreNLP(props)

      partition.map { text =>
        // 去除换行符
        val cleanedText = text.replaceAll("\n", " ")

        // 使用Stanford CoreNLP进行分词和词形还原
        val document = new CoreDocument(cleanedText)
        pipeline.annotate(document)

        val tokens = document.tokens().asScala

        // 获取词形还原后的词汇
        val lemmatizedWords = tokens.map(_.lemma().toLowerCase)

        // 去除停用词
        val remover = new StopWordsRemover()
        val stopWords = remover.getStopWords
        val filteredWords = lemmatizedWords.filter(word => !stopWords.contains(word))

        filteredWords.filter(_.nonEmpty).toList
      }
    }
  }
}
