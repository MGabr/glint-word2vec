/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.feature

import java.lang.{Iterable => JavaIterable}

import breeze.linalg.convert
import com.github.fommil.netlib.BLAS.{getInstance => blas}
import glint.{Client, Word2VecArguments}
import glint.models.client.granular.GranularBigWord2VecMatrix
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.rdd._
import org.apache.spark.util.{BoundedPriorityQueue, Utils}
import org.apache.spark.util.random.XORShiftRandom
import org.apache.spark.{SparkContext, TaskContext}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.concurrent.duration._
import scala.concurrent.{Await, ExecutionContext, Future}

/**
  *  Entry in vocabulary
  */
private case class VocabWordCn(var word: String, var cn: Int)

/**
  * Word2Vec creates vector representation of words in a text corpus.
  * The algorithm first constructs a vocabulary from the corpus
  * and then learns vector representation of words in the vocabulary.
  * The vector representation can be used as features in
  * natural language processing and machine learning algorithms.
  *
  * We used skip-gram model in our implementation and hierarchical softmax
  * method to train the model. The variable names in the implementation
  * matches the original C implementation.
  *
  * For original C implementation, see https://code.google.com/p/word2vec/
  * For research papers, see
  * Efficient Estimation of Word Representations in Vector Space
  * and
  * Distributed Representations of Words and Phrases and their Compositionality.
  */
class ServerSideGlintWord2Vec extends Serializable with Logging {

  private var vectorSize = 100
  private var learningRate = 0.01875
  private var numPartitions = 1
  private var numIterations = 1
  private var seed = Utils.random.nextLong()
  private var minCount = 5
  private var maxSentenceLength = 1000

  private var batchSize = 50
  private var n = 5
  private var numParameterServers = 5
  private var unigramTableSize = 100000000

  // default maximum payload size is 262144 bytes, akka.remote.OversizedPayloadException
  // use a twentieth of this as maximum message size to account for size of primitive types and overheads
  private val maximumMessageSize = 10000

  /**
    * Sets the maximum length (in words) of each sentence in the input data.
    * Any sentence longer than this threshold will be divided into chunks of
    * up to `maxSentenceLength` size (default: 1000)
    */
  def setMaxSentenceLength(maxSentenceLength: Int): this.type = {
    require(maxSentenceLength > 0,
      s"Maximum length of sentences must be positive but got ${maxSentenceLength}")
    this.maxSentenceLength = maxSentenceLength
    this
  }

  /**
    * Sets vector size (default: 100).
    */
  def setVectorSize(vectorSize: Int): this.type = {
    require(vectorSize > 0,
      s"vector size must be positive but got ${vectorSize}")
    this.vectorSize = vectorSize
    this
  }

  /**
    * Sets initial learning rate (default: 0.025).
    */
  def setLearningRate(learningRate: Double): this.type = {
    require(learningRate > 0,
      s"Initial learning rate must be positive but got ${learningRate}")
    this.learningRate = learningRate
    this
  }

  /**
    * Sets number of partitions (default: 1). Use a small number for accuracy.
    */
  def setNumPartitions(numPartitions: Int): this.type = {
    require(numPartitions > 0,
      s"Number of partitions must be positive but got ${numPartitions}")
    this.numPartitions = numPartitions
    this
  }

  /**
    * Sets number of iterations (default: 1), which should be smaller than or equal to number of
    * partitions.
    */
  def setNumIterations(numIterations: Int): this.type = {
    require(numIterations >= 0,
      s"Number of iterations must be nonnegative but got ${numIterations}")
    this.numIterations = numIterations
    this
  }

  /**
    * Sets random seed (default: a random long integer).
    */
  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }

  /**
    * Sets the window of words (default: 5)
    */
  def setWindowSize(window: Int): this.type = {
    require(window > 0,
      s"Window of words must be positive but got ${window}")
    require(batchSize * n * window <= maximumMessageSize,
      s"Batch size * n * window has to be below or equal to ${maximumMessageSize} to avoid oversized Akka payload")
    this.window = window
    this
  }

  /**
    * Sets minCount, the minimum number of times a token must appear to be included in the word2vec
    * model's vocabulary (default: 5).
    */
  def setMinCount(minCount: Int): this.type = {
    require(minCount >= 0,
      s"Minimum number of times must be nonnegative but got ${minCount}")
    this.minCount = minCount
    this
  }

  /**
    * Sets the mini batch size (default: 50)
    */
  def setBatchSize(batchSize: Int): this.type = {
    require(batchSize * n * window <= maximumMessageSize,
      s"Batch size * n * window has to be below or equal to ${maximumMessageSize} to avoid oversized Akka payload")
    this.batchSize = batchSize
    this
  }

  /**
    * Sets n, the number of random negative examples (default: 5)
    */
  def setN(n: Int): this.type = {
    require(batchSize * n * window <= maximumMessageSize,
      s"Batch size * n * window has to be below or equal to ${maximumMessageSize} to avoid oversized Akka payload")
    this.n = n
    this
  }

  /**
    * Sets the number of parameter servers to create (default: 5)
    */
  def setNumParameterServers(numParameterServers: Int): this.type = {
    this.numParameterServers = numParameterServers
    this
  }

  /**
    * Sets the size of the unigram table.
    * Only needs to be changed to a lower value if there is not enough memory for local testing.
    * (default: 100000000)
    */
  def setUnigramTableSize(unigramTableSize: Int): this.type = {
    this.unigramTableSize = unigramTableSize
    this
  }

  private val EXP_TABLE_SIZE = 1000
  private val MAX_EXP = 6
  private val MAX_CODE_LENGTH = 40

  /** context words from [-window, window] */
  private var window = 5

  private var trainWordsCount = 0L
  private var vocabSize = 0
  @transient private var vocab: Array[VocabWordCn] = null
  @transient private var vocabHash = mutable.HashMap.empty[String, Int]

  private def learnVocab[S <: Iterable[String]](dataset: RDD[S]): Unit = {
    val words = dataset.flatMap(x => x)

    vocab = words.map(w => (w, 1))
      .reduceByKey(_ + _)
      .filter(_._2 >= minCount)
      .map(x => VocabWordCn(x._1, x._2))
      .collect()
      .sortWith((a, b) => a.cn > b.cn)

    vocabSize = vocab.length
    require(vocabSize > 0, "The vocabulary size should be > 0. You may need to check " +
      "the setting of minCount, which could be large enough to remove all your words in sentences.")

    var a = 0
    while (a < vocabSize) {
      vocabHash += vocab(a).word -> a
      trainWordsCount += vocab(a).cn
      a += 1
    }
    logInfo(s"vocabSize = $vocabSize, trainWordsCount = $trainWordsCount")
  }

  private def createExpTable(): Array[Float] = {
    val expTable = new Array[Float](EXP_TABLE_SIZE)
    var i = 0
    while (i < EXP_TABLE_SIZE) {
      val tmp = math.exp((2.0 * i / EXP_TABLE_SIZE - 1.0) * MAX_EXP)
      expTable(i) = (tmp / (tmp + 1.0)).toFloat
      i += 1
    }
    expTable
  }

  private def getSigmoid(expTable: Broadcast[Array[Float]], f: Float, label: Float): Float = {
    if (f > MAX_EXP) {
      return label - 1
    }
    if (f < -MAX_EXP) {
      return label
    }

    val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2.0)).toInt
    label - expTable.value(ind)
  }

  /**
    * Computes the vector representation of each word in vocabulary.
    * @param dataset an RDD of sentences,
    *                each sentence is expressed as an iterable collection of words
    * @return a ServerSideGlintWord2VecModel
    */
  def fit[S <: Iterable[String]](dataset: RDD[S]): ServerSideGlintWord2VecModel = {

    learnVocab(dataset)

    val sc = dataset.context

    val expTable = sc.broadcast(createExpTable())
    val bcVocabCns = sc.broadcast(vocab.map(v => v.cn))
    val bcVocabHash = sc.broadcast(vocabHash)
    try {
      doFit(dataset, sc, expTable, bcVocabCns, bcVocabHash)
    } finally {
      expTable.destroy(blocking = false)
      bcVocabCns.destroy(blocking = false)
      bcVocabHash.destroy(blocking = false)
    }
  }


  private def doFit[S <: Iterable[String]](
                                            dataset: RDD[S], sc: SparkContext,
                                            expTable: Broadcast[Array[Float]],
                                            bcVocabCns: Broadcast[Array[Int]],
                                            bcVocabHash: Broadcast[mutable.HashMap[String, Int]]) = {
    // each partition is a collection of sentences,
    // will be translated into arrays of Index integer
    val sentences: RDD[Array[Int]] = dataset.mapPartitions { sentenceIter =>
      // Each sentence will map to 0 or more Array[Int]
      sentenceIter.flatMap { sentence =>
        // Sentence of words, some of which map to a word index
        val wordIndexes = sentence.flatMap(bcVocabHash.value.get)
        // break wordIndexes into trunks of maxSentenceLength when has more
        wordIndexes.grouped(maxSentenceLength).map(_.toArray)
      }
    }

    val newSentences = sentences.repartition(numPartitions).cache()
    val initRandom = new XORShiftRandom(seed)

    if (vocabSize.toLong * vectorSize >= Int.MaxValue) {
      throw new RuntimeException("Please increase minCount or decrease vectorSize in ServerSideGlintWord2Vec" +
        " to avoid an OOM. You are highly recommended to make your vocabSize*vectorSize, " +
        "which is " + vocabSize + "*" + vectorSize + " for now, less than `Int.MaxValue`.")
    }

    @transient
    implicit val ec = ExecutionContext.Implicits.global

    @transient
    val (client, matrix) = Client.runWithWord2VecMatrixOnSpark(sc)(
      Word2VecArguments(vectorSize, window, batchSize, n, unigramTableSize), bcVocabCns, numParameterServers)
    val syn = new GranularBigWord2VecMatrix(matrix, maximumMessageSize)

    val totalWordsCounts = numIterations * trainWordsCount + 1
    var alpha = learningRate

    for (k <- 1 to numIterations) {
      val numWordsProcessedInPreviousIterations = (k - 1) * trainWordsCount

      val sentencesContext: RDD[Array[Array[Int]]] = newSentences.mapPartitionsWithIndex { (idx, sentenceIter) =>
        val random = new XORShiftRandom(seed ^ ((idx + 1) << 16) ^ ((-k - 1) << 8))
        sentenceIter.map { sentence =>
          sentence.indices.toArray.map { i =>
            val b = random.nextInt(window)
            val contextIndices = (Math.max(0, i - b) until Math.min(i + b, sentence.length)).filter(j => j != i)
            contextIndices.map(ci => sentence(ci)).toArray
          }
        }
      }

      newSentences.zip(sentencesContext).foreachPartition { iter =>
        @transient
        implicit val ec = ExecutionContext.Implicits.global

        val idx = TaskContext.getPartitionId()
        val random = new XORShiftRandom(seed ^ ((idx + 1) << 16) ^ ((-k - 1) << 8))

        iter.foldLeft((0L, 0L)) {
          case ((lastWordCount, wordCount), (sentence, sentenceContext)) =>
            var lwc = lastWordCount
            var wc = wordCount
            if (wordCount - lastWordCount > 10000) {
              lwc = wordCount
              alpha = learningRate *
                (1 - (numPartitions * wordCount.toDouble + numWordsProcessedInPreviousIterations) /
                  totalWordsCounts)
              if (alpha < learningRate * 0.0001) alpha = learningRate * 0.0001
              logInfo(s"wordCount = ${wordCount + numWordsProcessedInPreviousIterations}, " +
                s"alpha = $alpha")
            }
            wc += sentence.length

            // actual training - communicate with parameter servers
            val sentenceMiniBatches = sentence.sliding(batchSize, batchSize)
            val sentenceContextMiniBatches = sentenceContext.sliding(batchSize, batchSize)
            val miniBatchFutures = sentenceMiniBatches.zip(sentenceContextMiniBatches).map { case (wInput, wOutput) =>
              val seed = random.nextLong()
              syn.dotprod(wInput, wOutput, seed).map { case (fPlus, fMinus) =>
                val gPlus = fPlus.map(f => getSigmoid(expTable, f, 1.0f) * alpha.toFloat)
                val gMinus = fMinus.map(f => getSigmoid(expTable, f, 0.0f) * alpha.toFloat)
                syn.adjust(wInput, wOutput, gPlus, gMinus, seed)
              }
            }
            // the map here is important because simply using foreach would start all futures at the same time
            miniBatchFutures.map(Await.ready(_, 1 minute)).foreach(identity)

            (lwc, wc)
        }
      }
    }

    newSentences.unpersist()

    val wordArray = vocab.map(_.word)
    new ServerSideGlintWord2VecModel(wordArray.zipWithIndex.toMap, syn, client)
  }

  /**
    * Computes the vector representation of each word in vocabulary (Java version).
    * @param dataset a JavaRDD of words
    * @return a ServerSideGlintWord2VecModel
    */
  def fit[S <: JavaIterable[String]](dataset: JavaRDD[S]): ServerSideGlintWord2VecModel = {
    fit(dataset.rdd.map(_.asScala))
  }

}

/**
  * ServerSideGlintWord2Vec model
  *
  * @param wordIndex maps each word to an index, which can retrieve the corresponding vector from the word vector matrix
  * @param matrix holding the word vector parameter servers
  * @param client to the parameter servers
  * @param ec the implicit execution context in which to execute the parameter server requests
  */
class ServerSideGlintWord2VecModel private[spark](private[spark] val wordIndex: Map[String, Int],
                                                  private[spark] val matrix: GranularBigWord2VecMatrix,
                                                  @transient private[spark] val client: Client)
  extends Serializable with Saveable {

  /**
    * The number of words in the vocabulary
    */
  val numWords: Int = wordIndex.size

  /**
    * The dimension of each word's vector
    */
  val vectorSize: Int = matrix.cols.toInt

  /**
    * Ordered list of words obtained from wordIndex
    */
  private val wordList: Array[String] = {
    val (wl, _) = wordIndex.toSeq.sortBy(_._2).unzip
    wl.toArray
  }

  /**
    * Array of length numWords, each value being the Euclidean norm of the wordVector
    */
  private val wordVecNorms: Array[Float] = Await.result(matrix.norms(), 1 minute)

  override protected def formatVersion = "1.0"

  @transient
  implicit private lazy val ec: ExecutionContext = ExecutionContext.Implicits.global

  override def save(sc: SparkContext, path: String): Unit = {
    val savedFuture = matrix.save(path, sc.hadoopConfiguration)
    val wordArrayPath = path + "/words"
    sc.parallelize(wordList, 1).saveAsTextFile(wordArrayPath)
    Await.ready(savedFuture, 3 minutes)
  }

  /**
    * Transforms a word to its vector representation.
    *
    * Note that this implementation makes a blocking call to the underlying distributed matrix.
    * In most cases you will want to use the more efficient
    * [[org.apache.spark.mllib.feature.ServerSideGlintWord2VecModel.transform(words* transform]]
    * to not block for each word but for a batch of words.
    *
    * @param word a word
    * @return vector representation of word
    */
  def transform(word: String): Vector = {
    wordIndex.get(word) match {
      case Some(ind) =>
        val vec = Await.result(matrix.pull(Array(ind)), 1 minute)(0)
        Vectors.fromBreeze(convert(vec, Double))
      case None =>
        throw new IllegalStateException(s"$word not in vocabulary")
    }
  }

  /**
    * Transforms a set of word to their vector representations.
    *
    * Note that this implementation makes a blocking call to the underlying distributed matrix.
    *
    * @param words a set of words
    * @return vector representations of the words
    */
  def transform(words: Iterator[String]): Iterator[Vector] = {
    // make requests to parameter server with 10.000 word batches
    val wordSlidesIter = words.sliding(10000, 10000).withPartial(true)
    val vectors = wordSlidesIter.flatMap { slideWords =>
      val slideWordIndices = slideWords.map { word =>
        wordIndex.get(word) match {
          case Some(index) => index
          case None => throw new IllegalStateException(s"$word not in vocabulary")
        }
      }.map(_.toLong).toArray
      val vecs = Await.result(matrix.pull(slideWordIndices), 1 minute)
      vecs.map(vec => Vectors.fromBreeze(convert(vec, Double)))
    }
    vectors
  }

  /**
    * Find synonyms of a word; do not include the word itself in results.
    *
    * Note that this implementation makes a blocking call to the underlying distributed matrix.
    *
    * @param word a word
    * @param num number of synonyms to find
    * @return array of (word, cosineSimilarity)
    */
  def findSynonyms(word: String, num: Int): Array[(String, Double)] = {
    val vector = transform(word)
    findSynonyms(vector, num, Some(word))
  }

  /**
    * Find synonyms of the vector representation of a word, possibly
    * including any words in the model vocabulary whose vector respresentation
    * is the supplied vector.
    *
    * Note that this implementation makes a blocking call to the underlying distributed matrix.
    *
    * @param vector vector representation of a word
    * @param num number of synonyms to find
    * @return array of (word, cosineSimilarity)
    */
  def findSynonyms(vector: Vector, num: Int): Array[(String, Double)] = {
    findSynonyms(vector, num, None)
  }

  /**
    * Find synonyms of the vector representation of a word, rejecting
    * words identical to the value of wordOpt, if one is supplied.
    *
    * @param vector vector representation of a word
    * @param num number of synonyms to find
    * @param wordOpt optionally, a word to reject from the results list
    * @return array of (word, cosineSimilarity)
    */
  private def findSynonyms(
                            vector: Vector,
                            num: Int,
                            wordOpt: Option[String]): Array[(String, Double)] = {
    require(num > 0, "Number of similar words should > 0")

    val fVector = vector.toArray.map(_.toFloat)
    val alpha: Float = 1
    val beta: Float = 0
    // Normalize input vector before blas.sgemv to avoid Inf value
    val vecNorm = blas.snrm2(vectorSize, fVector, 1)
    if (vecNorm != 0.0f) {
      blas.sscal(vectorSize, 1 / vecNorm, fVector, 0, 1)
    }

    val cosineVec = Await.result(matrix.multiply(fVector), 1 minute)

    var i = 0
    while (i < numWords) {
      val norm = wordVecNorms(i)
      if (norm == 0.0f) {
        cosineVec(i) = 0.0f
      } else {
        cosineVec(i) /= norm
      }
      i += 1
    }

    val pq = new BoundedPriorityQueue[(String, Float)](num + 1)(Ordering.by(_._2))

    var j = 0
    while (j < numWords) {
      pq += Tuple2(wordList(j), cosineVec(j))
      j += 1
    }

    val scored = pq.toSeq.sortBy(-_._2)

    val filtered = wordOpt match {
      case Some(w) => scored.filter(tup => w != tup._1)
      case None => scored
    }

    filtered
      .take(num)
      .map { case (word, score) => (word, score.toDouble) }
      .toArray
  }

  /**
    * Returns a map of words to their vector representations.
    *
    * Note that this implementation pulls the whole distributed matrix to the client and might therefore not work with
    * large matrices which do not fit into the client's memory.
    */
  def getVectors: Map[String, Array[Float]] = {
    val vectors =  Await.result(matrix.pull((0L until numWords).toArray), 3 minutes)
    wordIndex.map { case (word, ind) => (word, vectors(ind).toArray) }
  }

  /**
    * Returns a local [[org.apache.spark.mllib.feature.Word2VecModel Word2VecModel]], the default Spark implementation.
    * This can be used if only the training should be performed with a Glint cluster.
    *
    * Note that this implementation pulls the whole distributed matrix to the client and might therefore not work with
    * large matrices which do not fit into the client's memory.
    *
    * Note also that while this implementation can train large models, the word vectors in the default Spark
    * implementation are limited to 8GB because of the broadcast size limit.
    */
  def toLocal: Word2VecModel = {
    val vectors =  Await.result(matrix.pull((0L until numWords).toArray), 3 minutes)
    new Word2VecModel(wordIndex, vectors.flatMap(_.toArray))
  }

  /**
    * Stops the model and releases the underlying distributed matrix and broadcasts.
    * This model can't be used anymore afterwards.
    *
    * @param sc The Spark context
    */
  def stop(sc: SparkContext): Unit = {
    client.terminateOnSpark(sc)
  }

}

object ServerSideGlintWord2VecModel extends Loader[ServerSideGlintWord2VecModel] {

  private val maximumMessageSize = 10000

  override def load(sc: SparkContext, path: String): ServerSideGlintWord2VecModel = {
    val wordArrayPath = path + "/words"
    val wordIndex = sc.textFile(wordArrayPath, minPartitions = 1).collect().zipWithIndex.toMap
    val (client, matrix) = Client.runWithLoadedWord2VecMatrixOnSpark(sc)(path)
    val granularMatrix = new GranularBigWord2VecMatrix(matrix, maximumMessageSize)
    new ServerSideGlintWord2VecModel(wordIndex, granularMatrix, client)
  }
}