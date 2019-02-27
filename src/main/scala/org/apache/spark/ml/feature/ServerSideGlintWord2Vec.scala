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

package org.apache.spark.ml.feature

import breeze.linalg.convert
import org.apache.spark.annotation.Since
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib.feature.{ServerSideGlintWord2Vec => MLlibServerSideGlintWord2Vec, ServerSideGlintWord2VecModel => MLlibServerSideGlintWord2VecModel}
import org.apache.spark.mllib.linalg.VectorImplicits._
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import scala.concurrent.duration._
import scala.concurrent.{Await, ExecutionContext}

/**
  * Params for [[ServerSideGlintWord2Vec]] and [[ServerSideGlintWord2VecModel]].
  */
private[feature] trait ServerSideGlintWord2VecBase extends Params
  with HasInputCol with HasOutputCol with HasMaxIter with HasStepSize with HasSeed {

  /**
    * The dimension of the code that you want to transform from words.
    * Default: 100
    * @group param
    */
  final val vectorSize = new IntParam(
    this, "vectorSize", "the dimension of codes after transforming from words (> 0)",
    ParamValidators.gt(0))
  setDefault(vectorSize -> 100)

  /** @group getParam */
  def getVectorSize: Int = $(vectorSize)

  /**
    * The window size (context words from [-window, window]).
    * Default: 5
    * @group expertParam
    */
  final val windowSize = new IntParam(
    this, "windowSize", "the window size (context words from [-window, window]) (> 0)",
    ParamValidators.gt(0))
  setDefault(windowSize -> 5)

  /** @group expertGetParam */
  def getWindowSize: Int = $(windowSize)

  /**
    * Number of partitions for sentences of words.
    * Default: 1
    * @group param
    */
  final val numPartitions = new IntParam(
    this, "numPartitions", "number of partitions for sentences of words (> 0)",
    ParamValidators.gt(0))
  setDefault(numPartitions -> 1)

  /** @group getParam */
  def getNumPartitions: Int = $(numPartitions)

  /**
    * The minimum number of times a token must appear to be included in the word2vec model's
    * vocabulary.
    * Default: 5
    * @group param
    */
  final val minCount = new IntParam(this, "minCount", "the minimum number of times a token must " +
    "appear to be included in the word2vec model's vocabulary (>= 0)", ParamValidators.gtEq(0))
  setDefault(minCount -> 5)

  /** @group getParam */
  def getMinCount: Int = $(minCount)

  /**
    * Sets the maximum length (in words) of each sentence in the input data.
    * Any sentence longer than this threshold will be divided into chunks of
    * up to `maxSentenceLength` size.
    * Default: 1000
    * @group param
    */
  final val maxSentenceLength = new IntParam(this, "maxSentenceLength", "Maximum length " +
    "(in words) of each sentence in the input data. Any sentence longer than this threshold will " +
    "be divided into chunks up to the size (> 0)", ParamValidators.gt(0))
  setDefault(maxSentenceLength -> 1000)

  /** @group getParam */
  def getMaxSentenceLength: Int = $(maxSentenceLength)

  /**
    * The mini batch size
    * Default: 50
    */
  final val batchSize = new IntParam(this, "batchSize", "the mini batch size")
  setDefault(batchSize -> 50)

  /** @group getParam */
  def getBatchSize: Int = $(batchSize)

  /**
    * The number of random negative examples
    * Default: 5
    */
  final val n = new IntParam(this, "n", "the number of random negative examples")
  setDefault(n -> 5)

  /** @group getParam */
  def getN: Int = $(n)

  /**
    * The number of parameter servers to create
    * Default: 5
    */
  final val numParameterServers = new IntParam(this, "numParameterServers",
    "the number of parameter servers to create")
  setDefault(numParameterServers -> 5)

  /** @group getParam */
  def getNumParameterServers: Int = $(numParameterServers)

  /**
    * The host name of the master of the parameter servers.
    * Set to "" for automatic detection which may not always work and "127.0.0.1" for local testing
    * Default: ""
    */
  final val parameterServerMasterHost = new Param[String](this, "parameterServerMasterHost",
    "the host name of the master of the parameter servers. Set to \"\" for automatic detection which may not " +
      "always work and \"127.0.0.1\" for local testing")
  setDefault(parameterServerMasterHost -> "")

  /** @group getParam */
  def getParameterServerMasterHost: String = $(parameterServerMasterHost)

  /**
    * The size of the unigram table.
    * Only needs to be changed to a lower value if there is not enough memory for local testing.
    * Default: 100.000.000
    */
  final val unigramTableSize = new IntParam(this, "unigramTableSize", "the size of the " +
    "unigram table. Only needs to be changed to a lower value if there is not enough memory for local testing")
  setDefault(unigramTableSize -> 100000000)

  /** @group getParam */
  def getUnigramTableSize: Int = $(unigramTableSize)

  setDefault(stepSize -> 0.025)
  setDefault(maxIter -> 1)

  /**
    * Validate and transform the input schema.
    */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    val typeCandidates = List(new ArrayType(StringType, true), new ArrayType(StringType, false))
    SchemaUtils.checkColumnTypes(schema, $(inputCol), typeCandidates)
    SchemaUtils.appendColumn(schema, $(outputCol), new VectorUDT)
  }
}

/**
  * ServerSideGlintWord2Vec trains a model of `Map(String, Vector)`, i.e. transforms a word into a code for further
  * natural language processing or machine learning process.
  */
@Since("1.4.0")
final class ServerSideGlintWord2Vec @Since("1.4.0")(
                                       @Since("1.4.0") override val uid: String)
  extends Estimator[ServerSideGlintWord2VecModel] with ServerSideGlintWord2VecBase with DefaultParamsWritable {

  @Since("1.4.0")
  def this() = this(Identifiable.randomUID("gw2v"))

  /** @group setParam */
  @Since("1.4.0")
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  @Since("1.4.0")
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group setParam */
  @Since("1.4.0")
  def setVectorSize(value: Int): this.type = set(vectorSize, value)

  /** @group expertSetParam */
  @Since("1.6.0")
  def setWindowSize(value: Int): this.type = set(windowSize, value)

  /** @group setParam */
  @Since("1.4.0")
  def setStepSize(value: Double): this.type = set(stepSize, value)

  /** @group setParam */
  @Since("1.4.0")
  def setNumPartitions(value: Int): this.type = set(numPartitions, value)

  /** @group setParam */
  @Since("1.4.0")
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /** @group setParam */
  @Since("1.4.0")
  def setSeed(value: Long): this.type = set(seed, value)

  /** @group setParam */
  @Since("1.4.0")
  def setMinCount(value: Int): this.type = set(minCount, value)

  /** @group setParam */
  @Since("2.0.0")
  def setMaxSentenceLength(value: Int): this.type = set(maxSentenceLength, value)

  /** @group setParam */
  def setBatchSize(value: Int): this.type = set(batchSize, value)

  /** @group expertSetParam */
  def setN(value: Int): this.type = set(n, value)

  /** @group setParam */
  def setNumParameterServers(value: Int): this.type = set(numParameterServers, value)

  /** @group setParam */
  def setParameterServerMasterHost(value: String): this.type = set(parameterServerMasterHost, value)

  def setUnigramTableSize(value: Int): this.type = set(unigramTableSize, value)

  @Since("2.0.0")
  override def fit(dataset: Dataset[_]): ServerSideGlintWord2VecModel = {
    transformSchema(dataset.schema, logging = true)
    val input = dataset.select($(inputCol)).rdd.map(_.getAs[Seq[String]](0))
    val wordVectors = new MLlibServerSideGlintWord2Vec()
      .setLearningRate($(stepSize))
      .setMinCount($(minCount))
      .setNumIterations($(maxIter))
      .setNumPartitions($(numPartitions))
      .setSeed($(seed))
      .setVectorSize($(vectorSize))
      .setWindowSize($(windowSize))
      .setMaxSentenceLength($(maxSentenceLength))
      .setBatchSize($(batchSize))
      .setN($(n))
      .setNumParameterServers($(numParameterServers))
      .setParameterServerMasterHost($(parameterServerMasterHost))
      .setUnigramTableSize($(unigramTableSize))
      .fit(input)
    copyValues(new ServerSideGlintWord2VecModel(uid, wordVectors).setParent(this))
  }

  @Since("1.4.0")
  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  @Since("1.4.1")
  override def copy(extra: ParamMap): ServerSideGlintWord2Vec = defaultCopy(extra)
}

@Since("1.6.0")
object ServerSideGlintWord2Vec extends DefaultParamsReadable[ServerSideGlintWord2Vec] {

  @Since("1.6.0")
  override def load(path: String): ServerSideGlintWord2Vec = super.load(path)
}

/**
  * Model fitted by [[ServerSideGlintWord2Vec]].
  */
@Since("1.4.0")
class ServerSideGlintWord2VecModel private[ml](
                                  @Since("1.4.0") override val uid: String,
                                  @transient private val mllibModel: MLlibServerSideGlintWord2VecModel)
  extends Model[ServerSideGlintWord2VecModel] with ServerSideGlintWord2VecBase with MLWritable {

  import ServerSideGlintWord2VecModel._

  /**
    * The number of words in the vocabulary
    */
  val numWords: Int = mllibModel.numWords

  private var bcWordIndexOpt: Option[Broadcast[Map[String, Int]]] = None

  /**
    * Returns a dataframe with two fields, "word" and "vector", with "word" being a String and
    * and the vector the DenseVector that it is mapped to.
    *
    * Note that this implementation pulls the whole distributed matrix to the client and might therefore not work with
    * large matrices which do not fit into the client's memory.
    */
  @Since("1.5.0")
  @transient lazy val getVectors: DataFrame = {
    val spark = SparkSession.builder().getOrCreate()
    val wordVec = mllibModel.getVectors.mapValues(vec => Vectors.dense(vec.map(_.toDouble)))
    spark.createDataFrame(wordVec.toSeq).toDF("word", "vector")
  }

  /**
    * Find "num" number of words closest in similarity to the given word, not
    * including the word itself.
    *
    * Note that this implementation makes a blocking call to the underlying distributed matrix.
    *
    * @return a dataframe with columns "word" and "similarity" of the word and the cosine
    * similarities between the synonyms and the given word.
    */
  @Since("1.5.0")
  def findSynonyms(word: String, num: Int): DataFrame = {
    val spark = SparkSession.builder().getOrCreate()
    spark.createDataFrame(findSynonymsArray(word, num)).toDF("word", "similarity")
  }

  /**
    * Find "num" number of words whose vector representation is most similar to the supplied vector.
    * If the supplied vector is the vector representation of a word in the model's vocabulary,
    * that word will be in the results.
    *
    * Note that this implementation makes a blocking call to the underlying distributed matrix.
    *
    * @return a dataframe with columns "word" and "similarity" of the word and the cosine
    * similarities between the synonyms and the given word vector.
    */
  @Since("2.0.0")
  def findSynonyms(vec: Vector, num: Int): DataFrame = {
    val spark = SparkSession.builder().getOrCreate()
    spark.createDataFrame(findSynonymsArray(vec, num)).toDF("word", "similarity")
  }

  /**
    * Find "num" number of words whose vector representation is most similar to the supplied vector.
    * If the supplied vector is the vector representation of a word in the model's vocabulary,
    * that word will be in the results.
    *
    * Note that this implementation makes a blocking call to the underlying distributed matrix.
    *
    * @return an array of the words and the cosine similarities between the synonyms given
    * word vector.
    */
  @Since("2.2.0")
  def findSynonymsArray(vec: Vector, num: Int): Array[(String, Double)] = {
    mllibModel.findSynonyms(vec, num)
  }

  /**
    * Find "num" number of words closest in similarity to the given word, not
    * including the word itself.
    *
    * Note that this implementation makes a blocking call to the underlying distributed matrix.
    *
    * @return an array of the words and the cosine similarities between the synonyms given
    * word vector.
    */
  @Since("2.2.0")
  def findSynonymsArray(word: String, num: Int): Array[(String, Double)] = {
    mllibModel.findSynonyms(word, num)
  }

  /** @group setParam */
  @Since("1.4.0")
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  @Since("1.4.0")
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /**
    * Transform a sentence column to a vector column to represent the whole sentence. The transform
    * is performed by averaging all word vectors it contains.
    */
  @Since("2.0.0")
  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformedSchema = transformSchema(dataset.schema, logging = true)

    val bcWordIndex = bcWordIndexOpt.getOrElse {
      val bcWordIndex = SparkSession.builder().getOrCreate().sparkContext.broadcast(mllibModel.wordIndex)
      bcWordIndexOpt = Some(bcWordIndex)
      bcWordIndex
    }
    val matrix = mllibModel.matrix

    // use mapPartitions instead of withColumn and udf to prevent a blocking parameter server request per sentence
    dataset.toDF().mapPartitions { rowIter =>
      @transient
      implicit val ec = ExecutionContext.Implicits.global
      val wordIndex = bcWordIndex.value

      // make requests to parameter server with 10.000 sentence batches
      val rowSlidesIter = rowIter.sliding(10000, 10000).withPartial(true)
      val vectors = rowSlidesIter.flatMap { rows =>
        val sentences = rows.toArray.map(_.getAs[Seq[String]]($(inputCol)).toArray)
        val sentenceIndices = sentences.map(_.flatMap(wordIndex.get).map(_.toLong))
        val averageVecs = Await.result(matrix.pullAverage(sentenceIndices), 1 minute)
        rows.zip(averageVecs).map { case (row, vec) =>
          Row.fromSeq(row.toSeq :+ Vectors.fromBreeze(convert(vec, Double)))
        }
      }
      vectors
    }(RowEncoder(transformedSchema))
  }

  @Since("1.4.0")
  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  @Since("1.4.1")
  override def copy(extra: ParamMap): ServerSideGlintWord2VecModel = {
    val copied = new ServerSideGlintWord2VecModel(uid, mllibModel)
    copyValues(copied, extra).setParent(parent)
  }

  @Since("1.6.0")
  override def write: MLWriter = new ServerSideGlintWord2VecModelWriter(this)

  /**
    * Returns a local [[org.apache.spark.ml.feature.Word2VecModel Word2VecModel]], the default Spark implementation.
    * This can be used if only the training should be performed with a Glint cluster.
    *
    * Note that this implementation pulls the whole distributed matrix to the client and might therefore not work with
    * large matrices which do not fit into the client's memory.
    *
    * Note also that while this implementation can train large models, the word vectors in the default Spark
    * implementation are limited to 8GB because of the broadcast size limit.
    */
  def toLocal: Word2VecModel = new Word2VecModel(Identifiable.randomUID("w2v"), mllibModel.toLocal)

  /**
    * Stops the model and releases the underlying distributed matrix and broadcasts.
    * This model can't be used anymore afterwards.
    */
  def stop(): Unit = {
    bcWordIndexOpt.foreach(_.destroy())
    mllibModel.stop(SparkSession.builder().getOrCreate().sparkContext)
  }
}

@Since("1.6.0")
object ServerSideGlintWord2VecModel extends MLReadable[ServerSideGlintWord2VecModel] {

  private[ServerSideGlintWord2VecModel]
  class ServerSideGlintWord2VecModelWriter(instance: ServerSideGlintWord2VecModel) extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      instance.mllibModel.save(sparkSession.sparkContext, path)
    }
  }

  private class ServerSideGlintWord2VecModelReader extends MLReader[ServerSideGlintWord2VecModel] {

    private val className = classOf[ServerSideGlintWord2VecModel].getName

    override def load(path: String): ServerSideGlintWord2VecModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val oldModel = MLlibServerSideGlintWord2VecModel.load(sparkSession.sparkContext, path)
      val model = new ServerSideGlintWord2VecModel(metadata.uid, oldModel)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

  @Since("1.6.0")
  override def read: MLReader[ServerSideGlintWord2VecModel] = new ServerSideGlintWord2VecModelReader

  @Since("1.6.0")
  override def load(path: String): ServerSideGlintWord2VecModel = super.load(path)
}

