package org.apache.spark.ml.feature

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, FileUtil, Path}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.mllib
import org.apache.spark.sql.SparkSession
import org.scalatest.concurrent.ScalaFutures
import org.scalatest.{BeforeAndAfterAll, FlatSpec, Inspectors, Matchers}

import scala.concurrent.ExecutionContext

/**
  * ServerSideGlintWord2Vec integration test specification
  */
class ServerSideGlintWord2VecSpec extends FlatSpec with ScalaFutures with BeforeAndAfterAll
  with Matchers with Inspectors {

  /**
    * Path to small test data set consisting of wikipedia articles about countries and capital cities.
    * Contains the german wikipedia articles of
    *
    * Österreich (Austria), Wien (Vienna)
    * Deutschland (Germany), Berlin
    * Frankreich (France), Paris
    * Spanien (Spain), Madrid
    * Großbritannien (UK), London
    * Finnland, Helsinki
    *
    * Results in a vocabulary of 3611 distinct words occuring at least 5 times.
    *
    * Created with a script from https://github.com/MGabr/evaluate-glint-word2vec
    */
  private val testdataPath = "de_wikipedia_articles_country_capitals.txt"

  /**
    * Path to save model to. The first test will create it and subsequent tests will rely on it being present
    */
  private val modelPath = "/var/tmp/de_wikipedia_articles_country_capitals.model"

  private var modelCreated = false

  /**
    * Path to save the local model, the default Spark implementation, to
    */
  private val localModelPath = "/var/tmp/local/de_wikipedia_articles_country_capitals.model"

  /**
    * The Spark session to use
    */
  private lazy val s: SparkSession = SparkSession.builder().appName(getClass.getSimpleName).getOrCreate()


  implicit val ec = ExecutionContext.Implicits.global


  override def beforeAll(): Unit = {
    super.beforeAll()

    val fs = FileSystem.get(new Configuration())
    FileUtil.fullyDelete(fs, fs.getHomeDirectory)
    fs.copyFromLocalFile(new Path(testdataPath), new Path(testdataPath))
  }

  override def afterAll(): Unit = {
    s.stop()
  }

  "ServerSideGlintWord2Vec" should "train and save a model" in {
    import s.sqlContext.implicits._
    val sentences = s.sparkContext.textFile(testdataPath).map(row => row.split(" ")).toDF("sentence")

    val word2vec = new ServerSideGlintWord2Vec()
      .setSeed(1)
      .setNumPartitions(2)
      .setNumParameterServers(2)
      .setInputCol("sentence")
      .setOutputCol("model")
      .setUnigramTableSize(1000000)

    val model = word2vec.fit(sentences)
    try {
      model.save(modelPath)

      FileSystem.get(s.sparkContext.hadoopConfiguration).exists(new Path(modelPath)) shouldBe true

      modelCreated = true
    } finally {
      model.stop()
    }
  }

  it should "load a model" in {
    if (!modelCreated) {
      pending
    }

    val model = ServerSideGlintWord2VecModel.load(modelPath)
    try {
      model.getSeed shouldBe 1
      model.getNumPartitions shouldBe 2
      model.getNumParameterServers shouldBe 2
      model.getInputCol shouldBe "sentence"
      model.getOutputCol shouldBe "model"
      model.getUnigramTableSize shouldBe 1000000

      model.getVectorSize shouldBe 100
    } finally {
      model.stop()
    }
  }

  it should "transform a data frame" in {
    if (!modelCreated) {
      pending
    }

    import s.sqlContext.implicits._
    val countries = Seq(Seq("österreich"), Seq("deutschland"), Seq("frankreich"), Seq("spanien"))
    val countriesDf = s.sparkContext.parallelize(countries).toDF("sentence")

    val model = ServerSideGlintWord2VecModel.load(modelPath)
    try {
      val countryVecs = model.transform(countriesDf).collect().map(row => row.getAs[Vector]("model").asBreeze)

      countryVecs should have length countries.length
      all (countryVecs) should have length model.getVectorSize
      forAll (countryVecs) {countryVec => countryVec.sum should not be equal(0.0) }
    } finally {
      model.stop()
    }
  }

  // note that the next test "transform an RDD" presents a more efficient implementation of the same task
  it should "transform single words" in {
    if (!modelCreated) {
      pending
    }

    val countries = Seq("österreich", "deutschland", "frankreich", "spanien")
    val countriesRDD = s.sparkContext.parallelize(countries)

    val model = mllib.feature.ServerSideGlintWord2VecModel.load(s.sparkContext, modelPath)
    try {
      val countryVecs = countriesRDD.map(model.transform).collect().map(_.asBreeze)

      countryVecs should have length countries.length
      all (countryVecs) should have length model.vectorSize
      forAll (countryVecs) {countryVec => countryVec.sum should not be equal(0.0) }
    } finally {
      model.stop(s.sparkContext)
    }
  }

  it should "transform an RDD" in {
    if (!modelCreated) {
      pending
    }

    val countries = Seq("österreich", "deutschland", "frankreich", "spanien")
    val countriesRDD = s.sparkContext.parallelize(countries)

    val model = mllib.feature.ServerSideGlintWord2VecModel.load(s.sparkContext, modelPath)
    try {
      val countryVecs = countriesRDD.mapPartitions(model.transform).collect().map(_.asBreeze)

      countryVecs should have length countries.length
      all (countryVecs) should have length model.vectorSize
      forAll (countryVecs) {countryVec => countryVec.sum should not be equal(0.0) }
    } finally {
      model.stop(s.sparkContext)
    }
  }

  it should "transform a data frame with multiple columns" in {
    if (!modelCreated) {
      pending
    }

    import s.sqlContext.implicits._
    val data = Seq(
      (0, Seq("österreich"), 8.7, "wien", 1.9),
      (1, Seq("deutschland"), 82.8, "berlin", 3.6),
      (2, Seq("frankreich"), 67.1, "paris", 2.2),
      (3, Seq("spanien"), 46.6, "madrid", 3.2)
    )
    val colNames = Array("id", "sentence", "country inhabitants", "city", "city inhabitants")
    val dataDf = s.sparkContext.parallelize(data).toDF(colNames : _*)

    val model = ServerSideGlintWord2VecModel.load(modelPath)
    try {
      val vecDf = model.transform(dataDf)
      val vecColNames = vecDf.schema.fieldNames
      val vecs = vecDf.collect().map(row => row.getAs[Vector]("model").asBreeze)

      vecColNames should equal(colNames :+ "model")
      vecs should have length data.length
      all (vecs) should have length model.getVectorSize
      forAll (vecs) {vec => vec.sum should not be equal(0.0)}
    } finally {
      model.stop()
    }
  }

  it should "find synonyms of a word as array" in {
    if (!modelCreated) {
      pending
    }

    val model = ServerSideGlintWord2VecModel.load(modelPath)
    try {
      val (synonyms, similarities) = model.findSynonymsArray("frankreich", 10).unzip

      synonyms should have length 10
      synonyms should contain("paris")
      similarities(synonyms.indexOf("paris")) should be > 0.9
    } finally {
      model.stop()
    }
  }

  it should "find synonyms of a word as data frame" in {
    if (!modelCreated) {
      pending
    }

    val model = ServerSideGlintWord2VecModel.load(modelPath)
    try {
      val (synonyms, similarities) = model.findSynonyms("frankreich", 10)
        .collect()
        .map(row => (row.getAs[String]("word"), row.getAs[Double]("similarity")))
        .unzip

      synonyms should have length 10
      synonyms should contain("paris")
      similarities(synonyms.indexOf("paris")) should be > 0.9
    } finally {
      model.stop()
    }
  }

  it should "find analogies of a word vector as array" in {
    if (!modelCreated) {
      pending
    }

    import s.sqlContext.implicits._
    val countries = Seq(Seq("österreich"), Seq("deutschland"))
    val countriesDf = s.sparkContext.parallelize(countries).toDF("sentence")
    val capitals = Seq(Seq("wien"), Seq("berlin"))
    val capitalsDf = s.sparkContext.parallelize(capitals).toDF("sentence")

    val model = ServerSideGlintWord2VecModel.load(modelPath)
    try {
      val countryVecs = model.transform(countriesDf).collect().map(row => row.getAs[Vector]("model"))
      val capitalsVecs = model.transform(capitalsDf).collect().map(row => row.getAs[Vector]("model"))
      val capitalAnalogyVec = capitalsVecs(0).asBreeze - countryVecs(0).asBreeze + countryVecs(1).asBreeze

      val (analogies, similarities) = model.findSynonymsArray(Vectors.fromBreeze(capitalAnalogyVec), 10).unzip

      analogies should have length 10
      analogies should contain("berlin")
      similarities(analogies.indexOf("berlin")) should be > 0.9
    } finally {
      model.stop()
    }
  }

  it should "find analogies of a word vector as data frame" in {
    if (!modelCreated) {
      pending
    }

    import s.sqlContext.implicits._
    val countries = Seq(Seq("österreich"), Seq("deutschland"))
    val countriesDf = s.sparkContext.parallelize(countries).toDF("sentence")
    val capitals = Seq(Seq("wien"), Seq("berlin"))
    val capitalsDf = s.sparkContext.parallelize(capitals).toDF("sentence")

    val model = ServerSideGlintWord2VecModel.load(modelPath)
    try {
      val countryVecs = model.transform(countriesDf).collect().map(row => row.getAs[Vector]("model"))
      val capitalsVecs = model.transform(capitalsDf).collect().map(row => row.getAs[Vector]("model"))
      val capitalAnalogyVec = capitalsVecs(0).asBreeze - countryVecs(0).asBreeze + countryVecs(1).asBreeze

      val (analogies, similarities) = model.findSynonyms(Vectors.fromBreeze(capitalAnalogyVec), 10)
        .collect()
        .map(row => (row.getAs[String]("word"), row.getAs[Double]("similarity")))
        .unzip

      analogies should have length 10
      analogies should contain("berlin")
      similarities(analogies.indexOf("berlin")) should be > 0.9
    } finally {
      model.stop()
    }
  }

  it should "get all word vectors" in {
    if (!modelCreated) {
      pending
    }

    val model = ServerSideGlintWord2VecModel.load(modelPath)
    try {
      val vectorDf = model.getVectors

      vectorDf.schema.fieldNames should equal(Array("word", "vector"))
      vectorDf.count().toInt shouldBe model.numWords
    } finally {
      model.stop()
    }
  }

  it should "convert and save as local model" in {
    if (!modelCreated) {
      pending
    }

    val model = ServerSideGlintWord2VecModel.load(modelPath)
    try {
      val localModel = model.toLocal
      localModel.save(localModelPath)

      localModel shouldBe a[Word2VecModel]
      FileSystem.get(s.sparkContext.hadoopConfiguration).exists(new Path(localModelPath)) shouldBe true
    } finally {
      model.stop()
    }
  }
}
