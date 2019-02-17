package org.apache.spark.ml.feature

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.scalatest.{Matchers, fixture}

/**
  * ServerSideGlintWord2Vec integration test specification
  */
class ServerSideGlintWord2VecSpec extends fixture.FlatSpec with fixture.TestDataFixture with SparkTest with Matchers {

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
    * Created with a script from https://github.com/MGabr/evaluate-glint-word2vec
    */
  private val testdataPath = "de_wikipedia_articles_country_capitals.txt"

  /**
    * Path to save model to. The first test will create it and subsequent tests will rely on it being present
    */
  private val modelPath = "/var/tmp/de_wikipedia_articles_country_capitals.model"

  private var modelCreated = false


  "ServerSideGlintWord2Vec" should "train and save a model" in withSession { s =>

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

    model.save(modelPath)

    modelCreated = true
  }

  it should "load a model" in withSession { s =>

    if (!modelCreated) {
      pending
    }

    val model = ServerSideGlintWord2VecModel.load(modelPath)

    model.getSeed shouldBe 1
    model.getNumPartitions shouldBe 2
    model.getNumParameterServers shouldBe 2
    model.getInputCol shouldBe "sentence"
    model.getOutputCol shouldBe "model"
    model.getUnigramTableSize shouldBe 1000000

    model.getVectorSize shouldBe 100
  }

  it should "transform a data frame" in withSession { s =>

    if (!modelCreated) {
      pending
    }

    import s.sqlContext.implicits._
    val countries = Seq(Seq("österreich"), Seq("deutschland"), Seq("frankreich"), Seq("spanien"))
    val countriesDf = s.sparkContext.parallelize(countries).toDF("sentence")

    val model = ServerSideGlintWord2VecModel.load(modelPath)

    val countryVecs = model.transform(countriesDf).collect().map(row => row.getAs[Vector]("model"))

    countryVecs should have length countries.length
    for (countryVec <- countryVecs) {
      countryVec.asBreeze should have length model.getVectorSize
      countryVec.asBreeze.sum should not equal 0
    }
  }

  it should "find synonyms of a word as array" in withSession { s =>

    if (!modelCreated) {
      pending
    }

    val model = ServerSideGlintWord2VecModel.load(modelPath)

    val (synonyms, similarities) = model.findSynonymsArray("frankreich", 10).unzip

    synonyms should have length 10
    synonyms should contain("paris")
    similarities(synonyms.indexOf("paris")) should be > 0.9
  }

  it should "find synonyms of a word as data frame" in withSession { s =>

    if (!modelCreated) {
      pending
    }

    val model = ServerSideGlintWord2VecModel.load(modelPath)

    val (synonyms, similarities) = model.findSynonyms("frankreich", 10)
      .collect()
      .map(row => (row.getAs[String]("word"), row.getAs[Double]("similarity")))
      .unzip

    synonyms should have length 10
    synonyms should contain("paris")
    similarities(synonyms.indexOf("paris")) should be > 0.9
  }

  it should "find analogies of a word vector as array" in withSession { s =>

    if (!modelCreated) {
      pending
    }

    import s.sqlContext.implicits._
    val countries = Seq(Seq("österreich"), Seq("deutschland"))
    val countriesDf = s.sparkContext.parallelize(countries).toDF("sentence")
    val capitals = Seq(Seq("wien"), Seq("berlin"))
    val capitalsDf = s.sparkContext.parallelize(capitals).toDF("sentence")

    val model = ServerSideGlintWord2VecModel.load(modelPath)

    val countryVecs = model.transform(countriesDf).collect().map(row => row.getAs[Vector]("model"))
    val capitalsVecs = model.transform(capitalsDf).collect().map(row => row.getAs[Vector]("model"))
    val capitalAnalogyVec = capitalsVecs(0).asBreeze - countryVecs(0).asBreeze + countryVecs(1).asBreeze

    val (analogies, similarities) = model.findSynonymsArray(Vectors.fromBreeze(capitalAnalogyVec), 10).unzip

    analogies should have length 10
    analogies should contain("berlin")
    similarities(analogies.indexOf("berlin")) should be > 0.9
  }

  it should "find analogies of a word vector as data frame" in withSession { s =>

    if (!modelCreated) {
      pending
    }

    import s.sqlContext.implicits._
    val countries = Seq(Seq("österreich"), Seq("deutschland"))
    val countriesDf = s.sparkContext.parallelize(countries).toDF("sentence")
    val capitals = Seq(Seq("wien"), Seq("berlin"))
    val capitalsDf = s.sparkContext.parallelize(capitals).toDF("sentence")

    val model = ServerSideGlintWord2VecModel.load(modelPath)

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
  }
}
