package org.apache.spark.ml.feature

import org.apache.spark.sql.SparkSession
import org.scalatest.concurrent.ScalaFutures
import org.scalatest.time.{Millis, Seconds, Span}
import org.scalatest.{TestData, fixture}

import scala.concurrent.ExecutionContext

/**
  * Provides basic functions for Spark tests
  */
trait SparkTest extends ScalaFutures { this: fixture.TestDataFixture =>

  implicit val ec = ExecutionContext.Implicits.global

  implicit val defaultPatience =
    PatienceConfig(timeout = Span(60, Seconds), interval = Span(500, Millis))

  /**
    * Fixture that starts a Spark session before running test code and stops it afterwards
    *
    * @param testCode The test code to run
    */
  def withSession(testCode: SparkSession => Any): TestData => Any = { td =>
    val session = SparkSession.builder()
      .appName(td.name)
      .getOrCreate()
    try {
      testCode(session)
    } finally {
      session.close()
    }
  }
}
