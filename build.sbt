name := "glint-word2vec"

version := "1.0"

scalaVersion := "2.11.8"
val scalaMajorMinorVersion = "2.11"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.3.0" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.3.0" % "provided"

// use this instead of the github dependency for easier local development if you are modifying glint
// libraryDependencies += "at.mgabr" %% "glint" % "0.2-SNAPSHOT"

lazy val glint = RootProject(uri("https://github.com/MGabr/glint.git#0.2-word2vec"))

// Integration tests

libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.1" % "it"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % "it"


// Lower Aeron buffer to prevent space on /dev/shm running out during local or CI tests

val aeronBufferLength = "-Daeron.term.buffer.length=1048576" // 1024 * 1024
javaOptions in Test += aeronBufferLength
javaOptions in IntegrationTest += aeronBufferLength

// Add it:assembly task to build separate jar containing only the integration test sources

Project.inConfig(IntegrationTest)(baseAssemblySettings)
assemblyJarName in (IntegrationTest, assembly) := s"${name.value}-it-assembly-${version.value}.jar"
test in (IntegrationTest, assembly) := {}
fullClasspath in (IntegrationTest, assembly) := {
  val cp = (fullClasspath in (IntegrationTest, assembly)).value
  cp.filter({ x => Seq("it-classes", "scalatest", "scalactic").exists(x.data.getPath.contains(_)) })
}

// Override it:test task to execute integration tests in Spark docker container

val sparkTestsMain = "org.apache.spark.ml.feature.Main"

import scala.sys.process._

test in IntegrationTest := {
  val startSparkTestEnv = "./spark-test-env.sh"
  val execSparkTests =
    s"""./spark-test-env.sh exec
        spark-submit
        --driver-java-options=$aeronBufferLength
        --conf spark.executor.extraJavaOptions=$aeronBufferLength
        --total-executor-cores 2
        --jars target/scala-$scalaMajorMinorVersion/${name.value}-assembly-${version.value}.jar
        --class $sparkTestsMain
        target/scala-$scalaMajorMinorVersion/${name.value}-it-assembly-${version.value}.jar
    """
  val stopSparkTestEnv = "./spark-test-env.sh stop"
  val rmSparkTestEnv = "./spark-test-env.sh rm"
  val exitCode = (startSparkTestEnv #&& execSparkTests #&& stopSparkTestEnv #&& rmSparkTestEnv !)
  if (exitCode != 0) {
    (stopSparkTestEnv ### rmSparkTestEnv !)
    throw new RuntimeException(s"Integration tests failed with nonzero exit value: $exitCode")
  }
}

test in IntegrationTest := (test in IntegrationTest).dependsOn(
  assembly,
  assembly in IntegrationTest
).value

// Add integration tests to sbt project

lazy val root = (project in file(".")).dependsOn(glint)
  .configs(IntegrationTest)
  .settings(Defaults.itSettings)

