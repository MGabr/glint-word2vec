name := "glint-word2vec"

version := "1.0"

scalaVersion := "2.11.11"

libraryDependencies += "org.apache.spark" % "spark-core_2.11" % "2.3.2" % "provided"
libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "2.3.2" % "provided"
libraryDependencies += "ch.ethz.inf.da" %% "glint" % "0.2-SNAPSHOT"
