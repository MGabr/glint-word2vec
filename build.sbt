name := "glint-word2vec"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.3.0" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.3.0" % "provided"

// use this instead of the github dependency for local development
// libraryDependencies += "ch.ethz.inf.da" %% "glint" % "0.2-SNAPSHOT"

lazy val root = (project in file(".")).dependsOn(glint)
lazy val glint = RootProject(uri("https://github.com/MGabr/glint.git#0.2-word2vec"))
