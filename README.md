# Glint-Word2Vec
[![Build Status](https://travis-ci.com/MGabr/glint-word2vec.svg)](https://travis-ci.com/MGabr/glint-word2vec)

> Network-efficient distributed Word2Vec for large vocabularies on Spark 
using customized [Glint](https://github.com/MGabr/glint) parameter servers

This Word2Vec implementation allows training large vocabularies which do not fit in the memory of a single machine.
It uses parameter servers to achieve this and custom parameter server operations to perform the training in a
network-efficient way. The vectors are trained asynchronously in mini-batches as skip-gram model with negative sampling. 
This approach is based on 

Erik Ordentlich, Lee Yang, Andy Feng, Peter Cnudde, Mihajlo Grbovic,
Nemanja Djuric, Vladan Radosavljevic and Gavin Owens.
**"Network-Efficient Distributed Word2vec Training System for Large Vocabularies."**
*In CIKM, 2016, Pages 1139-1148*

The asynchronous distributed training is sensitive to very frequent words and may result in exploding gradients.
Therefore stop words and other very frequent words without much meaning should be removed beforehand. Use for example
Sparks [StopWordsRemover](https://spark.apache.org/docs/2.2.0/ml-features.html#stopwordsremover).

Large parts of the actual functionality can be found in the [Glint fork](https://github.com/MGabr/glint).

## Build

You can either use the [release jar](https://github.com/MGabr/glint-word2vec/releases) or build the project yourself.
To build it run:

    sbt assembly
 
To also execute integration tests run:

    sbt it:test

The resulting fat jar contains all dependencies and can be passed to `spark-submit` with `--jar`.
To use the python bindings zip them and pass them as `--py-files`.

    cd src/main/python
    zip ml_glintword2vec.zip ml_glintword2vec.py

## Usage

The API is similar to the existing Word2Vec implementation of Spark and implements the Spark ML and MLlib
interfaces.

There are two modes in which Glint-Word2vec can be run. 
You can either start the parameter servers automatically on some executors in the same Spark application
or start up a parameter server cluster in a separate Spark application beforehand and then specify the
IP of the parameter server master to use this cluster. The first mode is more convenient but the second
mode scales better so it is recommended to use a separate parameter server at least for training and 
the integrated parameter servers only when transforming words to vectors.

To start parameter servers as separate Spark application run:

    spark-submit --num-executors num-servers --executor-cores server-cores --class glint.Main /path/to/compiled/Glint-Word2Vec.jar spark

The parameter server master will be started on the driver and the drivers IP will be written to the log output.
Pass this IP as `parameterServerHost` to connect to these parameter servers from the Glint-Word2Vec Spark application. 

More information can be found in the [Scaladoc](https://mgabr.github.io/glint-word2vec/latest/api/) of this project.

Scala examples can be found in the [integration tests](https://github.com/MGabr/glint-word2vec/blob/master/src/it/scala/org/apache/spark/ml/feature/ServerSideGlintWord2VecSpec.scala)
and Python examples can be found in the [evaluate-glint-word2vec](https://github.com/MGabr/evaluate-glint-word2vec) project.

## Spark parameters

Use a higher number of executor cores (`--executor-cores`) instead of more executors (`--num-executors`).
Preferably set `--executor-cores` to the number of available virtual cores per machine.

Each of the n parameter servers will require enough executor memory (`--executor-memory`) to store 1/n of the word vector matrix.

Currently, for transforming words to vectors the map of words to indices still has to be broadcasted to all
executors. Because of Sparks 8GB broadcast size limit this means that the vocabulary has to be below 80 million words 
assuming an average word length of 10.
