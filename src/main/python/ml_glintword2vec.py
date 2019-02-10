#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
if sys.version > '3':
    basestring = str

from pyspark import since, keyword_only
from pyspark.rdd import ignore_unicode_prefix
from pyspark.ml import feature
from pyspark.ml.linalg import _convert_to_vector
from pyspark.ml.param.shared import *
from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.ml.wrapper import JavaEstimator, JavaModel
from pyspark.ml.common import inherit_doc

__all__ = ['ServerSideGlintWord2Vec', 'ServerSideGlintWord2VecModel']



@inherit_doc
@ignore_unicode_prefix
class ServerSideGlintWord2Vec(JavaEstimator, HasStepSize, HasMaxIter, HasSeed, HasInputCol, HasOutputCol,
                         JavaMLReadable, JavaMLWritable):
    """
    ServerSideGlintWord2Vec trains a model of `Map(String, Vector)`, i.e. transforms a word into a code for further
    natural language processing or machine learning process.
    >>> sent = ("a b " * 100 + "a c " * 10).split(" ")
    >>> doc = spark.createDataFrame([(sent,), (sent,)], ["sentence"])
    >>> word2Vec = ServerSideGlintWord2Vec(vectorSize=5, seed=42, inputCol="sentence", outputCol="model")
    >>> model = word2Vec.fit(doc)
    >>> model.getVectors().show()
    +----+--------------------+
    |word|              vector|
    +----+--------------------+
    |   a|[0.09461779892444...|
    |   b|[1.15474212169647...|
    |   c|[-0.3794820010662...|
    +----+--------------------+
    ...
    >>> model.findSynonymsArray("a", 2)
    [(u'b', 0.25053444504737854), (u'c', -0.6980510950088501)]
    >>> from pyspark.sql.functions import format_number as fmt
    >>> model.findSynonyms("a", 2).select("word", fmt("similarity", 5).alias("similarity")).show()
    +----+----------+
    |word|similarity|
    +----+----------+
    |   b|   0.25053|
    |   c|  -0.69805|
    +----+----------+
    ...
    >>> model.transform(doc).head().model
    DenseVector([0.5524, -0.4995, -0.3599, 0.0241, 0.3461])
    >>> word2vecPath = temp_path + "/word2vec"
    >>> word2Vec.save(word2vecPath)
    >>> loadedWord2Vec = ServerSideGlintWord2Vec.load(word2vecPath)
    >>> loadedWord2Vec.getVectorSize() == word2Vec.getVectorSize()
    True
    >>> loadedWord2Vec.getNumPartitions() == word2Vec.getNumPartitions()
    True
    >>> loadedWord2Vec.getMinCount() == word2Vec.getMinCount()
    True
    >>> modelPath = temp_path + "/word2vec-model"
    >>> model.save(modelPath)
    >>> loadedModel = ServerSideGlintWord2VecModel.load(modelPath)
    >>> loadedModel.getVectors().first().word == model.getVectors().first().word
    True
    >>> loadedModel.getVectors().first().vector == model.getVectors().first().vector
    True
    .. versionadded:: 1.4.0
    """

    __module__ = "pyspark.ml.feature"

    vectorSize = Param(Params._dummy(), "vectorSize",
                       "the dimension of codes after transforming from words",
                       typeConverter=TypeConverters.toInt)
    numPartitions = Param(Params._dummy(), "numPartitions",
                          "number of partitions for sentences of words",
                          typeConverter=TypeConverters.toInt)
    minCount = Param(Params._dummy(), "minCount",
                     "the minimum number of times a token must appear to be included in the " +
                     "word2vec model's vocabulary", typeConverter=TypeConverters.toInt)
    windowSize = Param(Params._dummy(), "windowSize",
                       "the window size (context words from [-window, window]). Default value is 5",
                       typeConverter=TypeConverters.toInt)
    maxSentenceLength = Param(Params._dummy(), "maxSentenceLength",
                              "Maximum length (in words) of each sentence in the input data. " +
                              "Any sentence longer than this threshold will " +
                              "be divided into chunks up to the size.",
                              typeConverter=TypeConverters.toInt)

    batchSize = Param(Params._dummy(), "batchSize",
                      "the mini batch size",
                      typeConverter=TypeConverters.toInt)
    n = Param(Params._dummy(), "n",
              "the number of random negative examples",
              typeConverter=TypeConverters.toInt)
    numParameterServers = Param(Params._dummy(), "numParameterServers",
                                "the number of parameter servers to create",
                                typeConverter=TypeConverters.toInt)
    parameterServerMasterHost = Param(Params._dummy(), "parameterServerMasterHost",
                                      "the host name of the master of the parameter servers. Set to \"\" for " +
                                      "automatic detection which may not always work and \"127.0.0.1\" for local " +
                                      "testing",
                                      typeConverter=TypeConverters.toString)
    unigramTableSize = Param(Params._dummy(), "unigramTableSize",
                             "the size of the unigram table. Only needs to be changed to a lower value if there is " +
                             "not enough memory for local testing")

    @keyword_only
    def __init__(self, vectorSize=100, minCount=5, numPartitions=1, stepSize=0.025, maxIter=1,
                 seed=None, inputCol=None, outputCol=None, windowSize=5, maxSentenceLength=1000,
                 batchSize=50, n=5, numParameterServers=5, parameterServerMasterHost="",
                 unigramTableSize=100000000):
        """
        __init__(self, vectorSize=100, minCount=5, numPartitions=1, stepSize=0.025, maxIter=1, \
                 seed=None, inputCol=None, outputCol=None, windowSize=5, maxSentenceLength=1000, \
                 batchSize=50, n=5, numParameterServers=5, parameterServerMasterHost="",
                 unigramTableSize=100000000)
        """
        super(ServerSideGlintWord2Vec, self).__init__()
        self._java_obj = self._new_java_obj("org.apache.spark.ml.feature.ServerSideGlintWord2Vec", self.uid)
        self._setDefault(vectorSize=100, minCount=5, numPartitions=1, stepSize=0.025, maxIter=1,
                         windowSize=5, maxSentenceLength=1000, batchSize=50, n=5, numParameterServers=5,
                         parameterServerMasterHost="", unigramTableSize=100000000)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    @since("1.4.0")
    def setParams(self, vectorSize=100, minCount=5, numPartitions=1, stepSize=0.025, maxIter=1,
                  seed=None, inputCol=None, outputCol=None, windowSize=5, maxSentenceLength=1000,
                  batchSize=50, n=5, numParameterServers=5, parameterServerMasterHost="",
                  unigramTableSize=100000000):
        """
        setParams(self, minCount=5, numPartitions=1, stepSize=0.025, maxIter=1, seed=None, \
                 inputCol=None, outputCol=None, windowSize=5, maxSentenceLength=1000, \
                 batchSize=50, n=5, numParameterServers=5, parameterServerMasterHost="",
                 unigramTableSize=100000000)
        Sets params for this ServerSideGlintWord2Vec.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    @since("1.4.0")
    def setVectorSize(self, value):
        """
        Sets the value of :py:attr:`vectorSize`.
        """
        return self._set(vectorSize=value)

    @since("1.4.0")
    def getVectorSize(self):
        """
        Gets the value of vectorSize or its default value.
        """
        return self.getOrDefault(self.vectorSize)

    @since("1.4.0")
    def setNumPartitions(self, value):
        """
        Sets the value of :py:attr:`numPartitions`.
        """
        return self._set(numPartitions=value)

    @since("1.4.0")
    def getNumPartitions(self):
        """
        Gets the value of numPartitions or its default value.
        """
        return self.getOrDefault(self.numPartitions)

    @since("1.4.0")
    def setMinCount(self, value):
        """
        Sets the value of :py:attr:`minCount`.
        """
        return self._set(minCount=value)

    @since("1.4.0")
    def getMinCount(self):
        """
        Gets the value of minCount or its default value.
        """
        return self.getOrDefault(self.minCount)

    @since("2.0.0")
    def setWindowSize(self, value):
        """
        Sets the value of :py:attr:`windowSize`.
        """
        return self._set(windowSize=value)

    @since("2.0.0")
    def getWindowSize(self):
        """
        Gets the value of windowSize or its default value.
        """
        return self.getOrDefault(self.windowSize)

    @since("2.0.0")
    def setMaxSentenceLength(self, value):
        """
        Sets the value of :py:attr:`maxSentenceLength`.
        """
        return self._set(maxSentenceLength=value)

    @since("2.0.0")
    def getMaxSentenceLength(self):
        """
        Gets the value of maxSentenceLength or its default value.
        """
        return self.getOrDefault(self.maxSentenceLength)

    def setBatchSize(self, value):
        """
        Sets the value of :py:attr:`batchSize`.
        """
        return self._set(batchSize=value)

    def getBatchSize(self):
        """
        Gets the value of batchSize or its default value.
        """
        return self.getOrDefault(self.batchSize)

    def setN(self, value):
        """
        Sets the value of :py:attr:`n`.
        """
        return self._set(n=value)

    def getN(self):
        """
        Gets the value of n or its default value.
        """
        return self.getOrDefault(self.n)

    def setNumParameterServers(self, value):
        """
        Sets the value of :py:attr:`numParameterServers`.
        """
        return self._set(numParameterServers=value)

    def getNumParameterServers(self):
        """
        Gets the value of numParameterServers or its default value.
        """
        return self.getOrDefault(self.numParameterServers)

    def setParameterServerMasterHost(self, value):
        """
        Sets the value of :py:attr:`parameterServerMasterHost`.
        """
        return self._set(parameterServerMasterHost=value)

    def getParameterServerMasterHost(self):
        """
        Gets the value of parameterServerMasterHost or its default value.
        """
        return self.getOrDefault(self.parameterServerMasterHost)

    def setUnigramTableSize(self, value):
        """
        Sets the value of :py:attr:`unigramTableSize`.
        """
        return self._set(unigramTableSize=value)

    def getUnigramTableSize(self):
        """
        Gets the value of unigramTableSize or its default value.
        """
        return self.getOrDefault(self.unigramTableSize)

    def _create_model(self, java_model):
        return ServerSideGlintWord2VecModel(java_model)


feature.ServerSideGlintWord2Vec = ServerSideGlintWord2Vec


class ServerSideGlintWord2VecModel(JavaModel, JavaMLReadable, JavaMLWritable):
    """
    Model fitted by :py:class:`ServerSideGlintWord2Vec`.
    .. versionadded:: 1.4.0
    """

    __module__ = "pyspark.ml.feature"

    @since("1.5.0")
    def getVectors(self):
        """
        Returns the vector representation of the words as a dataframe
        with two fields, word and vector.
        """
        return self._call_java("getVectors")

    @since("1.5.0")
    def findSynonyms(self, word, num):
        """
        Find "num" number of words closest in similarity to "word".
        word can be a string or vector representation.
        Returns a dataframe with two fields word and similarity (which
        gives the cosine similarity).
        """
        if not isinstance(word, basestring):
            word = _convert_to_vector(word)
        return self._call_java("findSynonyms", word, num)

    @since("2.3.0")
    def findSynonymsArray(self, word, num):
        """
        Find "num" number of words closest in similarity to "word".
        word can be a string or vector representation.
        Returns an array with two fields word and similarity (which
        gives the cosine similarity).
        """
        if not isinstance(word, basestring):
            word = _convert_to_vector(word)
        tuples = self._java_obj.findSynonymsArray(word, num)
        return list(map(lambda st: (st._1(), st._2()), list(tuples)))


feature.ServerSideGlintWord2VecModel = ServerSideGlintWord2VecModel