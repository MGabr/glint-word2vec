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

"""
Python package for feature in MLlib.
"""
from __future__ import absolute_import

import sys

if sys.version >= '3':
    basestring = str
    unicode = str

from py4j.protocol import Py4JJavaError

from pyspark import since
from pyspark.rdd import RDD, ignore_unicode_prefix
from pyspark.mllib.common import callMLlibFunc
from pyspark.mllib import feature
from pyspark.mllib.feature import JavaVectorTransformer
from pyspark.mllib.linalg import _convert_to_vector
from pyspark.mllib.util import JavaLoader, JavaSaveable

__all__ = ['NaiveGlintWord2Vec', 'NaiveGlintWord2VecModel']



class NaiveGlintWord2VecModel(JavaVectorTransformer, JavaSaveable, JavaLoader):
    """
    class for Word2Vec model
    .. versionadded:: 1.2.0
    """

    __module__ = "pyspark.mllib.feature"

    @since('1.2.0')
    def transform(self, word):
        """
        Transforms a word to its vector representation
        .. note:: Local use only
        :param word: a word
        :return: vector representation of word(s)
        """
        try:
            return self.call("transform", word)
        except Py4JJavaError:
            raise ValueError("%s not found" % word)

    @since('1.2.0')
    def findSynonyms(self, word, num):
        """
        Find synonyms of a word
        :param word: a word or a vector representation of word
        :param num: number of synonyms to find
        :return: array of (word, cosineSimilarity)
        .. note:: Local use only
        """
        if not isinstance(word, basestring):
            word = _convert_to_vector(word)
        words, similarity = self.call("findSynonyms", word, num)
        return zip(words, similarity)

    @since('1.4.0')
    def getVectors(self):
        """
        Returns a map of words to their vector representations.
        """
        return self.call("getVectors")

    @classmethod
    @since('1.5.0')
    def load(cls, sc, path):
        """
        Load a model from the given path.
        """
        jmodel = sc._jvm.org.apache.spark.mllib.feature \
            .NaiveGlintWord2VecModel.load(sc._jsc.sc(), path)
        model = sc._jvm.org.apache.spark.mllib.api.python.NaiveGlintWord2VecModelWrapper(jmodel)
        return NaiveGlintWord2VecModel(model)


feature.NaiveGlintWord2VecModel = NaiveGlintWord2VecModel


@ignore_unicode_prefix
class NaiveGlintWord2Vec(object):
    """Word2Vec creates vector representation of words in a text corpus.
    The algorithm first constructs a vocabulary from the corpus
    and then learns vector representation of words in the vocabulary.
    The vector representation can be used as features in
    natural language processing and machine learning algorithms.
    We used skip-gram model in our implementation and hierarchical
    softmax method to train the model. The variable names in the
    implementation matches the original C implementation.
    For original C implementation,
    see https://code.google.com/p/word2vec/
    For research papers, see
    Efficient Estimation of Word Representations in Vector Space
    and Distributed Representations of Words and Phrases and their
    Compositionality.
    >>> sentence = "a b " * 100 + "a c " * 10
    >>> localDoc = [sentence, sentence]
    >>> doc = sc.parallelize(localDoc).map(lambda line: line.split(" "))
    >>> model = Word2Vec().setVectorSize(10).setSeed(42).fit(doc)
    Querying for synonyms of a word will not return that word:
    >>> syms = model.findSynonyms("a", 2)
    >>> [s[0] for s in syms]
    [u'b', u'c']
    But querying for synonyms of a vector may return the word whose
    representation is that vector:
    >>> vec = model.transform("a")
    >>> syms = model.findSynonyms(vec, 2)
    >>> [s[0] for s in syms]
    [u'a', u'b']
    >>> import os, tempfile
    >>> path = tempfile.mkdtemp()
    >>> model.save(sc, path)
    >>> sameModel = Word2VecModel.load(sc, path)
    >>> model.transform("a") == sameModel.transform("a")
    True
    >>> syms = sameModel.findSynonyms("a", 2)
    >>> [s[0] for s in syms]
    [u'b', u'c']
    >>> from shutil import rmtree
    >>> try:
    ...     rmtree(path)
    ... except OSError:
    ...     pass
    .. versionadded:: 1.2.0
    """

    __module__ = "pyspark.mllib.feature"

    def __init__(self):
        """
        Construct Word2Vec instance
        """
        self.vectorSize = 100
        self.learningRate = 0.025
        self.numPartitions = 1
        self.numIterations = 1
        self.seed = None
        self.minCount = 5
        self.windowSize = 5

    @since('1.2.0')
    def setVectorSize(self, vectorSize):
        """
        Sets vector size (default: 100).
        """
        self.vectorSize = vectorSize
        return self

    @since('1.2.0')
    def setLearningRate(self, learningRate):
        """
        Sets initial learning rate (default: 0.025).
        """
        self.learningRate = learningRate
        return self

    @since('1.2.0')
    def setNumPartitions(self, numPartitions):
        """
        Sets number of partitions (default: 1). Use a small number for
        accuracy.
        """
        self.numPartitions = numPartitions
        return self

    @since('1.2.0')
    def setNumIterations(self, numIterations):
        """
        Sets number of iterations (default: 1), which should be smaller
        than or equal to number of partitions.
        """
        self.numIterations = numIterations
        return self

    @since('1.2.0')
    def setSeed(self, seed):
        """
        Sets random seed.
        """
        self.seed = seed
        return self

    @since('1.4.0')
    def setMinCount(self, minCount):
        """
        Sets minCount, the minimum number of times a token must appear
        to be included in the word2vec model's vocabulary (default: 5).
        """
        self.minCount = minCount
        return self

    @since('2.0.0')
    def setWindowSize(self, windowSize):
        """
        Sets window size (default: 5).
        """
        self.windowSize = windowSize
        return self

    @since('1.2.0')
    def fit(self, data):
        """
        Computes the vector representation of each word in vocabulary.
        :param data: training data. RDD of list of string
        :return: Word2VecModel instance
        """
        if not isinstance(data, RDD):
            raise TypeError("data should be an RDD of list of string")
        jmodel = callMLlibFunc("trainWord2VecModel", data, int(self.vectorSize),
                               float(self.learningRate), int(self.numPartitions),
                               int(self.numIterations), self.seed,
                               int(self.minCount), int(self.windowSize))
        return NaiveGlintWord2VecModel(jmodel)


feature.NaiveGlintWord2Vec = NaiveGlintWord2Vec
