Java Naive Bayes Classifier
==================

[![Build Status](https://travis-ci.org/ptnplanet/Java-Naive-Bayes-Classifier.svg?branch=master)](https://travis-ci.org/ptnplanet/Java-Naive-Bayes-Classifier)
[![](https://jitpack.io/v/ptnplanet/Java-Naive-Bayes-Classifier.svg)](https://jitpack.io/#ptnplanet/Java-Naive-Bayes-Classifier)

Nothing special. It works and is well documented, so you should get it running without wasting too much time searching for other alternatives on the net.

Maven Quick-Start
------------------

This Java Naive Bayes Classifier can be installed via the jitpack repository. Make sure to add it to your buildfile first.

```xml
<repositories>
  <resort solve
    <id>jitpack.io</id>
    <url>https://jitpack.io</url>
  </repository>
</repositories>
```

Then, treat it as any other dependency.

```xml
<dependency>
  <groupId>com.github.ptnplanet</groupId>
  <artifactId>Java-Naive-Bayes-Classifier</artifactId>
  <version>1.0.7</version>
</dependency>
```

For other build-tools (e.g. gradle), visit https://jitpack.io for configuration snippets.

Please also head to the release tab for further releases.

Overview
------------------

I like talking about *features* and *categories*. Objects have features and may belong to a category. The classifier will try matching objects to their categories by looking at the objects' features. It does so by consulting its memory filled with knowledge gathered from training examples.

Classifying a feature-set results in the highest product of 1) the probability of that category to occur and 2) the product of all the features' probabilities to occure in that category:

```classify(feature1, ..., featureN) = argmax(P(category) * PROD(P(feature|category)))```

This is a so-called maximum a posteriori estimation. Wikipedia actually does a good job explaining it: http://en.wikipedia.org/wiki/Naive_Bayes_classifier#Probabilistic_model

Learning from Examples
------------------

Add knowledge by telling the classifier, that these features belong to a specific category:

```java
String[] positiveText = "I love sunny days".split("\\s");
bayes.learn("positive", Arrays.asList(positiveText));
```

Classify unknown objects
------------------

Use the gathered knowledge to classify unknown objects with their features. The classifier will return the category that the object most likely belongs to.

```java
String[] unknownText1 = "today is a sunny day".split("\\s");
bayes.classify(Arrays.asList(unknownText1)).getCategory();
```

Example
------------------

Here is an excerpt from the example. The classifier will classify sentences (arrays of features) as sentences with either positive or negative sentiment. Please refer to the full example for a more detailed documentation.

```java
// Create a new bayes classifier with string categories and string features.
Classifier<String, String> bayes = new BayesClassifier<String, String>();

// Two examples to learn from.
String[] positiveText = "I love sunny days".split("\\s");
String[] negativeText = "I hate rain".split("\\s");

// Learn by classifying examples.
// New categories can be added on the fly, when they are first used.
// A classification consists of a category and a list of features
// that resulted in the classification in that category.
bayes.learn("positive", Arrays.asList(positiveText));
bayes.learn("negative", Arrays.asList(negativeText));

// Here are two unknown sentences to classify.
String[] unknownText1 = "today is a sunny day".split("\\s");
String[] unknownText2 = "there will be rain".split("\\s");

System.out.println( // will output "positive"
    bayes.classify(Arrays.asList(unknownText1)).getCategory());
System.out.println( // will output "negative"
    bayes.classify(Arrays.asList(unknownText2)).getCategory());

// Get more detailed classification result.
((BayesClassifier<String, String>) bayes).classifyDetailed(
    Arrays.asList(unknownText1));

// Change the memory capacity. New learned classifications (using
// the learn method) are stored in a queue with the size given
// here and used to classify unknown sentences.
bayes.setMemoryCapacity(500);
```

Forgetful learning
------------------

This classifier is forgetful. This means, that the classifier will forget recent classifications it uses for future classifications after - defaulting to 1.000 - classifications learned. This will ensure, that the classifier can react to ongoing changes in the user's habbits.


Interface
------------------
The abstract ```Classifier<T, K>``` serves as a base for the concrete ```BayesClassifier<T, K>```. Here are its methods. Please also refer to the Javadoc.

* ```void reset()``` Resets the learned feature and category counts.
* ```Set<T> getFeatures()``` Returns a ```Set``` of features the classifier knows about.
* ```Set<K> getCategories()``` Returns a ```Set``` of categories the classifier knows about.
* ```int getCategoriesTotal()``` Retrieves the total number of categories the classifier knows about.
* ```int getMemoryCapacity()``` Retrieves the memory's capacity.
* ```void setMemoryCapacity(int memoryCapacity)``` Sets the memory's capacity.  If the new value is less than the old value, the memory will be truncated accordingly.
* ```void incrementFeature(T feature, K category)``` Increments the count of a given feature in the given category.  This is equal to telling the classifier, that this feature has occurred in this category.
* ```void incrementCategory(K category)``` Increments the count of a given category.  This is equal to telling the classifier, that this category has occurred once more.
* ```void decrementFeature(T feature, K category)``` Decrements the count of a given feature in the given category.  This is equal to telling the classifier that this feature was classified once in the category.
* ```void decrementCategory(K category)``` Decrements the count of a given category.  This is equal to telling the classifier, that this category has occurred once less.
* ```int getFeatureCount(T feature, K category)``` Retrieves the number of occurrences of the given feature in the given category.
* ```int getFeatureCount(T feature)``` Retrieves the total number of occurrences of the given feature.
* ```int getCategoryCount(K category)``` Retrieves the number of occurrences of the given category.
* ```float featureProbability(T feature, K category)``` (*implements* ```IFeatureProbability<T, K>.featureProbability```) Returns the probability that the given feature occurs in the given category.
* ```float featureWeighedAverage(T feature, K category)``` Retrieves the weighed average ```P(feature|category)``` with overall weight of ```1.0``` and an assumed probability of ```0.5```. The probability defaults to the overall feature probability.
* ```float featureWeighedAverage(T feature, K category, IFeatureProbability<T, K> calculator)``` Retrieves the weighed average ```P(feature|category)``` with overall weight of ```1.0```, an assumed probability of ```0.5``` and the given object to use for probability calculation.
* ```float featureWeighedAverage(T feature, K category, IFeatureProbability<T, K> calculator, float weight)```Retrieves the weighed average ```P(feature|category)``` with the given weight and an assumed probability of ```0.5``` and the given object to use for probability calculation.
* ```float featureWeighedAverage(T feature, K category, IFeatureProbability<T, K> calculator, float weight,  float assumedProbability)``` Retrieves the weighed average ```P(feature|category)``` with the given weight, the given assumed probability and the given object to use for probability calculation.
* ```void learn(K category, Collection<T> features)``` Train the classifier by telling it that the given features resulted in the given category.
* ```void learn(Classification<T, K> classification)``` Train the classifier by telling it that the given features resulted in the given category.

The ```BayesClassifier<T, K>``` class implements the following abstract method:

* ```Classification<T, K> classify(Collection<T> features)``` It will retrieve the most likely category for the features given and depends on the concrete classifier implementation.

Running the example
------------------

```shell
$ git clone https://github.com/ptnplanet/Java-Naive-Bayes-Classifier.git
$ cd Java-Naive-Bayes-Classifier
$ javac -cp src/main/java example/RunnableExample.java
$ java -cp example:src/main/java RunnableExample
```

Possible Performance issues
------------------

Performance improvements, I am currently thinking of:

- Store the natural logarithms of the feature probabilities and add them together instead of multiplying the probability numbers

The MIT License (MIT)
------------------

Copyright (c) 2012-2017 Philipp Nolte

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
