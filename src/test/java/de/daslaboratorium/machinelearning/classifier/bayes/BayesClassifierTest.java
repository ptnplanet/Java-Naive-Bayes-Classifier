package de.daslaboratorium.machinelearning.classifier.bayes;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import de.daslaboratorium.machinelearning.classifier.Classification;
import de.daslaboratorium.machinelearning.classifier.Classifier;

public class BayesClassifierTest {

	private static final double EPSILON = 0.001;
	private static final String CATEGORY_NEGATIVE = "negative";
	private static final String CATEGORY_POSITIVE = "positive";
	private Classifier<String, String> bayes;
	
	@Before
	public void setUp() {
		/*
         * Create a new classifier instance. The context features are
         * Strings and the context will be classified with a String according
         * to the featureset of the context.
         */
		bayes = new BayesClassifier<String, String>();
		
		/*
         * The classifier can learn from classifications that are handed over
         * to the learn methods. Imagin a tokenized text as follows. The tokens
         * are the text's features. The category of the text will either be
         * positive or negative.
         */
        final String[] positiveText = "I love sunny days".split("\\s");
        bayes.learn(CATEGORY_POSITIVE, Arrays.asList(positiveText));

        final String[] negativeText = "I hate rain".split("\\s");
        bayes.learn(CATEGORY_NEGATIVE, Arrays.asList(negativeText));
	}
	
	@Test
	public void testStringClassification() {
		final String[] unknownText1 = "today is a sunny day".split("\\s");
        final String[] unknownText2 = "there will be rain".split("\\s");

        Assert.assertEquals(CATEGORY_POSITIVE, bayes.classify(Arrays.asList(unknownText1)).getCategory());
        Assert.assertEquals(CATEGORY_NEGATIVE, bayes.classify(Arrays.asList(unknownText2)).getCategory());
	}
	
	@Test
	public void testStringClassificationInDetails() {
		
		final String[] unknownText1 = "today is a sunny day".split("\\s");
		
		Collection<Classification<String, String>> classifications = ((BayesClassifier<String, String>) bayes).classifyDetailed(
                Arrays.asList(unknownText1));
		
		List<Classification<String, String>> list = new ArrayList<Classification<String,String>>(classifications);
		
		Assert.assertEquals(CATEGORY_NEGATIVE, list.get(0).getCategory());
		Assert.assertEquals(0.0078125, list.get(0).getProbability(), EPSILON);
		
		Assert.assertEquals(CATEGORY_POSITIVE, list.get(1).getCategory());
		Assert.assertEquals(0.0234375, list.get(1).getProbability(), EPSILON);
	}

}
