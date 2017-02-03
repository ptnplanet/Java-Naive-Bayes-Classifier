package de.daslaboratorium.machinelearning.classifier.bayes;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import de.daslaboratorium.machinelearning.classifier.Classifier;

public class BayesClassifierCategoryNumbersTest {

    private static final int CATEGORY_COUNT = 100;
    private static final int TEST_CYCLES = 100;
    private static final int FEATURESET_LENGTH = 10;
    private static final int FEATURE_LENGTH = 3;
    private static final Random RANDOM = new Random();
    private Integer[] categories = new Integer[CATEGORY_COUNT];
    private Classifier<String, Integer> bayes;

    @Before
    public void setUp() {

        /*
         * Create a new classifier instance. The context features are String and
         * the context will be classified with a Integer according to the
         * featureset of the context.
         */
        bayes = new BayesClassifier<String, Integer>();

        // Create categories
        for (Integer i = 0; i < CATEGORY_COUNT; i++) {
            categories[i] = i;

            // The feature we want to associate with the category is the int.
            ArrayList<String> featureList = new ArrayList<String>(1);
            featureList.add(Integer.toString(i));

            // Now learn the translation from int to String.
            bayes.learn(categories[i], featureList);
        }
    }

    @Test
    public void testStringClassification() {

        int currentCategorytoTest = 0;

        /*
         * Create random Strings with a number (0<i<CATEGORY_COUNT) and classify
         * them. The expected result is, that the classifier has learned to
         * classify the string according to the number existing in it.
         */
        for (Integer i = 0; i < TEST_CYCLES; i++) {

            String initString = Integer.toString(categories[currentCategorytoTest]);
            StringBuilder sb = new StringBuilder(initString);

            for (int j = 0; j < FEATURESET_LENGTH; j++) {

                sb.append(" ");

                for (int k = 0; k < FEATURE_LENGTH; k++) {
                    char c = (char) (RANDOM.nextInt(128 - 64) + 64);
                    sb.append(c);
                }
            }

            String randomString = sb.toString();

            Integer category = bayes.classify(Arrays.asList(randomString.split(" "))).getCategory();
            Assert.assertEquals(categories[currentCategorytoTest], category);

            currentCategorytoTest = (currentCategorytoTest + 1) % CATEGORY_COUNT;
        }
    }
}
