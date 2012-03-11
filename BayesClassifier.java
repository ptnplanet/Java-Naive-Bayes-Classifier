package de.daslaboratorium.machinelearning.classifier;

import java.util.AbstractMap.SimpleEntry;
import java.util.Collection;
import java.util.Comparator;
import java.util.Map.Entry;
import java.util.SortedSet;
import java.util.TreeSet;

/**
 * A concrete implementation of the abstract Classifier class.  The Bayes
 * classifier implements a naive Bayes approach to classifying a given set of
 * features: classify(feat1,...,featN) = argmax(P(cat)*PROD(P(featI|cat)
 *
 * @author Philipp Nolte
 *
 * @see http://en.wikipedia.org/wiki/Naive_Bayes_classifier
 *
 * @param <T> The feature class.
 * @param <K> The category class.
 */
public class BayesClassifier<T, K> extends Classifier<T, K> {

    /**
     * Calculates the product of all feature probabilities: PROD(P(featI|cat)
     *
     * @param features The set of features to use.
     * @param category The category to test for.
     * @return The product of all feature probabilities.
     */
    private float featuresProbabilityProduct(Collection<T> features,
            K category) {
        float product = 1.0f;
        for (T feature : features)
            product *= this.featureWeighedAverage(feature, category);
        return product;
    }

    /**
     * Calculates the probability that the features can be classified as the
     * category given.
     *
     * @param features The set of features to use.
     * @param category The category to test for.
     * @return The probability that the features can be classified as the
     *    category.
     */
    private float categoryProbability(Collection<T> features, K category) {
        return ((float) this.categoryCount(category)
                    / (float) this.getCategoriesTotal())
                * featuresProbabilityProduct(features, category);
    }

    /**
     * Retrieves a sorted <code>Set</code> of probabilities that the given set
     * of features is classified as the available categories.
     *
     * @param features The set of features to use.
     * @return A sorted <code>Set</code> of category-probability-entries.
     */
    private SortedSet<Classification<T, K>> categoryProbabilities(
            Collection<T> features) {

        /*
         * Sort the set according to the possibilities. Because we have to sort
         * by the mapped value and not by the mapped key, we can not use a
         * sorted tree (TreeMap) and we have to use a set-entry approach to
         * achieve the desired functionality. A custom comparator is therefore
         * needed.
         */
        SortedSet<Classification<T, K>> probabilities =
                new TreeSet<Classification<T, K>>(
                        new Comparator<Classification<T, K>>() {

                    @Override
                    public int compare(Classification<T, K> o1,
                            Classification<T, K> o2) {
                        int toReturn = Float.compare(
                                o1.getProbability(), o2.getProbability());
                        if ((toReturn == 0)
                                && !o1.getCategory().equals(o2.getCategory()))
                            toReturn = -1;
                        return toReturn;
                    }
                });

        for (K category : this.getCategories())
            probabilities.add(new Classification<T, K>(
                    features, category,
                    this.categoryProbability(features, category)));
        return probabilities;
    }

    /**
     * Classifies the given set of features.
     *
     * @return The category the set of features is classified as.
     */
    @Override
    public K classify(Collection<T> features) {
        SortedSet<Classification<T, K>> probabilites =
                this.categoryProbabilities(features);

        System.out.println("Results:\t");
        for (Classification<T, K> prob : probabilites)
            System.out.println(prob);

        if (probabilites.size() > 0) {
            System.out.println("Classified as " +
                    probabilites.last().getCategory());
            return probabilites.last().getCategory();
        } else {
            System.out.println("No results");
        }
        return null;
    }

}
