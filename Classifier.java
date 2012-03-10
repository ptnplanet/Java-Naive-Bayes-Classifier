package de.daslaboratorium.machinelearning.classifier;

import java.util.Collection;
import java.util.Dictionary;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Set;

/**
 * Abstract base extended by any concrete classifier.  It implements the basic
 * functionality for storing categories or features and can be used to calculate
 * basic probabilities – both category and feature probabilities. The classify
 * function has to be implemented by the concrete classifier class.
 *
 * @author Philipp Nolte
 *
 * @param <T> A feature class
 * @param <K> A category class
 */
public abstract class Classifier<T, K>
        implements BasicProbabilityCalculator<T, K>{

    /**
     * A dictionary mapping features to their counts in the categories.
     */
    private Dictionary<T, Dictionary<K, Integer>> featureCounts;

    /**
     * A dictionary mapping categories to their number of occurrences.
     */
    private Dictionary<K, Integer> categoryCounts;

    /**
     * Constructs a new classifier without any trained knowledge.
     */
    public Classifier() {
        this.reset();
    }

    /**
     * Resets the <i>learned</i> feature and category counts.
     */
    public void reset() {
        this.featureCounts = new Hashtable<T, Dictionary<K, Integer>>();
        this.categoryCounts = new Hashtable<K, Integer>();
    }

    /**
     * Returns a <code>Set</code> of features the classifier knows about.
     *
     * @return The <code>Set</code> of features the classifier knows about.
     */
    public Set<T> getFeatures() {
        return ((Hashtable<T, Dictionary<K, Integer>>) this.featureCounts)
                .keySet();
    }

    /**
     * Returns a <code>Set</code> of categories the classifier knows about.
     *
     * @return The <code>Set</code> of categories the classifier knows about.
     */
    public Set<K> getCategories() {
        return ((Hashtable<K, Integer>) this.categoryCounts).keySet();
    }

    /**
     * Retrieves the total number of categories the classifier knows about.
     *
     * @return The total category count.
     */
    public int getTotalCategoryCount() {
        int total = 0;
        for(Enumeration<Integer> e = this.categoryCounts.elements();
                e.hasMoreElements();)
            total += e.nextElement();
        return total;
    }

    /**
     * Increments the count of a given feature in the given category.  This is
     * equal to telling the classifier, that this feature has occurred in this
     * category.
     *
     * @param feature The feature, which count to increase.
     * @param category The category the feature occurred in.
     */
    public void incrementWord(T feature, K category) {
        Dictionary<K, Integer> categories = this.featureCounts.get(feature);
        if (categories == null) {
            this.featureCounts.put(feature, new Hashtable<K, Integer>());
            categories = this.featureCounts.get(feature);
        }
        Integer count = categories.get(category);
        if (count == null) {
            categories.put(category, 1);
            count = categories.get(category);
        }
        count++;
    }

    /**
     * Increments the count of a given category.  This is equal to telling the
     * classifier, that this category has occurred once more.
     *
     * @param category The category, which count to increase.
     */
    public void incrementCategory(K category) {
        Integer count = this.categoryCounts.get(category);
        if (count == null) {
            this.categoryCounts.put(category, 1);
            count = this.categoryCounts.get(category);
        }
        count++;
    }

    /**
     * Retrieves the number of occurrences of the given feature in the given
     * category.
     *
     * @param feature The feature, which count to retrieve.
     * @param category The category, which the feature occurred in.
     * @return The number of occurrences of the feature in the category.
     */
    public int featureCount(T feature, K category) {
        Dictionary<K, Integer> categories = this.featureCounts.get(feature);
        if (categories == null)
            return 0;
        Integer count = categories.get(category);
        return (count == null) ? 0 : count.intValue();
    }

    /**
     * Retrieves the number of occurrences of the given category.
     * 
     * @param category The category, which count should be retrieved.
     * @return The number of occurrences.
     */
    public int categoryCount(K category) {
        Integer count = this.categoryCounts.get(category);
        return (count == null) ? 0 : count.intValue();
    }

    /**
     * The probability that the feature given occurred in the the category
     * given.
     *
     * @return The probability.
     */
    public float featureProbability(T feature, K category) {
        if (categoryCount(category) == 0)
            return 0.0f;
        return (float) featureCount(feature, category)
                / (float) categoryCount(category);
    }

    /**
     * Retrieves the weighed average <code>P(feature|category)</code> with
     * overall weight of <code>1.0</code> and an assumed probability of
     * <code>0.5</code>. The probability defaults to the overall feature
     * probability.
     *
     * @see de.daslaboratorium.machinelearning.classifier.Classifier#featureProbability(Object, Object)
     * @see de.daslaboratorium.machinelearning.classifier.Classifier#featureWeighedAverage(Object, Object, BasicProbabilityCalculator, float, float)
     *
     * @param feature The feature, which probability to calculate.
     * @param category The category.
     * @return The weighed average probability.
     */
    public float featureWeighedAverage(T feature, K category) {
        return this.featureWeighedAverage(feature, category,
                null, 1.0f, 0.5f);
    }

    /**
     * Retrieves the weighed average <code>P(feature|category)</code> with
     * overall weight of <code>1.0</code>, an assumed probability of
     * <code>0.5</code> and the given object to use for probability calculation.
     *
     * @see de.daslaboratorium.machinelearning.classifier.Classifier#featureWeighedAverage(Object, Object, BasicProbabilityCalculator, float, float)
     *
     * @param feature The feature, which probability to calculate.
     * @param category The category.
     * @param calculator The calculating object.
     * @return The weighed average probability.
     */
    public float featureWeighedAverage(T feature, K category,
            BasicProbabilityCalculator<T, K> calculator) {
        return this.featureWeighedAverage(feature, category,
                calculator, 1.0f, 0.5f);
    }

    /**
     * Retrieves the weighed average <code>P(feature|category)</code> with
     * the given weight and an assumed probability of <code>0.5</code> and the
     * given object to use for probability calculation.
     *
     * @see de.daslaboratorium.machinelearning.classifier.Classifier#featureWeighedAverage(Object, Object, BasicProbabilityCalculator, float, float)
     *
     * @param feature The feature, which probability to calculate.
     * @param category The category.
     * @param calculator The calculating object.
     * @param weight The feature weight.
     * @return The weighed average probability.
     */
    public float featureWeighedAverage(T feature, K category,
            BasicProbabilityCalculator<T, K> calculator, float weight) {
        return this.featureWeighedAverage(feature, category,
                calculator, weight, 0.5f);
    }

    /**
     * Retrieves the weighed average <code>P(feature|category)</code> with
     * the given weight, the given assumed probability and the given object to
     * use for probability calculation.
     *
     * @param feature The feature, which probability to calculate.
     * @param category The category.
     * @param calculator The calculating object.
     * @param weight The feature weight.
     * @param assumedProbability The assumed probability.
     * @return The weighed average probability.
     */
    public float featureWeighedAverage(T feature, K category,
            BasicProbabilityCalculator<T, K> calculator, float weight,
            float assumedProbability) {

        /*
         * use the given calculating object or the default method to calculate
         * the probability that the given feature occurred in the given
         * category.
         */
        final float basicProbability =
                (calculator == null)
                    ? this.featureProbability(feature, category)
                            : calculator.featureProbability(feature, category);

        // Calculate the total occurrences of the given feature.
        int totals = 0;
        for (K cat : this.getCategories())
            totals += this.featureCount(feature, cat);

        return (weight * assumedProbability + totals  * basicProbability)
                / (weight + totals);
    }

    /**
     * Train the classifier by telling it that the given features resulted in
     * the given category.
     *
     * @param category The category the features belong to.
     * @param features The features that resulted in the given category.
     */
    public void train(K category, Collection<T> features) {
        for (T feature : features)
            this.incrementWord(feature, category);
        this.incrementCategory(category);
    }

    /**
     * The classify method.  It will retrieve the most likely category for the
     * features given and depends on the concrete classifier implementation.
     *
     * @param features The features to classify.
     * @return The category most likely.
     */
    public abstract K classify(Collection<T> features);

}
