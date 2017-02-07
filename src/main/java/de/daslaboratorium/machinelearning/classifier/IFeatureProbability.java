package de.daslaboratorium.machinelearning.classifier;

/**
 * Simple interface defining the method to calculate the feature probability.
 *
 * @author Philipp Nolte
 *
 * @param <T>
 *            The feature class.
 * @param <K>
 *            The category class.
 */
public interface IFeatureProbability<T, K> {

    /**
     * Returns the probability of a <code>feature</code> being classified as
     * <code>category</code> in the learning set.
     * 
     * @param feature
     *            the feature to return the probability for
     * @param category
     *            the category to check the feature against
     * @return the probability <code>p(feature|category)</code>
     */
    public float featureProbability(T feature, K category);

}
