package de.daslaboratorium.machinelearning.classifier;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Collection;

public class ClassifierTester {

    /**
     * @param args
     * @throws IOException 
     */
    public static void main(String[] args) throws IOException {
        BayesClassifier<String, String> classifier =
                new BayesClassifier<String, String>();

        BufferedReader io =new BufferedReader(new InputStreamReader(System.in));
        String line = "";
        System.out.print("> ");
        while ((line = io.readLine()) != null) {
            String[] tokens = line.split("\\s");
            if (tokens.length < 3) {
                System.out.println("not enough params");
                continue;
            }
            if (tokens[0].startsWith("t")) {
                Collection<String> context =
                        Arrays.asList(
                                Arrays.copyOfRange(tokens, 2, tokens.length));
                classifier.learn(tokens[1], context);
            } else if (tokens[0].startsWith("c")) {
                Collection<String> context =
                        Arrays.asList(
                                Arrays.copyOfRange(tokens, 1, tokens.length));
                classifier.classify(context);
            }
            System.out.print("> ");
        }
    }

}
