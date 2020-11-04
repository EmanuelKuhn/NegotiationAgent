import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class UtilityModelTest {

    @Test
    // This is obviously a probabilistic test, not sure how to auto run more than once though
    void trainAccuracyGrt80() {

        // Setup a model with known preferences to sample from
        KnownUtilityValueModel hiddenModel = new KnownUtilityValueModel(2, 10);

        // The tensorflow model that will be trained
        UtilityModel model = new UtilityModel(hiddenModel.issuesOptions);

        // Train the model
        model.train(hiddenModel.generateExamples(100));

        List<UtilityModel.TrainingExample> testSet = hiddenModel.generateExamples(1000);

        float correct = 0;
        float wrong = 0;

        // Compute accuracy of the trained model
        for(UtilityModel.TrainingExample example: testSet) {
            float prediction = model.predict(example.options);

            if (Math.abs(prediction - (example.accepted ? 1.0f : 0.0f)) < 0.5) {
                correct += 1;
            } else {
                wrong += 1;
            }
        }

        model.printWeights();

        System.out.println("Accuracy: " + correct / (correct + wrong));

        assertTrue(correct / (correct + wrong) >= 0.8);
    }

    @Test
    // This is obviously a probabilistic test, not sure how to auto run more than once though
    void utilityPredictionClose() {

        // The known model to generate training examples with
        KnownUtilityValueModel hiddenModel = new KnownUtilityValueModel(5, 10);

        // The tensorflow model that should be trained
        UtilityModel model = new UtilityModel(hiddenModel.issuesOptions);

        model.train(hiddenModel.generateExamples(100));

        model.printWeights();

        List<UtilityModel.TrainingExample> testSet = hiddenModel.generateExamples(1000);

        List<Float> differences = new ArrayList<>();

        for(UtilityModel.TrainingExample example: testSet) {
            float prediction = model.predict(example.options);

            differences.add(Math.abs(prediction - example.actualValue));
        }

        float avg = differences.stream().reduce(0.0f, Float::sum) / differences.size();

        System.out.println("Average difference: " + avg);

        assertTrue(avg <= 0.2);
    }

    // A class to generate training samples with known preference paramaters
    private class KnownUtilityValueModel {

        List<float[]> issues;
        float[] weights;

        float threshold;

        int[] issuesOptions;

        public KnownUtilityValueModel(int minIssues, int maxIssues) {
            Random r = new Random();

            do {
                threshold = r.nextFloat();
            } while (threshold < 0.3 || threshold > 0.7);

            int numberOfIssues = r.nextInt(maxIssues - minIssues) + minIssues;

            issues = new ArrayList<>();

            for (int i = 0; i < numberOfIssues; i++) {
                float[] issueValues = new float[r.nextInt(3) + 2];

                for (int j = 0; j < issueValues.length; j++) {
                    issueValues[j] = r.nextFloat();
                }

                issues.add(issueValues);
            }

            weights = new float[numberOfIssues];

            float sum = 0f;
            for (int i = 0; i < weights.length; i++) {
                weights[i] = r.nextFloat();
                sum += weights[i];
            }

            for (int i = 0; i < weights.length; i++) {
                weights[i] = weights[i] / sum;
            }

            issuesOptions = issues.stream().mapToInt(x->x.length).toArray();
        }

        public float calcValue(int[] options) {

            float value = 0;

            for (int i = 0; i < issues.size(); i++) {
                value += issues.get(i)[options[i]] * weights[i];
            }

            return value;
        }

        public UtilityModel.TrainingExample randomExample() {
            Random r = new Random();

            int[] options = new int[issues.size()];

            for (int i = 0; i < issues.size(); i++) {
                options[i] = r.nextInt(issues.get(i).length);
            }

            UtilityModel.TrainingExample example = new UtilityModel.TrainingExample(options, calcValue(options) >= threshold);

            example.actualValue = calcValue(options);

            return example;
        }


        public List<UtilityModel.TrainingExample> generateExamples(int n) {
            List<UtilityModel.TrainingExample> list = new ArrayList<>();

            for (int i = 0; i < n; i++) {
                list.add(randomExample());
            }

            return list;
        }
    }


}