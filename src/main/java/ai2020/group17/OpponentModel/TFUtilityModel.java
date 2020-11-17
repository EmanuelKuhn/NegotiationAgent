package ai2020.group17.OpponentModel;

import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.FloatDataBuffer;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Assign;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.train.ApplyGradientDescent;
import org.tensorflow.types.TFloat32;

import java.util.Arrays;
import java.util.List;
import java.util.stream.LongStream;

import static org.tensorflow.op.core.Placeholder.shape;

/**
 * Class that setups and manages the tf model.
 */
public class TFUtilityModel {

    /**
     * Class that represents training examples of int[] bids
     */
    public static class TrainingExample {
        // The input bid
        int[] options;

        // Expected output
        boolean accepted;

        // Used for testing
        float actualValue;

        public TrainingExample(int[] options, boolean accepted) {
            this.options = options;
            this.accepted = accepted;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            TrainingExample example = (TrainingExample) o;
            return accepted == example.accepted &&
                    Arrays.equals(options, example.options);
        }

		@Override
		public String toString() {
			return "TrainingExample [options=" + Arrays.toString(options) + ", accepted=" + accepted + "]";
		}
    }

    // The tensorflow Graph and Ops objects
    Graph graph;
    Ops tf;

    // Object for setting up the tf graph
    TFUtility tfUtility;

    // Tensorflow placeholders
    // The other placeholders are contained in tfUtility
    List<ApplyGradientDescent<TFloat32>> gradientDescents;
    Placeholder<TFloat32> actuallyAccepted;
    Operand<TFloat32> predicted;

    // The tf session to actually operate on the graph
    private final Session session;

    /**
     * The class that manages the tf model states, and allows for training and inference of weights.
     *
     * @param issuesOptions The shape of the domain, e.g. [3, 4] means that there are 2 issues,
     *                      and the first issue has 3 possible options and the second issue has 4 possible options.
     */
    public TFUtilityModel(int[] issuesOptions) {
        graph = new Graph();

        tf = Ops.create(graph);

        // This creates all the necessary tensorflow Placeholder's and Variable's
        // They can be referenced by e.g. utility.issues[0].issueOneHotVectorPlaceholder for the first issues onehotvector
        tfUtility = new TFUtility(tf, issuesOptions);


        List<Assign<TFloat32>> assigns = tfUtility.initIssueWeights(tf, 0.5f);
        assigns.add(tfUtility.initWeights(tf, 1.0f));

        // Store reference to the prediction operation
        predicted = tfUtility.predictUtility(tf);

        // Actually accepted placeholder
        actuallyAccepted = tf.placeholder(TFloat32.DTYPE, shape(Shape.scalar()));

        // A different loss might work better, but this seemed to work a lot better than MSE.
        Operand<TFloat32> loss = tf.nn.elu(tf.math.square(tf.math.sub(actuallyAccepted, predicted)));
        tf.ensureShape(loss, Shape.scalar());

        // Learning rate
        Constant<TFloat32> alpha = tf.constant(0.01f);

        // Keep reference to gradient descents
        gradientDescents = tfUtility.applyGradientDescent(tf, loss, alpha);

        // Initialize session
        session = new Session(graph);

        // Initialize graph variables
        Session.Runner runner = session.runner();
        for (Assign<TFloat32> assign: assigns) {
            runner.addTarget(assign);
        }
        runner.run();
    }

    // Each int[] represents the choice for an option per issue, e.g. [4, 2] represents choice of option 4 on issue 0 and option 2 on issue 1
    public void train(List<TrainingExample> trainingExamples) {
        // Repeat training loop 10 times
        for (int i = 0; i < 10; i++) {

            // Train on each trainingexample
            for (TrainingExample example : trainingExamples) {
                float actual = example.accepted ? 1.0f : 0.0f;

                Session.Runner runner = session.runner();

                // Add gradient descent targets
                for (ApplyGradientDescent<TFloat32> gradDescent : this.gradientDescents) {
                    runner.addTarget(gradDescent);
                }

                // Feed in training data
                tfUtility.feed(runner, example.options);
                runner.feed(this.actuallyAccepted, TFloat32.scalarOf(actual));

                // Run with training targets
                runner.run();
            }
        }
    }

    /**
     * Compute the prediction for a bid.
     * @param options input bid
     * @return predicted utility
     */
    public float predict(int[] options) {
        Session.Runner runner = session.runner();

        tfUtility.feed(runner, options);

        float predictedAcceptedValue = runner.fetch(this.predicted)
                .run().get(0).rawData().asFloats().getFloat(0);

        return predictedAcceptedValue;
    }

    /**
     * Print model weights.
     *
     * Useful for debugging.
     */
    public void printWeights() {

        for(TFIssue issue: tfUtility.issues) {
            Tensor<?> computedIssueWeights = session.runner().fetch(issue.issueWeights).run().get(0);

            System.out.println("Weight issue" + issue.issueIdx + " is " + computedIssueWeights);

            FloatDataBuffer floats = computedIssueWeights.rawData().asFloats();

            for (int i = 0; i < floats.size(); i++) {
                System.out.println("Weight issue" + issue.issueIdx + " data["+ i +"] " + floats.getFloat(i));
            }
        }

        Tensor<?> computedWeights = session.runner().fetch(tfUtility.weights).run().get(0);
        System.out.println("Weight is " + computedWeights);

        FloatDataBuffer floats = computedWeights.rawData().asFloats();

        for (int i = 0; i < floats.size(); i++) {
            System.out.println("Weight data["+ i +"] " + floats.getFloat(i));
        }

    }

    // Equivalent to the LinearAdditive UtilitySpace getWeights()
    public double[] computeWeights() {
        Operand<TFloat32> normalizedWeights = TFUtility.normalizedWeights(this.tf, this.tfUtility.weights);

        Tensor<?> computedWeights = session.runner().fetch(normalizedWeights).run().get(0);

        FloatDataBuffer floats = computedWeights.rawData().asFloats();

        return LongStream.range(0, floats.size()).mapToDouble(floats::getFloat).toArray();
    }

    // Equivalent to the LinearAdditive UtilitySpace getUtilities()
    public double[] computeIssueWeights(int issueIndex) {
        Operand<TFloat32> weightsClipped = TFIssue.weightsClipped(tf, tfUtility.issues[issueIndex].issueWeights);

        Tensor<?> computedWeights = session.runner().fetch(weightsClipped).run().get(0);

        FloatDataBuffer floats = computedWeights.rawData().asFloats();

        return LongStream.range(0, floats.size()).mapToDouble(floats::getFloat).toArray();
    }
}
