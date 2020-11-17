package ai2020.group17;

import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.*;
import org.tensorflow.op.math.Mul;
import org.tensorflow.op.train.ApplyGradientDescent;
import org.tensorflow.types.TFloat32;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

/**
 * Class for setting up and keeping references to tf placeholders and variables involved in computing the utility.
 */
public class TFUtility {

    // The TFIssue objects for each issue.
    TFIssue[] issues;

    // The relative weights between the issues.
    Variable<TFloat32> weights;

    /**
     * Create TFUtility with given bid shape.
     * @param tf The tf Ops.
     * @param issuesOptions The shape of the domain.
     */
    public TFUtility(Ops tf, int[] issuesOptions) {

        issues = new TFIssue[issuesOptions.length];

        for(int i = 0; i < issues.length; i++) {
            issues[i] = new TFIssue(tf, issuesOptions[i], i);
        }

        weights = tf.variable(Shape.of(issues.length), TFloat32.DTYPE);
    }

    /**
     * Method for applying the tf assign operation to the weights.
     * @param tf The tf Ops
     * @param initializationValue The value to initialize the weights to.
     * @return tf Assign operation.
     */
    public Assign<TFloat32> initWeights(Ops tf, float initializationValue) {
        return initWeights(tf, () -> Helper.repeat(initializationValue, issues.length));
    }

    /**
     * Method for applying the tf assign operation to the weights.
     * @param tf The tf Ops
     * @param initializer How to initialize the weights.
     * @return tf Assign operation.
     */
    public Assign<TFloat32> initWeights(Ops tf, Supplier<float[]> initializer) {
        return tf.assign(weights, tf.constant(initializer.get()));
    }

    /**
     * Initialize the issue utility weights for each issue.
     * @param tf The tf Ops.
     * @param initializationValue The value to initialize the utility weights to.
     * @return List of tf Assign operations.
     */
    public List<Assign<TFloat32>> initIssueWeights(Ops tf, float initializationValue) {

        List<Assign<TFloat32>> assigns = new ArrayList<>();

        for (TFIssue issue:issues) {
            assigns.add(issue.init(tf, initializationValue));
        }

        return assigns;
    }

    /**
     * Reference to the predict utility operation.
     * @param tf The tf Ops.
     * @return Operand to the predicted utility.
     */
    public Operand<TFloat32> predictUtility(Ops tf) {
        ArrayList<Operand<TFloat32>> issueUtilities = new ArrayList<>();

        for (int i = 0; i < this.issues.length; i++) {
            issueUtilities.add(this.issues[i].predictUtility(tf));
        }

        return utility(tf, issueUtilities, weights);
    }

    /**
     * Feed in a bid.
     * @param runner The tf runner.
     * @param options The bid to feed.
     */
    public void feed(Session.Runner runner, int[] options) {
        for (int i = 0; i < issues.length; i++) {
            issues[i].feed(runner, options[i]);
        }
    }

    /**
     * Method for gathering all weights that should be trained.
     * Including both the issue utility weights and the relative weights.
     * @return List of all weights.
     */
    private List<Variable<TFloat32>> getWeightVariables() {
        List<Variable<TFloat32>> variableList = new ArrayList<>();

        for (int i = 0; i < issues.length; i++) {
            variableList.add(issues[i].issueWeights);
        }

        variableList.add(weights);
        return variableList;
    }

    /**
     * Apply descent to all weights given a loss, and training rate.
     * @param tf The tf Ops.
     * @param loss Reference to the loss value.
     * @param alpha The training rate.
     * @return List of all ApplyGradientDescent operations.
     */
    public List<ApplyGradientDescent<TFloat32>> applyGradientDescent(Ops tf, Operand<TFloat32> loss, Operand<TFloat32> alpha) {

        List<Variable<TFloat32>> variableList = getWeightVariables();

        Gradients gradients = tf.gradients(loss, variableList);

        List<ApplyGradientDescent<TFloat32>> gradientDescentList = new ArrayList<>();

        for (int i = 0; i < variableList.size(); i++) {
            gradientDescentList.add(tf.train.applyGradientDescent(variableList.get(i), alpha, gradients.dy(i)));
        }

        return gradientDescentList;
    }

    /**
     * Compute normalized relative weights.
     * @param tf The tf Ops.
     * @param weights The weights to normalize.
     *
     * @return Normalized version of {@code weights}
     */
    public static Operand<TFloat32> normalizedWeights(Ops tf, Operand<TFloat32> weights) {
        ReduceSum<TFloat32> sumOfWeights = tf.reduceSum(weights, tf.constant(0));

        return tf.math.div(weights, sumOfWeights);
    }

    /**
     * Reference to the predict utility operation given issue utility weights and relative weights.
     * @param tf The tf Ops.
     * @param issueUtilities The issue utility weights.
     * @param weights The relative weights.
     * @return Operand to the predicted utility.
     */
    private static Operand<TFloat32> utility(Ops tf, Iterable<Operand<TFloat32>> issueUtilities, Operand<TFloat32> weights) {

        Stack<TFloat32> utilities = tf.stack(issueUtilities);

        Operand<TFloat32> normalizedWeights = normalizedWeights(tf, weights);

        Mul<TFloat32> inter1 = tf.math.mul(utilities, normalizedWeights);
        ReduceSum<TFloat32> result = tf.reduceSum(inter1, tf.constant(0));

        tf.ensureShape(result, Shape.scalar());

        return result;
    }
}
