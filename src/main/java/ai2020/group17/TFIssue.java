package ai2020.group17;

import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Assign;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Variable;
import org.tensorflow.op.math.Mul;
import org.tensorflow.types.TFloat32;

import java.util.function.Supplier;

import static org.tensorflow.op.core.Placeholder.shape;

/**
 * Class that sets up and keeps tf variables for a single issue.
 */
public class TFIssue {

    // Just for debugging
    int issueIdx;

    // Placeholder for the vector that represents which issue option is chosen,
    // e.g. [0, 0, 1, 0] means that the 3rd option is chosen
    Placeholder<TFloat32> issueOneHotVectorPlaceholder;

    // tf variable for the utilities that will be trained.
    Variable<TFloat32> issueWeights;

    int nOptions;

    public TFIssue(Ops tf, int nOptions, int issueIdx) {
        this.issueIdx = issueIdx;

        this.nOptions = nOptions;

        // Setup placeholders and variables
        issueOneHotVectorPlaceholder = tf.placeholder(TFloat32.DTYPE, shape(Shape.of(nOptions)));
        issueWeights = tf.variable(Shape.of(nOptions), TFloat32.DTYPE);
    }

    /**
     * Initialize the tf variable for the utility weights.
     *
     * @param tf The tf Ops
     * @param initializer How to initialize the utility weights.
     * @return The tf assign operation.
     */
    public Assign<TFloat32> init(Ops tf, Supplier<float[]> initializer) {
        return tf.assign(issueWeights, tf.constant(initializer.get()));
    }

    /**
     * Initialize the tf variable for the utility weights with a constant number.
     *
     * @param tf The tf Ops
     * @param initializationValue The value each weight should be initialized to.
     * @return The tf assign operation.
     */
    public Assign<TFloat32> init(Ops tf, float initializationValue) {
        return init(tf, () -> Helper.repeat(initializationValue, nOptions));
    }

    /**
     * Feed a chosen option as input.
     *
     * @param runner The tf runner
     * @param option Index of which option is chosen for this issue.
     */
    public void feed(Session.Runner runner, int option) {
        runner.feed(issueOneHotVectorPlaceholder.asOutput(), TFloat32.vectorOf(getOneHotVector(option)));
    }

    /**
     * Return a reference to a tf operand representing the predicted utility.
     *
     * @param tf The tf Ops.
     * @return tf operand of predicted utility
     */
    Operand<TFloat32> predictUtility(Ops tf) {
        return issueUtility(tf, issueOneHotVectorPlaceholder, issueWeights);
    }

    /**
     * Convert index to one hot vector.
     * @param option The index.
     * @return resulting one hot vector
     */
    private float[] getOneHotVector(int option) {
        float[] array = new float[nOptions];

        array[option] = 1.0f;

        return array;
    }

    // Returns a reference to the weights clipped between 0 and 1
    protected static Operand<TFloat32> weightsClipped(Ops tf, Operand<TFloat32> weights) {
        return Helper.clipByValuePreserveGradient(
                tf,
                weights,
                tf.constant(0.0f),
                tf.constant(1.0f)
        );
    }

    /**
     * Compute the utility of the current issue for a given option.
     * @param tf The tf Ops.
     * @param oneHotVector Vector representing the chosen option.
     * @param weights This issue's utility weights
     *
     * @return reference to the utility computation for this issue given a chosen option.
     */
    private static Operand<TFloat32> issueUtility(Ops tf, Operand<TFloat32> oneHotVector, Operand<TFloat32> weights) {

        Operand<TFloat32> weightsClipped = weightsClipped(tf, weights);

        Mul<TFloat32> inter1 = tf.math.mul(oneHotVector, weightsClipped);

        Operand<TFloat32> result = tf.sum(inter1, tf.constant(0));

        tf.ensureShape(result, Shape.scalar());

        return result;
    }
}
