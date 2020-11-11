package ai2020.group17.OpponentModel;

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

public class TFIssue {

    // Just for debugging
    int issueIdx;

    Placeholder<TFloat32> issueOneHotVectorPlaceholder;
    Variable<TFloat32> issueWeights;
//    Assign<TFloat32> issueWeightsInit;

    int nOptions;

    public TFIssue(Ops tf, int nOptions, int issueIdx) {
        this.issueIdx = issueIdx;

        this.nOptions = nOptions;

        // Setup placeholders and variables
        issueOneHotVectorPlaceholder = tf.placeholder(TFloat32.DTYPE, shape(Shape.of(nOptions)));
        issueWeights = tf.variable(Shape.of(nOptions), TFloat32.DTYPE);
    }

    public Assign<TFloat32> init(Ops tf, Supplier<float[]> initializer) {
        return tf.assign(issueWeights, tf.constant(initializer.get()));
    }

    public void feed(Session.Runner runner, int option) {
        runner.feed(issueOneHotVectorPlaceholder.asOutput(), TFloat32.vectorOf(getOneHotVector(option)));
    }

    Operand<TFloat32> predictUtility(Ops tf) {
        return issueUtility(tf, issueOneHotVectorPlaceholder, issueWeights);
    }

    private float[] getOneHotVector(int option) {
        float[] array = new float[nOptions];

        array[option] = 1.0f;

        return array;
    }

    public Assign<TFloat32> init(Ops tf, float initializationValue) {
        return init(tf, () -> Helper.repeat(initializationValue, nOptions));
    }

    protected static Operand<TFloat32> weightsClipped(Ops tf, Operand<TFloat32> weights) {
        return Helper.clipByValuePreserveGradient(
                tf,
                weights,
                tf.constant(0.0f),
                tf.constant(1.0f)
        );
    }

    private static Operand<TFloat32> issueUtility(Ops tf, Operand<TFloat32> oneHotVector, Operand<TFloat32> weights) {

        Operand<TFloat32> weightsClipped = weightsClipped(tf, weights);

        Mul<TFloat32> inter1 = tf.math.mul(oneHotVector, weightsClipped);

        Operand<TFloat32> result = tf.sum(inter1, tf.constant(0));

        tf.ensureShape(result, Shape.scalar());

        return result;
    }
}
