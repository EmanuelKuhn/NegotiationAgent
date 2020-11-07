package ai2020.group17.OpponentModel;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TType;

public class Helper {
    public static <T extends TType> Operand<T> clipByValuePreserveGradient(Ops tf, Operand<T> t, Operand<T> clipValueMin, Operand<T> clipValueMax) {
        // Clip, but stop gradients (clipByValue gradients are not supported in the java version, plus
        // it might not matter a lot see: https://github.com/tensorflow/tensorflow/issues/44333)
        // Stopping gradients adapted from: https://github.com/tensorflow/probability/blob/v0.11.1/tensorflow_probability/python/math/numeric.py#L68-L100

        return tf.math.add(t, tf.stopGradient(tf.math.sub(tf.clipByValue(t,
                clipValueMin,
                clipValueMax), t)));
    }

    public static float[] oneHot(int size, int hotIndex) {
        float[] array = new float[size];

        array[hotIndex] = 1.0f;

        return array;
    }

    public static float[] floatArray(float... numbers) {
        return numbers;
    }

    public static float[] repeat(float number, int n) {
        float[] array = new float[n];

        for (int i = 0; i < n; i++) {
            array[i] = number;
        }

        return array;
    }
}