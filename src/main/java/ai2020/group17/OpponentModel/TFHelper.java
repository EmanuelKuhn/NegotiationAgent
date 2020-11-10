package ai2020.group17.OpponentModel;

import java.util.stream.Stream;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TType;

public class TFHelper {
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

//    // I miss Python
//    public static Stream<Integer> range(int from, int to) {
////    	return Stream.iterate(from, (i) -> to > i, (i) -> i + 1);
//    }
    
//    // I miss Python
//    public static Stream<Integer> range(int to) {
//    	return range(0, to);
//    }
    
    
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
