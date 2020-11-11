package ai2020.group17.OpponentModel;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.DoubleStream;

import geniusweb.issuevalue.Domain;
import geniusweb.issuevalue.Value;
import geniusweb.profile.utilityspace.LinearAdditive;
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

    public static float[] repeat(float number, int n) {
        float[] array = new float[n];

        for (int i = 0; i < n; i++) {
            array[i] = number;
        }

        return array;
    }

    // Compute the squared error between two utility spaces
    public static double distance(LinearAdditive space1, LinearAdditive space2) {
        double dist = 0;

        assert space1.getDomain() == space2.getDomain();

        Domain domain = space1.getDomain();

        for (String issue: domain.getIssues()) {
            double weight = space1.getWeight(issue).doubleValue();

            double issueDist = 0.0;

            for (Value value: domain.getValues(issue)) {
                issueDist += Math.pow(space1.getUtilities().get(issue).getUtility(value).doubleValue() - space2.getUtilities().get(issue).getUtility(value).doubleValue(), 2);
            }

            dist += weight * issueDist;
        }

        return dist;
    }


    // If only there was numpy
    public static int drawFromDiscreteDist(ArrayList<Double> distribution) {
        double total =  distribution.stream().mapToDouble(x->x).sum();

        Random random = new Random();

        double cumulative = random.nextDouble() * total;

        int i = 0;
        do  {
            cumulative -= distribution.get(i);
            i++;
        } while (i < distribution.size() && cumulative > 0);

        return i - 1;
    }
}
