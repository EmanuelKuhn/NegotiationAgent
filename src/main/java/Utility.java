import org.jetbrains.annotations.NotNull;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.*;
import org.tensorflow.op.math.Div;
import org.tensorflow.op.math.Mul;
import org.tensorflow.op.train.ApplyGradientDescent;
import org.tensorflow.types.TFloat32;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

public class Utility {

    Issue[] issues;
    Map<String, Integer> issuesIdx = new HashMap<>();

    Variable<TFloat32> weights;
//    Assign<TFloat32> weightsInit;

    public Utility(Ops tf, int[] issuesOptions) {

        issues = new Issue[issuesOptions.length];

        for(int i = 0; i < issues.length; i++) {
            issues[i] = new Issue(tf, issuesOptions[i], i);
        }

        weights = tf.variable(Shape.of(issues.length), TFloat32.DTYPE);
    }

    public Assign<TFloat32> initWeights(Ops tf, float initializationValue) {
        return initWeights(tf, () -> Helper.repeat(initializationValue, issues.length));
    }

    public List<Assign<TFloat32>> initIssueWeights(Ops tf, float initializationValue) {

        List<Assign<TFloat32>> assigns = new ArrayList<>();

        for (Issue issue:issues) {
            assigns.add(issue.init(tf, initializationValue));
        }

        return assigns;
    }

    public Assign<TFloat32> initWeights(Ops tf, Supplier<float[]> initializer) {
        return tf.assign(weights, tf.constant(initializer.get()));
    }

    public Operand<TFloat32> predictUtility(Ops tf) {
        ArrayList<Operand<TFloat32>> issueUtilities = new ArrayList<>();

        for (int i = 0; i < this.issues.length; i++) {
            issueUtilities.add(this.issues[i].predictUtility(tf));
        }

        return utility(tf, issueUtilities, weights);
    }

    public void feed(Session.Runner runner, int[] options) {
        for (int i = 0; i < issues.length; i++) {
            issues[i].feed(runner, options[i]);
        }
    }

    private List<Variable<TFloat32>> getWeightVariables() {
        List<Variable<TFloat32>> variableList = new ArrayList<>();

        for (int i = 0; i < issues.length; i++) {
            variableList.add(issues[i].issueWeights);
        }

        variableList.add(weights);
        return variableList;
    }

    public List<ApplyGradientDescent<TFloat32>> applyGradientDescent(Ops tf, Operand<TFloat32> loss, Operand<TFloat32> alpha) {

        List<Variable<TFloat32>> variableList = getWeightVariables();

        Gradients gradients = tf.gradients(loss, variableList);

        List<ApplyGradientDescent<TFloat32>> gradientDescentList = new ArrayList<>();

        for (int i = 0; i < variableList.size(); i++) {
            gradientDescentList.add(tf.train.applyGradientDescent(variableList.get(i), alpha, gradients.dy(i)));
        }

        return gradientDescentList;
    }

    private static Operand<TFloat32> utility(Ops tf, Iterable<Operand<TFloat32>> issueUtilities, Operand<TFloat32> weights) {

        Stack<TFloat32> utilities = tf.stack(issueUtilities);

        ReduceSum<TFloat32> sumOfWeights = tf.reduceSum(weights, tf.constant(0));
        Div<TFloat32> normalizedWeights = tf.math.div(weights, sumOfWeights);

        Mul<TFloat32> inter1 = tf.math.mul(utilities, normalizedWeights);
        ReduceSum<TFloat32> result = tf.reduceSum(inter1, tf.constant(0));

        tf.ensureShape(result, Shape.scalar());

        return (result);
    }
}
