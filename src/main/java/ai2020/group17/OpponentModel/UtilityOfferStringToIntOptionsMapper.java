package ai2020.group17.OpponentModel;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import geniusweb.issuevalue.Value;
import geniusweb.issuevalue.ValueSet;

// Class to map offers in the form of Map<String, String> to int[] offers accepted by the OpponentModel.UtilityModel class.
// First instantiate it with a Map of issue names to lists of possible options for the issue.
public class UtilityOfferStringToIntOptionsMapper {

    Map<String, Integer> issueIndices = new HashMap<>();
    List<String> issueNames = new ArrayList<>();

    List<Map<Value, Integer>> issueOptionIndices = new ArrayList<>();
    List<List<Value>> issueOptionValues = new ArrayList<>();

    int[] issuesOptions;
    
    public UtilityOfferStringToIntOptionsMapper(Map<String, ValueSet> issues) {
        for (Entry<String, ValueSet> entry: issues.entrySet()) {
            Map<Value, Integer> options = new HashMap<>();
            List<Value> values = new ArrayList<>();
            for (Value option: entry.getValue()) {
                options.put(option, options.size());
                values.add(option);
            }

            issueOptionIndices.add(options);
            issueOptionValues.add(values);
            int index = issueOptionIndices.size() - 1;

            issueIndices.put(entry.getKey(), index);
            issueNames.add(entry.getKey());
        }

        issuesOptions = issueOptionIndices.stream().mapToInt(Map::size).toArray();
    }

    public int[] convertOptions(Map<String, Value> input) {
        int[] options = new int[issueIndices.size()];

        for (Map.Entry<String, Value> issue: input.entrySet()) {
            int issueIndex = issueIndices.get(issue.getKey());
            int optionIndex = issueOptionIndices.get(issueIndex).get(issue.getValue());

            options[issueIndex] = optionIndex;
        }

        return options;
    }

    public String mapIndexToIssue(int issueIndex) {
        return issueNames.get(issueIndex);
    }

    public Value mapOptionIndexToValue(int issueIndex, int optionIndex) {
        return issueOptionValues.get(issueIndex).get(optionIndex);
    }

    public TFUtilityModel.TrainingExample convertTrainingExample(StringTrainingExample input) {
        return new TFUtilityModel.TrainingExample(convertOptions(input.options), input.accepted);
    }

    public static class StringTrainingExample {
        // input
        Map<String, Value> options;

        // Expected output
        boolean accepted;

        public StringTrainingExample(Map<String, Value> options, boolean accepted) {
            this.options = options;
            this.accepted = accepted;
        }
    }
}
