package ai2020.group17.OpponentModel;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

// Class to map offers in the form of Map<String, String> to int[] offers accepted by the OpponentModel.UtilityModel class.
// First instantiate it with a Map of issue names to lists of possible options for the issue.
public class UtilityOfferStringToIntOptionsMapper {

    Map<String, Integer> issueIndices = new HashMap<>();
    List<Map<String, Integer>> issueOptionIndices = new ArrayList<>();

    int[] issuesOptions;

    public UtilityOfferStringToIntOptionsMapper(Map<String, List<String>> issues) {
        for (Map.Entry<String, List<String>> entry: issues.entrySet()) {
            Map<String, Integer> options = new HashMap<>();
            for (String option: entry.getValue()) {
                options.put(option, options.size());
            }

            issueOptionIndices.add(options);
            int index = issueOptionIndices.size() - 1;

            issueIndices.put(entry.getKey(), index);
        }

        issuesOptions = issueOptionIndices.stream().mapToInt(Map::size).toArray();
    }

    public int[] convertOptions(Map<String, String> input) {
        int[] options = new int[issueIndices.size()];

        for (Map.Entry<String, String> issue: input.entrySet()) {
            int issueIndex = issueIndices.get(issue.getKey());
            int optionIndex = issueOptionIndices.get(issueIndex).get(issue.getValue());

            options[issueIndex] = optionIndex;
        }

        return options;
    }

    public UtilityModel.TrainingExample convertTrainingExample(StringTrainingExample input) {
        return new UtilityModel.TrainingExample(convertOptions(input.options), input.accepted);
    }

    public static class StringTrainingExample {
        // input
        Map<String, String> options;

        // Expected output
        boolean accepted;

        public StringTrainingExample(Map<String, String> options, boolean accepted) {
            this.options = options;
            this.accepted = accepted;
        }
    }
}
