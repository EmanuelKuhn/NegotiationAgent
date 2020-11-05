import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class UtilityOfferStringToIntOptionsMapperTest {

    @Test
    void convertTrainingExample() {
        Map<String, List<String>> issues = Map.of("Issue1", List.of("Issue1 option1", "Issue1 option2"),
                "Issue2", List.of("Issue2 option1", "Issue2 option2", "Issue2 option3"));

        UtilityOfferStringToIntOptionsMapper mapper = new UtilityOfferStringToIntOptionsMapper(issues);

        UtilityModel.TrainingExample mappedExample = mapper.convertTrainingExample(
                new UtilityOfferStringToIntOptionsMapper.StringTrainingExample(
                        Map.of("Issue1", "Issue1 option2", "Issue2", "Issue2 option1"),
                        true)
        );

        assertEquals(mappedExample, new UtilityModel.TrainingExample(new int[] {1, 0}, true));
    }
}