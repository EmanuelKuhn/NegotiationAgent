package ai2020.group17.OpponentModel;

import org.junit.jupiter.api.Test;

import ai2020.group17.OpponentModel.UtilityModel;
import ai2020.group17.OpponentModel.UtilityOfferStringToIntOptionsMapper;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class UtilityOfferStringToIntOptionsMapperTest {

    @Test
    void convertTrainingExample() {
        Map<String, List<String>> issues = Map.of("Menu", List.of("Menu Fish", "Menu Meat"),
                "Cost", List.of("Cost 2000 EUR", "Cost 5000 EUR", "Cost 9000 EUR"));

        UtilityOfferStringToIntOptionsMapper mapper = new UtilityOfferStringToIntOptionsMapper(issues);

        UtilityModel.TrainingExample mappedExample = mapper.convertTrainingExample(
                new UtilityOfferStringToIntOptionsMapper.StringTrainingExample(
                        Map.of("Menu", "Menu Meat", "Cost", "Cost 2000 EUR"),
                        true)
        );

        assertEquals(mappedExample, new UtilityModel.TrainingExample(new int[] {1, 0}, true));
    }

    @Test
    void testModelIntegrationWorks() {
        Map<String, List<String>> issues = Map.of("Issue1", List.of("Issue1 option1", "Issue1 option2"),
                "Issue2", List.of("Issue2 option1", "Issue2 option2", "Issue2 option3"));

        UtilityOfferStringToIntOptionsMapper mapper = new UtilityOfferStringToIntOptionsMapper(issues);

        UtilityModel.TrainingExample example1 = mapper.convertTrainingExample(
                new UtilityOfferStringToIntOptionsMapper.StringTrainingExample(
                        Map.of("Issue1", "Issue1 option2", "Issue2", "Issue2 option1"),
                        true)
        );

        UtilityModel.TrainingExample example2 = mapper.convertTrainingExample(
                new UtilityOfferStringToIntOptionsMapper.StringTrainingExample(
                        Map.of("Issue1", "Issue1 option2", "Issue2", "Issue2 option2"),
                        true)
        );
        UtilityModel.TrainingExample example3 = mapper.convertTrainingExample(
                new UtilityOfferStringToIntOptionsMapper.StringTrainingExample(
                        Map.of("Issue1", "Issue1 option1", "Issue2", "Issue2 option2"),
                        false)
        );

        // Instantiate model with issue structure as specified in the mapper
        UtilityModel model = new UtilityModel(mapper.issuesOptions);

        model.train(List.of(example1, example2, example3));

        float prediction = model.predict(mapper.convertOptions(Map.of("Issue1", "Issue1 option1", "Issue2", "Issue2 option2")));

        assertTrue(prediction < 0.5);
    }

}