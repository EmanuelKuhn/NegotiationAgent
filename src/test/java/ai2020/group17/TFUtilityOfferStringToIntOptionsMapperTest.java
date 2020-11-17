package ai2020.group17.OpponentModel;

import org.junit.jupiter.api.Test;

import geniusweb.issuevalue.DiscreteValue;
import geniusweb.issuevalue.DiscreteValueSet;
import geniusweb.issuevalue.ValueSet;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class TFUtilityOfferStringToIntOptionsMapperTest {

    @Test
    void convertTrainingExample() {
        Map<String, ValueSet> issues = Map.of("Menu", new DiscreteValueSet(new DiscreteValue("Fish"), new DiscreteValue("Meat")),
        		"Cost", new DiscreteValueSet(new DiscreteValue("2000 EUR"), new DiscreteValue("5000 EUR"), new DiscreteValue("9000 EUR")));

        UtilityOfferStringToIntOptionsMapper mapper = new UtilityOfferStringToIntOptionsMapper(issues);

        TFUtilityModel.TrainingExample mappedExample = mapper.convertTrainingExample(
                new UtilityOfferStringToIntOptionsMapper.StringTrainingExample(
                        Map.of("Menu", new DiscreteValue("Meat"), "Cost", new DiscreteValue("2000 EUR")),
                        true)
        );
        

        assertEquals(mapper.issueOptionIndices.get(mapper.issueIndices.get("Menu")).get(new DiscreteValue("Meat")), mappedExample.options[mapper.issueIndices.get("Menu")]);
        assertEquals(mapper.issueOptionIndices.get(mapper.issueIndices.get("Cost")).get(new DiscreteValue("2000 EUR")), mappedExample.options[mapper.issueIndices.get("Cost")]);
        assertTrue(mappedExample.accepted);
    }

    @Test
    void testModelIntegrationWorks() {
        Map<String, ValueSet> issues = Map.of("Menu", new DiscreteValueSet(new DiscreteValue("Fish"), new DiscreteValue("Meat")),
        		"Cost", new DiscreteValueSet(new DiscreteValue("2000 EUR"), new DiscreteValue("5000 EUR"), new DiscreteValue("9000 EUR")));

        UtilityOfferStringToIntOptionsMapper mapper = new UtilityOfferStringToIntOptionsMapper(issues);

        TFUtilityModel.TrainingExample example1 = mapper.convertTrainingExample(
                new UtilityOfferStringToIntOptionsMapper.StringTrainingExample(
                        Map.of("Menu", new DiscreteValue("Meat"), "Cost", new DiscreteValue("2000 EUR")),
                        true));

                TFUtilityModel.TrainingExample example2 = mapper.convertTrainingExample(
                        new UtilityOfferStringToIntOptionsMapper.StringTrainingExample(
                                Map.of("Menu", new DiscreteValue("Meat"), "Cost", new DiscreteValue("5000 EUR")),
                                true));

                TFUtilityModel.TrainingExample example3 = mapper.convertTrainingExample(
                        new UtilityOfferStringToIntOptionsMapper.StringTrainingExample(
                                Map.of("Menu", new DiscreteValue("Fish"), "Cost", new DiscreteValue("5000 EUR")),
                                false));


        // Instantiate model with issue structure as specified in the mapper
        TFUtilityModel model = new TFUtilityModel(mapper.issuesOptions);

        model.train(List.of(example1, example2, example3));

        float prediction = model.predict(mapper.convertOptions(Map.of("Menu", new DiscreteValue("Fish"), "Cost", new DiscreteValue("5000 EUR"))));

        assertTrue(prediction < 0.5);
    }

}