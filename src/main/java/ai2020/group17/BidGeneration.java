package ai2020.group17;

import geniusweb.actions.PartyId;
import geniusweb.issuevalue.*;
import geniusweb.profile.utilityspace.LinearAdditive;
import geniusweb.profile.utilityspace.ValueSetUtilities;
import tudelft.utilities.immutablelist.Tuple;

import java.math.BigDecimal;
import java.util.*;
import java.util.stream.Collectors;

import static ai2020.group17.OpponentModel.Helper.distance;
import static ai2020.group17.OpponentModel.Helper.drawFromDiscreteDist;

public class BidGeneration {

    private LinearAdditive myProfile;
    private Map<PartyId, Integer> powers;
    private int minPower;

    public BidGeneration(LinearAdditive myProfile, Map<PartyId, Integer> powers, int minPower) {
        this.myProfile = myProfile;

        this.powers = powers;
        this.minPower = minPower;
    }


    public Bid generateBid(Map<PartyId, LinearAdditive> opponents, double minTotalUtility) {

        List<Tuple<PartyId, Double>> partyDistances = new ArrayList<>();

        for(Map.Entry<PartyId, LinearAdditive> opp: opponents.entrySet()) {
            partyDistances.add(new Tuple<>(opp.getKey(), distance(this.myProfile, opp.getValue())));
        }

        partyDistances.sort(Collections.reverseOrder(Comparator.comparingDouble(Tuple::get2)));

        // List of closest parties that have enough power
        List<PartyId> targetParties = new ArrayList<>();

        for (int i = 0; i < partyDistances.size() && targetParties.stream().mapToInt(this.powers::get).sum() < minPower; i++) {
            targetParties.add(partyDistances.get(i).get1());
        }

        assert targetParties.stream().mapToInt(this.powers::get).sum() >= minPower;

        Map<String, BigDecimal> issueWeights = myProfile.getWeights();

        // Start by drawing for low weighted issues
        // TODO: make probabilistic?
        List<String> issueDrawOrder = this.myProfile.getDomain().getIssues().stream()
                .sorted(Comparator.comparing(issueWeights::get))
                .collect(Collectors.toList());

        Bid bid = null;

        for (String issue: issueDrawOrder) {
            bid = generatePartialBid(issue, bid, minTotalUtility, targetParties.stream().collect(Collectors.toMap(party -> party, opponents::get)));
        }

        return bid;
    }

    private Bid generatePartialBid(String issue, Bid currentPartialBid, double minTotalUtility, Map<PartyId, LinearAdditive> parties) {

        Set<String> remainingIssues = myProfile.getDomain().getIssues();
        if (currentPartialBid != null) {
            remainingIssues.removeAll(currentPartialBid.getIssues());
        }
        remainingIssues.remove(issue);

        double maxRemainingUtility = calculateMaxRemainingUtility(this.myProfile, remainingIssues);

        double currentUtility = this.myProfile.getUtility(currentPartialBid).doubleValue();

        double minUtilityForCurrentIssue = minTotalUtility - currentUtility - maxRemainingUtility;

        List<DiscreteValue> consideredValues = new ArrayList<>();

        for (Value value: this.myProfile.getDomain().getValues(issue)) {
            double utility = this.myProfile.getUtility(new Bid(issue, value)).doubleValue();

            if (utility >= minUtilityForCurrentIssue) {
                consideredValues.add((DiscreteValue) value);
            }
        }

        Bid currentIssuepartialBid = new Bid(issue, drawIssueValue(issue, consideredValues, parties));

        if (currentPartialBid == null) {
            return currentIssuepartialBid;
        }

        return currentPartialBid.merge(currentIssuepartialBid);
    }

    // Draw an issue value proportional to the summed utility of the given parties
    private Value drawIssueValue(String issue, List<DiscreteValue> valueSet, Map<PartyId, LinearAdditive> parties) {

        List<Value> possibleValues = new ArrayList<>();
        ArrayList<Double> distribution = new ArrayList<>();

        for(Map.Entry<Value, Double> entry: computeExpectedSumUtilities(issue, valueSet, parties).entrySet()) {
            possibleValues.add(entry.getKey());
            distribution.add(entry.getValue());
        }

        int index = drawFromDiscreteDist(distribution);

        return possibleValues.get(index);
    }

    // Compute the sum of utilities over all parties per issue value
    private Map<Value, Double> computeExpectedSumUtilities(String issue, List<DiscreteValue> valueSet, Map<PartyId, LinearAdditive> parties) {
        Map<Value, Double> expectedSumUtilities = new HashMap<>();

        for (Value value: valueSet) {
            Bid bid = new Bid(issue, value);

            expectedSumUtilities.put(value, parties.values().stream().mapToDouble(space -> space.getUtility(bid).doubleValue()).sum());
        }

        return expectedSumUtilities;
    }

    // Compute the maximum remaining utility if the maximum value is chosen for each remaining issue
    public static double calculateMaxRemainingUtility(LinearAdditive profile,  Set<String> remainingIssues) {

        double utility = 0;

        Map<String, ValueSetUtilities> utilities = profile.getUtilities();
        Map<String, BigDecimal> weights = profile.getWeights();

        for(String issue: remainingIssues) {
            List<Value> possibleValues = new ArrayList<>();

            profile.getDomain().getValues(issue).forEach(possibleValues::add);

            Value value = possibleValues.stream().max(Comparator.comparingDouble(v -> utilities.get(issue).getUtility(v).doubleValue())).get();

            utility += weights.get(issue).doubleValue() * utilities.get(issue).getUtility(value).doubleValue();
        }

        return utility;
    }


}
