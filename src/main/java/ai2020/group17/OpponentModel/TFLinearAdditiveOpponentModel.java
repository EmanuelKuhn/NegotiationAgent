package ai2020.group17.OpponentModel;

import geniusweb.actions.Action;
import geniusweb.actions.Offer;
import geniusweb.actions.Vote;
import geniusweb.actions.Votes;
import geniusweb.issuevalue.*;
import geniusweb.opponentmodel.OpponentModel;
import geniusweb.profile.utilityspace.DiscreteValueSetUtilities;
import geniusweb.profile.utilityspace.LinearAdditive;
import geniusweb.profile.utilityspace.ValueSetUtilities;
import geniusweb.progress.Progress;

import java.math.BigDecimal;
import java.util.*;
import java.util.stream.Stream;

// Opponent model that attempts to infer opponent's LinearAdditive utility function through gradient descent on a Tensorflow model.
public class TFLinearAdditiveOpponentModel implements OpponentModel, LinearAdditive {

	private Domain domain;
	private UtilityOfferStringToIntOptionsMapper mapper = null;

	private TFUtilityModel tfModel;

	private List<TFUtilityModel.TrainingExample> trainingData = new ArrayList<>();

	public TFLinearAdditiveOpponentModel() {

	}

	public TFLinearAdditiveOpponentModel(Domain domain, UtilityOfferStringToIntOptionsMapper mapper, TFUtilityModel tfModel,
										 List<TFUtilityModel.TrainingExample> trainingData) {
		this.domain = domain;
		this.mapper = mapper;
		this.tfModel = tfModel;
		this.trainingData = trainingData;
	}
	
	
	public TFLinearAdditiveOpponentModel(Domain domain) {
		this.domain = domain;

		Map<String, ValueSet> issuesOptions = new HashMap<>();

		for(String issue: this.domain.getIssues()) {
			issuesOptions.put(issue, domain.getValues(issue));
		}

		mapper = new UtilityOfferStringToIntOptionsMapper(issuesOptions);

		this.tfModel = new TFUtilityModel(mapper.issuesOptions);
	}

	
	@Override
	public String getName() {
		return "LinearAdditiveOpponentModel";
	}

	@Override
	public Domain getDomain() {
		return this.domain;
	}

	@Override
	public Bid getReservationBid() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public TFLinearAdditiveOpponentModel with(Domain domain, Bid resBid) {
		return new TFLinearAdditiveOpponentModel(domain);
	}

	@Override
	public TFLinearAdditiveOpponentModel with(Action action, Progress progress) {
		TFUtilityModel tfModel = this.tfModel;
		this.tfModel = null;

		if (action instanceof Votes) {

			Votes votes = (Votes) action;

			for (Vote vote : votes.getVotes()) {
				Bid bid = vote.getBid();

				// Heuristic for if a vote is accepted or not in a vote.
				boolean isAccepted = vote.getMinPower() < vote.getMaxPower() || vote.getMinPower() < Integer.MAX_VALUE;

				// Create a training example from the bid
				int[] bidOptions = mapper.convertOptions(bid.getIssueValues());
				TFUtilityModel.TrainingExample trainingExample = new TFUtilityModel.TrainingExample(bidOptions, isAccepted);

				trainingData.add(trainingExample);
			}
		} else if (action instanceof Offer) {
			Offer offer = (Offer) action;

			Bid bid = offer.getBid();

			// Create a training example from the bid
			int[] bidOptions = mapper.convertOptions(bid.getIssueValues());
			TFUtilityModel.TrainingExample trainingExample = new TFUtilityModel.TrainingExample(bidOptions, true);

			trainingData.add(trainingExample);
		}

		tfModel.train(this.trainingData);

		return new TFLinearAdditiveOpponentModel(domain, mapper, tfModel, trainingData);
	}

	@Override
	public Map<String, ValueSetUtilities> getUtilities() {

		Map<String, ValueSetUtilities> valueSetUtilitiesMap = new HashMap<>();

		for (String issueName: this.domain.getIssues()) {

			int issueIndex = mapper.issueIndices.get(issueName);

			double[] weights = this.tfModel.computeIssueWeights(issueIndex);

			Map<DiscreteValue, BigDecimal> result = new HashMap<>();

			for (int i = 0; i < weights.length; i++) {
				result.put((DiscreteValue) mapper.mapOptionIndexToValue(issueIndex, i), BigDecimal.valueOf(weights[i]));
			}

			valueSetUtilitiesMap.put(issueName, new DiscreteValueSetUtilities(result));
		}

		return valueSetUtilitiesMap;
	}

	@Override
	public Map<String, BigDecimal> getWeights() {
		double[] weights = this.tfModel.computeWeights();

		Map<String, BigDecimal> result = new HashMap<>();

		for (int i = 0; i < weights.length; i++) {
			result.put(mapper.mapIndexToIssue(i), BigDecimal.valueOf(weights[i]));
		}

		return result;
	}

	@Override
	public BigDecimal getWeight(String issue) {
		return getWeights().get(issue);
	}

	@Override
	public BigDecimal getUtility(Bid bid) {

		double utility = 0;

		Map<String, ValueSetUtilities> utilities = this.getUtilities();
		Map<String, BigDecimal> weights = this.getWeights();

		for(String issue: bid.getIssues()) {
			Value  value = bid.getIssueValues().get(issue);

			utility += weights.get(issue).doubleValue() * utilities.get(issue).getUtility(value).doubleValue();
		}

		return BigDecimal.valueOf(utility);
	}
}
