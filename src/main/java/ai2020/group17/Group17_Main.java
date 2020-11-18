package ai2020.group17;

import geniusweb.actions.*;
import geniusweb.bidspace.BidsWithUtility;
import geniusweb.bidspace.Interval;
import geniusweb.inform.*;
import geniusweb.issuevalue.Bid;
import geniusweb.party.Capabilities;
import geniusweb.party.DefaultParty;
import geniusweb.profile.Profile;
import geniusweb.profile.utilityspace.LinearAdditive;
import geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace;
import geniusweb.profile.utilityspace.UtilitySpace;
import geniusweb.profileconnection.ProfileConnectionFactory;
import geniusweb.profileconnection.ProfileInterface;
import geniusweb.progress.Progress;
import geniusweb.progress.ProgressRounds;
import tudelft.utilities.logging.Reporter;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.*;
import java.util.logging.Level;
import java.util.stream.Collectors;

/**
 * Party that generates random offers biased towards what we predict the opponents value.
 * It concedes slowly by reducing the minimum threshold of the offer proportional to the current round.
 *
 * Makes use of a custom opponent model implemented in tensorflow, and a custom bidding strategy that
 * generates random bids, which are biased to what we predict the opponents value based on the opponent models.
 *
 * <h2>parameters</h2>
 * <table >
 * <caption>parameters</caption>
 * <tr>
 * <td>minPower</td>
 * <td>This value is used as minPower for placed {@link Vote}s. Default value is
 * 2.</td>
 * </tr>
 * <tr>
 * <td>maxPower</td>
 * <td>This value is used as maxPower for placed {@link Vote}s. Default value is
 * infinity.</td>
 * </tr>
 * <tr>
 * <td>power</td>
 * <td>This value is used as power of the current agent. Default is 1</td>
 * </tr>
 * </table>
 */
public class Group17_Main extends DefaultParty {

	int DEFAULT_MIN_POWER = 2;

	public static final double START_THRESHOLD = 0.9;
	public static final double END_THRESHOLD = 0.6;
	private PartyId me;
	protected ProfileInterface profileint;
	private Progress progress;
	private Votes lastvotes;
	private String protocol;
	private BidsWithUtility bidsWithUtility;
	private BigDecimal maxRange;

	// Variable indicating how many rounds random very high valued bids should be placed.
	private final int firstBestBids = 5;

	private int minPower;
	private int maxPower;
	private int myPower;

	private final Map<PartyId, LinearAdditive> opponentModelMap = new HashMap<>();
	private Map<PartyId, Integer> powers;

	public Group17_Main() {
	}

	public Group17_Main(Reporter reporter) {
		super(reporter); // for debugging
	}

	@Override
	public void notifyChange(Inform info) {
		try {
			if (info instanceof Settings) {
				Settings settings = (Settings) info;
				this.profileint = ProfileConnectionFactory
						.create(settings.getProfile().getURI(), getReporter());
				this.me = settings.getID();
				this.progress = settings.getProgress();
				this.protocol = settings.getProtocol().getURI().getPath();

				LinearAdditiveUtilitySpace space = (LinearAdditiveUtilitySpace) this.profileint.getProfile();
				this.bidsWithUtility = new BidsWithUtility(space);

				Interval range = bidsWithUtility.getRange();
				this.maxRange = range.getMax();

				Object val = settings.getParameters().get("minPower");
				this.minPower = (val instanceof Integer) ? (Integer) val : DEFAULT_MIN_POWER;

				val = settings.getParameters().get("maxPower");
				maxPower = (val instanceof Integer) ? (Integer) val
						: Integer.MAX_VALUE;

				val = settings.getParameters().get("power");
				myPower = (val instanceof Integer) ? (Integer) val
						: 1;


			} else if (info instanceof ActionDone) {
				Action otheract = ((ActionDone) info).getAction();

				// Train the opponent models when a new Offer or Votes arrive.
				if (otheract instanceof Offer || otheract instanceof Votes) {
					PartyId actor = otheract.getActor();

					// initialize opponent model if not yet created
					opponentModelMap.putIfAbsent(actor, new TFLinearAdditiveOpponentModel(this.profileint.getProfile().getDomain()));

					// Update opponent with action
					TFLinearAdditiveOpponentModel opponentModel = (TFLinearAdditiveOpponentModel) opponentModelMap.get(actor);
					TFLinearAdditiveOpponentModel updatedOpponentModel = opponentModel.with(otheract, progress);
					opponentModelMap.put(actor, updatedOpponentModel);
				}
			} else if (info instanceof YourTurn) {
				makeOffer();
			} else if (info instanceof Finished) {
				getReporter().log(Level.INFO, "Final outcome:" + info);
			} else if (info instanceof Voting) {

				this.powers = ((Voting) info).getPowers();
				
				lastvotes = vote((Voting) info);
				getConnection().send(lastvotes);
			} else if (info instanceof OptIn) {
				// just repeat our last vote.
				getConnection().send(lastvotes);
			}
		} catch (Exception e) {
			throw new RuntimeException("Failed to handle info", e);
		}
		updateRound(info);
	}

	@Override
	public Capabilities getCapabilities() {
		return new Capabilities(
				new HashSet<>(Collections.singletonList("MOPAC")),
				Collections.singleton(Profile.class));
	}

	@Override
	public String getDescription() {
		return "Group 17 agent that runs in MOPaC.";
	}

	/**
	 * Update {@link #progress}
	 * 
	 * @param info the received info. Used to determine if this is the last info
	 *             of the round
	 */
	private void updateRound(Inform info) {
		if (protocol == null)
			return;
		if ("MOPAC".equals(protocol)) {
			if (!(info instanceof OptIn))
				//
				return;
		} else {
			return;
		}
		// if we get here, round must be increased.
		if (progress instanceof ProgressRounds) {
			progress = ((ProgressRounds) progress).advance();
		}

	}

	/**
	 * send our next offer
	 */
	private void makeOffer() throws IOException {
		Action action;
		LinearAdditive profile;

		try {
			profile = (LinearAdditive) profileint.getProfile();
		} catch (IOException e) {
			throw new IllegalStateException(e);
		}
		
		int round = ((ProgressRounds) progress).getCurrentRound();
		Bid bid;

		if (round <= firstBestBids) {
			// Generate random good bid for first few rounds

			Interval firstInterval = new Interval(maxRange.multiply(BigDecimal.valueOf(0.9)), maxRange);
			
			if (this.bidsWithUtility.getBids(firstInterval).size().compareTo(BigInteger.valueOf(round)) >= 0) {
				bid = this.bidsWithUtility.getBids(firstInterval).get(round);
			} else {
				bid = this.bidsWithUtility.getExtremeBid(true);
			}
		} else {
			// Generate bid using opponent model

			// Compute threshold to use this round
			double roundThreshold = computeRoundThreshold();

			BidGeneration bidGeneration = new BidGeneration(profile, powers, minPower - myPower);
			bid = bidGeneration.generateBid(opponentModelMap, roundThreshold);

		}

		action = new Offer(me, bid);
		getConnection().send(action);
	}

	/**
	 * Compute round threshold by interpolating between start and end thresholds.
	 * @return Interpolated threshold
	 */
	private double computeRoundThreshold() {
		ProgressRounds progressRounds = ((ProgressRounds) progress);

		double progressDouble = (double) progressRounds.getCurrentRound() / (double) progressRounds.getTotalRounds();
		double roundThreshold = progressDouble * END_THRESHOLD + (1 - progressDouble) * START_THRESHOLD;

		return roundThreshold;
	}

	/**
	 * @param bid the bid to check
	 * @return true iff bid is good for us.
	 */
	private boolean isGood(Bid bid, double t) {
		if (bid == null)
			return false;
		Profile profile;
		try {
			profile = profileint.getProfile();
		} catch (IOException e) {
			throw new IllegalStateException(e);
		}
		
		return ((UtilitySpace) profile).getUtility(bid).doubleValue() >= t;
	}

	/**
	 * @param voting the {@link Voting} object containing the options
	 * 
	 * @return our next Votes.
	 */
	private Votes vote(Voting voting) throws IOException {

		double roundThreshold = 0.9 * computeRoundThreshold();


		Set<Vote> votes = voting.getBids().stream().distinct()
				.filter(offer -> isGood(offer.getBid(), roundThreshold))
				.map(offer -> new Vote(me, offer.getBid(), minPower, maxPower))
				.collect(Collectors.toSet());
		return new Votes(me, votes);
	}

}
