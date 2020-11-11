package ai2020.group17;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.*;
import java.util.logging.Level;
import java.util.stream.Collectors;

import ai2020.group17.OpponentModel.TFLinearAdditiveOpponentModel;
import geniusweb.actions.Action;
import geniusweb.actions.Offer;
import geniusweb.actions.PartyId;
import geniusweb.actions.Vote;
import geniusweb.actions.Votes;
import geniusweb.inform.ActionDone;
import geniusweb.inform.Finished;
import geniusweb.inform.Inform;
import geniusweb.inform.OptIn;
import geniusweb.inform.Settings;
import geniusweb.inform.Voting;
import geniusweb.inform.YourTurn;
import geniusweb.issuevalue.Bid;
import geniusweb.party.Capabilities;
import geniusweb.party.DefaultParty;
import geniusweb.profile.PartialOrdering;
import geniusweb.profile.Profile;
import geniusweb.profile.utilityspace.LinearAdditive;
import geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace;
import geniusweb.profile.utilityspace.UtilitySpace;
import geniusweb.profileconnection.ProfileConnectionFactory;
import geniusweb.profileconnection.ProfileInterface;
import geniusweb.progress.Progress;
import geniusweb.progress.ProgressRounds;
import geniusweb.bidspace.BidsWithUtility;
import geniusweb.bidspace.Interval;
import tudelft.utilities.logging.Reporter;

/**
 * A simple party that places random bids and accepts when it receives an offer
 * with sufficient utility.
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
 * </table>
 */
public class Group17_Main extends DefaultParty {

	public static final double START_THRESHOLD = 0.9;
	public static final double END_THRESHOLD = 0.6;
	private Bid lastReceivedBid = null;
	private PartyId me;
	private final Random random = new Random();
	protected ProfileInterface profileint;
	private Progress progress;
	private Settings settings;
	private Votes lastvotes;
	private String protocol;
	private LinearAdditiveUtilitySpace space;
	private BidsWithUtility bidsWithUtility;
	private Interval range; 
	private BigDecimal maxRange;
	private BigDecimal minRange;
	private int firstBestBids = 5;


	private int minPower;
	private int maxPower;
	private int myPower;

	private Map<PartyId, LinearAdditive> opponentModelMap = new HashMap<>();
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
				this.settings = settings;
				this.protocol = settings.getProtocol().getURI().getPath();
				
				this.space = (LinearAdditiveUtilitySpace) this.profileint.getProfile();
				this.bidsWithUtility = new BidsWithUtility(this.space);
				
				this.range = bidsWithUtility.getRange();
				this.maxRange = range.getMax();
				this.minRange = range.getMin();

				Object val = settings.getParameters().get("minPower");
				this.minPower = (val instanceof Integer) ? (Integer) val : 76;

				val = settings.getParameters().get("maxPower");
				maxPower = (val instanceof Integer) ? (Integer) val
						: Integer.MAX_VALUE;

				val = settings.getParameters().get("power");
				myPower = (val instanceof Integer) ? (Integer) val
						: 1;

				System.out.println("minPower: " + minPower);
				System.out.println("maxPower: " + maxPower);


			} else if (info instanceof ActionDone) {
				Action otheract = ((ActionDone) info).getAction();
				
				if (otheract instanceof Offer) {
					lastReceivedBid = ((Offer) otheract).getBid();
				}

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
				getReporter().log(Level.INFO, "Final ourcome:" + info);
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
				new HashSet<>(Arrays.asList("SAOP", "AMOP", "MOPAC")),
				Collections.singleton(Profile.class));
	}

	@Override
	public String getDescription() {
		return "Agent that runs in MOPaC";
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
		switch (protocol) {
		case "SAOP":
		case "SHAOP":
			if (!(info instanceof YourTurn))
				return;
			break;
		case "MOPAC":
			if (!(info instanceof OptIn))
				// 
				return;
			break;
		default:
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
		Bid bid = null;

		if (round <= firstBestBids) {


			Interval firstInterval = new Interval(maxRange.multiply(BigDecimal.valueOf(0.9)), maxRange);
			
			if (this.bidsWithUtility.getBids(firstInterval).size().compareTo(BigInteger.valueOf(round)) >= 0) {
				bid = this.bidsWithUtility.getBids(firstInterval).get(round);
			} else {
				bid = this.bidsWithUtility.getExtremeBid(true);
			}
			

		} else {

			System.out.println("USING OPPONENT MODEL");

			double roundThreshold = computeRoundThreshold();

			System.out.println("before BidGeneration bidGeneration");

			BidGeneration bidGeneration = new BidGeneration(profile, powers, minPower);

			System.out.println("CREATED BidGeneration OBJECT");

			bid = bidGeneration.generateBid(opponentModelMap, roundThreshold);

			System.out.println("bid WAS GENERATED WITH OPPONENT MODEL");

		}

		action = new Offer(me, bid);
		
		getConnection().send(action);

	}

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
		
		if (profile instanceof UtilitySpace)
			return ((UtilitySpace) profile).getUtility(bid).doubleValue() >= t;
		
		if (profile instanceof PartialOrdering) {
			return ((PartialOrdering) profile).isPreferredOrEqual(bid,
					profile.getReservationBid());
		}
		return false;
	}

	/**
	 * @param voting the {@link Voting} object containing the options
	 * 
	 * @return our next Votes.
	 */
	private Votes vote(Voting voting) throws IOException {

		double roundThreshold = computeRoundThreshold();


		Set<Vote> votes = voting.getBids().stream().distinct()
				.filter(offer -> isGood(offer.getBid(), roundThreshold))
				.map(offer -> new Vote(me, offer.getBid(), minPower, maxPower))
				.collect(Collectors.toSet());
		return new Votes(me, votes);
	}

}
