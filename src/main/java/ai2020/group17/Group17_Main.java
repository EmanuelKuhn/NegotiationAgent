package ai2020.group17;

import java.io.IOException;
import java.math.BigDecimal;
import java.util.*;
import java.util.logging.Level;
import java.util.stream.Collectors;

import ai2020.group17.OpponentModel.TFLinearAdditiveOpponentModel;
import geniusweb.actions.Action;
import geniusweb.actions.Offer;
import geniusweb.actions.PartyId;
import geniusweb.actions.Vote;
import geniusweb.actions.Votes;
import geniusweb.bidspace.AllBidsList;
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
import geniusweb.bidspace.pareto.ParetoLinearAdditive;
import tudelft.utilities.immutablelist.ImmutableList;
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

	private Bid lastReceivedBid = null;
	private PartyId me;
	private final Random random = new Random();
	protected ProfileInterface profileint;
	private Progress progress;
	private Settings settings;
	private Votes lastvotes;
	private String protocol;
	private HashMap<PartyId, HashMap<Bid, Double>> otherBids = new HashMap<>();
	private LinearAdditiveUtilitySpace space;
	private ParetoLinearAdditive pareto;
	private BidsWithUtility bidsWithUtility;
	private Interval range; 
	private BigDecimal maxRange;
	private BigDecimal minRange;
	private double threshold = 0.9;
	private int firstBestBids = 5;
	private ImmutableList<Bid> firstBids;
	private List<Element> elements = new ArrayList<>();
	private AllBidsList allBids;


	private int minPower;
	private int maxPower;

	private Map<PartyId, LinearAdditive> opponentModelMap = new HashMap<>();
	private Map<PartyId, Integer> powers;

	public Group17_Main() {
	}

	public Group17_Main(Reporter reporter) {
		super(reporter); // for debugging
	}

	// Returns a list of my utility space and all opponent model utility spaces
	private List<LinearAdditive> getUtilitySpaces() {
		List<LinearAdditive> result = new ArrayList<>();

		result.add((LinearAdditive) profileint);

		result.addAll(opponentModelMap.values());

		return result;
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

				System.out.println("minPower: " + minPower);
				System.out.println("maxPower: " + maxPower);


//				allBids = new AllBidsList(this.profileint.getProfile().getDomain());
				
//				UtilitySpace utility = (UtilitySpace) this.profileint.getProfile();
				
//				firstBids = bidsWithUtility.getBids(new Interval(maxRange.multiply(BigDecimal.valueOf(threshold)), maxRange));
				
//				while (firstBids.size().compareTo(BigInteger.valueOf(firstBestBids)) == -1 && BigInteger.valueOf(firstBestBids).compareTo(allBids.size()) <= 0) {
//					threshold -= 0.1;
//					firstBids = bidsWithUtility.getBids(new Interval(maxRange.multiply(BigDecimal.valueOf(threshold)), maxRange));
//				}
				
//				int index = 0;
//				for (Bid b : firstBids) {
//					elements.add(new Element(index, utility.getUtility(b)));
//					index++;
//				}
				
//				Collections.sort(elements);
//				Collections.reverse(elements);

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

				Set<Offer> offers = ((Voting) info).getBids().stream().filter(offer -> !offer.getActor().equals(me)).collect(Collectors.toSet());
				
				
				for (Offer offer : offers) {
					if (!otherBids.containsKey(offer.getActor())) {
						otherBids.put(offer.getActor(), new HashMap<>());
					}
					
					otherBids.get(offer.getActor()).put(offer.getBid(), ((UtilitySpace) profileint.getProfile()).getUtility(offer.getBid()).doubleValue());
				}
				
				for (PartyId id : otherBids.keySet()) {
					System.out.println(otherBids.get(id).toString() + id);
				}
				
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
		return "places random bids until it can accept an offer with utility >0.6. "
				+ "Parameters minPower and maxPower can be used to control voting behaviour.";
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

		UtilitySpace utility = (UtilitySpace) profile;
		
		int round = ((ProgressRounds) progress).getCurrentRound();
		Bid bid = null;

		if (round <= firstBestBids) {

//			bid = firstBids.get(BigInteger.valueOf(elements.get(round).index));

			bid = this.bidsWithUtility.getBids(new Interval(maxRange.multiply(BigDecimal.valueOf(0.9)), maxRange)).get(round);

		} else {
			// TODO implement with opponentmodel
//			bid = firstBids.get(BigInteger.valueOf(elements.get(round).index));

			System.out.println("USING OPPONENT MODEL");

			ProgressRounds progressRounds = ((ProgressRounds) progress);

			double progressDouble = (double) progressRounds.getCurrentRound() / (double) progressRounds.getTotalRounds();
			double roundThreshold = progressDouble * 0.6 + (1 - progressDouble) * 0.9;

			assert roundThreshold > 0 && roundThreshold < 1;

			System.out.println("before BidGeneration bidGeneration");

			BidGeneration bidGeneration = new BidGeneration(profile, powers, minPower);

			System.out.println("CREATED BidGeneration OBJECT");

			bid = bidGeneration.generateBid(opponentModelMap, roundThreshold);

			System.out.println("bid WAS GENERATED WITH OPPONENT MODEL");

		}

		action = new Offer(me, bid);
		
		getConnection().send(action);

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

		Set<Vote> votes = voting.getBids().stream().distinct()
				.filter(offer -> isGood(offer.getBid(), 0.8))
				.map(offer -> new Vote(me, offer.getBid(), minPower, maxPower))
				.collect(Collectors.toSet());
		return new Votes(me, votes);
	}

}

class Element implements Comparable<Element> {

    int index;
    BigDecimal value;

    Element(int index, BigDecimal value){
        this.index = index;
        this.value = value;
    }

	@Override
	public int compareTo(Element e) {
		
		return this.value.compareTo(e.value);
	}
	
	public BigDecimal getBigDecimal() {
		return this.value;
	}

}
