package ai2020.group17;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.*;
import java.util.logging.Level;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import ai2020.group17.OpponentModel.TFLinearAdditiveOpponentModel;
import geniusweb.actions.Accept;
import geniusweb.actions.Action;
import geniusweb.actions.Offer;
import geniusweb.actions.PartyId;
import geniusweb.actions.Vote;
import geniusweb.actions.Votes;
import geniusweb.bidspace.AllBidsList;
import geniusweb.bidspace.AllPartialBidsList;
import geniusweb.inform.ActionDone;
import geniusweb.inform.Finished;
import geniusweb.inform.Inform;
import geniusweb.inform.OptIn;
import geniusweb.inform.Settings;
import geniusweb.inform.Voting;
import geniusweb.inform.YourTurn;
import geniusweb.issuevalue.Bid;
import geniusweb.issuevalue.Domain;
import geniusweb.party.Capabilities;
import geniusweb.party.DefaultParty;
import geniusweb.profile.FullOrdering;
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
	
	

	private Map<PartyId, TFLinearAdditiveOpponentModel> opponentModelMap = new HashMap<>();
	public Group17_Main() {
	}

	public Group17_Main(Reporter reporter) {
		super(reporter); // for debugging
	}

	// Returns a list of my utility space and all opponent model utility spaces
	private List<LinearAdditive> getUtilitySpaces() {
		List<LinearAdditive> result = new ArrayList<>();

		result.add((LinearAdditive) profileint);

		for (TFLinearAdditiveOpponentModel opponentModel: opponentModelMap.values()) {
			result.add(opponentModel);
		}

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
				

				
				allBids = new AllBidsList(this.profileint.getProfile().getDomain());
				
				UtilitySpace utility = (UtilitySpace) this.profileint.getProfile();
				
				firstBids = bidsWithUtility.getBids(new Interval(maxRange.multiply(BigDecimal.valueOf(threshold)), maxRange));
				
				while (firstBids.size().compareTo(BigInteger.valueOf(firstBestBids)) == -1 && BigInteger.valueOf(firstBestBids).compareTo(allBids.size()) <= 0) {
					threshold -= 0.1;
					firstBids = bidsWithUtility.getBids(new Interval(maxRange.multiply(BigDecimal.valueOf(threshold)), maxRange));
				}
				
				int index = 0;
				for (Bid b : firstBids) {
					elements.add(new Element(index, utility.getUtility(b)));
					index++;
				}
				
				Collections.sort(elements);
				Collections.reverse(elements);

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
					TFLinearAdditiveOpponentModel opponentModel = opponentModelMap.get(actor);
					TFLinearAdditiveOpponentModel updatedOpponentModel = opponentModel.with(otheract, progress);
					opponentModelMap.put(actor, updatedOpponentModel);
				}
			} else if (info instanceof YourTurn) {
				makeOffer();
			} else if (info instanceof Finished) {
				getReporter().log(Level.INFO, "Final ourcome:" + info);
			} else if (info instanceof Voting) {
				
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
		Profile currentProfile = profileint.getProfile();
		UtilitySpace utility = (UtilitySpace) currentProfile;
		
		int round = ((ProgressRounds) progress).getCurrentRound();
		Bid bid = null;
		if (firstBids.size().compareTo(BigInteger.valueOf(round)) >= 0) {

			bid = firstBids.get(BigInteger.valueOf(elements.get(round).index));
			
		} else {
			// TODO implement with opponentmodel
			bid = firstBids.get(BigInteger.valueOf(elements.get(round).index));
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
		Object val = settings.getParameters().get("minPower");
		Integer minpower = (val instanceof Integer) ? (Integer) val : 2;
		val = settings.getParameters().get("maxPower");
		Integer maxpower = (val instanceof Integer) ? (Integer) val
				: Integer.MAX_VALUE;

		Set<Vote> votes = voting.getBids().stream().distinct()
				.filter(offer -> isGood(offer.getBid(), 0.6))
				.map(offer -> new Vote(me, offer.getBid(), minpower, maxpower))
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
