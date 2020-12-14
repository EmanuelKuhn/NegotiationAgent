# NegotiationAgent

## How to run

1. Install the GeniusWeb negotiation environment, see: https://tracinsy.ewi.tudelft.nl/pubtrac/GeniusWeb#Installation
2. Run `mvn package` in the root directory, to build a jar file of the agent
3. Copy the built jar file from the target directory to the GeniusWeb parties repository folder, see the [GeniusWeb documentation](https://tracinsy.ewi.tudelft.nl/pubtrac/GeniusWebPartiesServer#AddingorchangingaParty) for details

## Features

* Tensorflow model to infer the opponent's Linear Additive Value function
* Bids are randomly sampled based on a minimum utility, biased towards bids that are preferred according to the inferred opponent's value function
* In the first few rounds, before the opponent models are trained, high value bids are proposed
