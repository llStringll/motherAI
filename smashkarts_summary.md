So these past months I have been working on a novel learning regime and multimodal agents, to learn to play leaderboards at online .io games, straight from pixels in the browser, with no synthetic environment training, no 'ideal' data pre-prep, and cleansing, and still learning online "policies".

## Bit of diving in:
The agent is to be trained on https://smashkarts.io/, you can visit the link and notice some key drawbacks:
- Online lobby multiplayer game, so obviously very difficult to curate a healthy/dense dataset for any kind of learning or reward policies. 
- Also there are separate maps(requires generalization) and powers that are used in different manners, a player has to maneuver in ways to use powers effectively, unique to each power. No labels were passed, specifically curated self-supervised bootstrapped rewards were formulated and a novel self-play regime was followed by a novel hierarchical graph neural net(with N-layers of imageNet trained backbone)
- There is a complete physics sim going on in the game, exploting which can lead to some cool kills(can be seen performed by the agent towards the end of the attached mp4).

*A game or two in online leaderboards is enough to understand the challenges present in learning such a task in such an environment, including:*
- *sparse inputs, extremely sparse environment*
- *discerning objects from pixels, and their relations in the environment*
- *maximization of kills, minimization of self-deaths, strategy formulation, keyboard mapping control from some latent understanding of the strategy*
- *current techniques being resource intensive and inapplicable in this environment.*

## Training brief (model hereafter refers to neural net driving our agent):
- Model's structure is not static, it is a learnable/growable hierarchical directed cyclic graph.
- Model takes as input cropped raw pixels at 30fps, b/w 1-channel images and controls keyboard strikes, W-A-S-D, and space bar tap/hold.
- Model takes as input a sequence of such images for a forward pass, and generates possible action/fire/move distributions.
- The structure does not use back prop, the latent variables are updated through a novel formulation, that seeds all latent variables as evolved forms of similar 'sister' graphs.
- A single forward pass updates coefficients partially, and rest reach an entropic state while waiting for the next sequence of 30fps dense images to gather.
- A single forward pass with a N sequenced images, took N*22ms on a single nvidia A100
- The delayed learning('pondering') takes average-case time of 223*ln(p) ms. (p is a paramter that measures growth of the graph in multiple dimensions(neuronal, inter-neuronal, intra-neuronal), ln() is the natural log)
- For ex: choosing N=30 is a pretty neat choice for p<5, you can plug numbers above and see why.(p=5 is equivalent in size to a 200M paramter transformer). One can easily see that such a model is processing its entire learning for given inouts X in under a second, at 30fps input resolution, i.e. pretty real-time learning.
- The model trained itself from scratch(apart from an N-layer backbone on imageNet), for only 14hours, playing live in the browser. No synthetic environments to provide dense distribution to learn from.
- Each game is 180secs generally, and some more seconds are used while automated clicks browse through lobbies to find matches online.
- Roughly the agent played no more than 150 games, where it had to learn:
  - to discern map terrain, players, itself, different powers, different projectiles from raw pixels
  - Plan and learn to strategize and climb the leaderboard.
  -  Amidst such sparse inputs with no explicit goal given, only bootstrapped self-supervised reward policies which are meta-learned inside each node of the agent's graph network.

## Results:
A stitched mp4 can be found [here's_a_gdrive_link_for_ya](https://drive.google.com/file/d/1ar2hUmIcORxy9vghKMptV9q4S6yjXgQi/view?usp=sharing), it is stitched 30fps at 60fps so it is choppy(basically every second frame is missing). It is exactly what the agent "sees", you can also see online ads rolling in the video as that was what the model was seeing at one point while learning inside the browser.

*PS - on bottom left you can see a basic classifier at work(that the agent inherently learns). It tells when the agent(bot) is active/playing, and when I or a script is clicking to change levels/lobbies, hence bot is inactive in those stages(no outputs to keyboard).*
*PPS - The above behaviour was not learned explicitly, it was learned inherently in the agent itself by pushing some latent priors of it towards orthogonal input/output manifolds embedded in a single space.*
