So these past months I have been working on a novel learning regime and multimodal agents, to learn to play leaderboards at online .io games, straight from pixels in the browser, with no synthetic environment training, no 'ideal' data pre-prep, and cleansing, and still learning online "policies".

## Bit of diving in:
The agent is to be trained on https://smashkarts.io/, you can visit the link and notice some key drawbacks:
- Online lobby multiplayer game, so obviously very difficult to curate a healthy/dense dataset for any kind of learning or reward policies. 
- Also there are separate maps(requires generalization) and powers that are used in different manners, a player has to maneuver in ways to use powers effectively, unique to each power. No labels were passed, specifically curated self-supervised bootstrapped rewards were formulated and a novel self-play regime was followed by a novel hierarchical graph neural net(with N-layers of imageNet trained backbone)
- There is a complete physics sim going on in the game, exploiting which can lead to some cool kills(can be seen performed by the agent towards the end of the attached mp4).

*A game or two in online leaderboards is enough to understand the challenges present in learning such a task in such an environment, including:*
- *sparse inputs, extremely sparse environment*
- *discerning objects from pixels, and their relations in the environment*
- *maximization of kills, minimization of self-deaths, strategy formulation, keyboard mapping control from some latent understanding of the strategy*
- *current techniques being resource intensive and inapplicable in this environment.*

## Training brief (model hereafter refers to neural net driving our agent):
The model's structure is not static, it is a learnable/growable hierarchical directed cyclic graph.
- Model takes as input cropped(3600x2000) raw pixels at 30fps, b/w 1-channel images and controls keyboard strikes, W-A-S-D, and space bar tap/hold.
- Model takes as input a sequence of such images for a forward pass, and generates possible action/fire/move distributions.
- The structure does not use back prop, the latent variables are updated through a novel formulation, that seeds all latent variables as evolved forms of similar 'daughter' graphs.
- A single forward pass updates coefficients partially, converging till a point $\hat{P}$<sup>*</sup> where daughter graphs of each mother node reach a quasi-entropic function state while waiting for the next sequence of 30fps dense images to gather.
- A single forward pass with N sequenced images, took N*22ms on a single Nvidia A100
- The delayed learning('pondering', 'secondary field learning stage') takes average-case time(in ms) of $\frac{Np\(e-1\)}{\pi}$(log<sub>p</sub>d). 
  - *p* is a parameter that measures the growth of the mother+daughter graph in multiple dimensions(neuronal, inter-neuronal, intra-neuronal)). A model with p=5 is equivalent in size to a 500M parameter transformer encoder.
  - *d* is the flattened-out dimensionality of input, here 3600x2000 = 7.2M
  - *N* is the number of images in a single input sequence, here 30
  - For ex: 
   - Choosing N<=30 is a pretty neat choice for p<=5, putting values above, limiting values tends to ~660+804=1.46s (primary and secondary field learning times respectively)
   - In this case, at t timestep, the model has a bandwidth of 30 images per 1.46s, where after 1.00s, new 30 images come in again for t+1 timestep(30fps).
   - Both primary and secondary field learning stages happen on 2 separate GPUs, with mild waiting stages.
- The model trained itself from scratch(apart from an n-layer backbone on imageNet), for only 14 hours, playing live in the browser. No synthetic environments to provide dense distribution to learn from.
- Each game is 180secs generally, and some more seconds are used while automated clicks browse through lobbies to find matches online.
- Roughly the agent played no more than 150 games(152 games), where it had to learn:
  - to discern map terrain, players, itself, different powers, and different projectiles from raw pixels
  - Plan and learn to strategize and climb the leaderboard.
  -  Amidst such sparse inputs with no explicit goal given, only bootstrapped self-supervised reward policies are meta-learned inside each node of the agent's graph network.

## Results:
A stitched mp4 can be found [here's_a_gdrive_link_for_ya](https://drive.google.com/file/d/1ar2hUmIcORxy9vghKMptV9q4S6yjXgQi/view?usp=sharing), it is stitched 30fps at 60fps so it is choppy(basically every second frame is missing). It is exactly what the agent "sees", You can also see online ads rolling in the video as that was what the model was seeing at one point while learning inside the browser.

*PS - On the bottom left you can see a basic classifier at work(that the agent inherently learns). It tells when the agent(bot) is active/playing, and when I or a script is clicking to change levels/lobbies, hence bot is inactive in those stages(no outputs to the keyboard).*
*PPS - The above behavior was not learned explicitly, it was learned inherently in the agent itself by pushing some latent priors towards orthogonal Tr(input, output) transformed manifolds embedded in a single space.*

*$\hat{P}$ is called the Primary Field Lower Bound of the 'primary learning stage'
