# DeepHoldem (deeper-stacker)

This is an implementation of [DeepStack](https://www.deepstack.ai/s/DeepStack.pdf)
for No Limit Texas Hold'em, extended from [DeepStack-Leduc](https://github.com/lifrordi/DeepStack-Leduc).

## Requisites

You will need following requisites:
* Ubuntu 16.04 or a Linux OS with Docker (and use a Docker image with Ubuntu 16.04).
* NVIDIA GPU with a dedicated graphics card with more than 8GB video RAM (NVIDIA GeForce GTX 1080 or NVIDIA Tesla P100 recommended).

## Setup

All configuration step are in `torch/prepare.sh` file.

Running any of the DeepHoldem code requires [Lua](https://www.lua.org/) and [torch](http://torch.ch/). Please install torch with lua version 5.2 instead of LuaJIT. Torch is only officially supported for \*NIX based systems (i.e. Linux and MacOS X).

Connecting DeepHoldem to a server or running DeepHoldem on a server will require the [luasocket](http://w3.impa.br/~diego/software/luasocket/)
package. This can be installed with [luarocks](https://luarocks.org/) (which is
installed as part of the standard torch distribution) using the command
`luarocks install luasocket`. Visualising the trees produced by DeepHoldem
requires the [graphviz](http://graphviz.org/) package, which can be installed
with `luarocks install graphviz`. Running the code on the GPU requires
[cutorch](https://github.com/torch/cutorch) which can be installed with
`luarocks install cutorch`.

The HandRanks file was too big for Github, so you will need to unzip it: `cd Source/Game/Evaluation && unzip HandRanks.zip`

### scatterAdd

When you try to run DeepHoldem, you will eventually run into a problem where `scatterAdd` is not defined.
Torch7 actually includes a C++ implementation of scatterAdd but for whatever reason, doesn't include a lua
wrapper for it.

I've included `TensorMath.lua` files in the torch folder of this repository that include the wrapper functions for both CPU and GPU. Copy them to their corresponding torch installation folders.

Now, from your torch installation directory, run:

```
$ ./clean.sh
$ TORCH_LUA_VERSION=LUA52 ./install.sh
```

and you should be good to go.

## Performance

This implementation was tested against Slumbot 2017, the only publicly playable bot as of June 2018. The action abstraction used was half pot, pot and all in for first action, pot and all in for second action onwards. It achieved a baseline winrate of **42bb/100** after 2616 hands (equivalent to ~5232 duplicate hands). Notably, it achieved this playing inside of Slumbot's action abstraction space.

![](Data/Images/slumbot_stats.png)

A comparison of preflop ranges was also done against [DeepStack's hand history](https://www.deepstack.ai/s/DeepStack_vs_IFP_pros.zip), showing similar results.

| Action                  | DeepStack                           | DeepHoldem                    |
| ---                     | ---                                 | ---                           |
| Open fold               |![](Data/Images/deepstack_folds.png) | ![](Data/Images/my_folds.png) |
| Open pot                |![](Data/Images/deepstack_pots.png)  | ![](Data/Images/my_pots.png)  |
| 3bet pot after pot open |![](Data/Images/deepstack_3bets.png) | ![](Data/Images/my_3bets.png) |

Average thinking speed comparison:

| Street  | DeepStack Thinking Speed (s) | DeepHoldem Thinking Speed (s) |
| ---     | ---                          | ---                           |
| Preflop | 0.2                          | 2.69                          |
| Flop    | 5.9                          | 12.42                         |
| Turn    | 5.5                          | 7.57                          |
| River   | 2.1                          | 3.33                          |

DeepHoldem using a NVIDIA Tesla P100.
DeepStack using a NVIDIA GeForce GTX 1080.

Training details:

| Network             | # samples | # poker situations | Validation huber loss | Epoch |
| ---                 | ---       | ---                | ---                   | ---   |
| River network       | 100,000   | 1,000,000          | 0.0415                | 54    |
| Turn network        | 100,000   | 1,000,000          | 0.045                 | 52    |
| Flop network        | 100,000   | 1,000,000          | 0.013                 | 52    |
| Preflop aux network | 100,000   | 1,000,000          | 0.0017                | 67    |

Training data and Validation Huber Loss comparison:

| Network             | DeepStack # ps | DeepStack vhb | DeepHoldem # ps | DeepHoldem vhb |
| ---                 | ---            | ---           | ---             | ---            |
| River network       | (no used)      | (no used)     | 1,000,000       | 0.0415         |
| Turn network        | 10,000,000     | 0.026         | 1,000,000       | 0.045          |
| Flop network        |  1,000,000     | 0.034         | 1,000,000       | 0.013          |
| Preflop aux network | 10,000,000     | 0.000055      | 1,000,000       | 0.0017         |

Note that:
* ps: poker situation
* vhb: validation huber loss

If you need to extract Neuronal network information, just type `th` and:
```
$ torch.load('final_cpu.info')
{
  gpu : false
  epoch : 54
  valid_loss : 0.041533345632636
}
```

## Samples Math

A training sample is composed of 2 files:
 * inputs.
 * targets.

By default, each training sample contains 10 poker situations (params.gen_batch_size).

If you want to generate 1,000,000 poker situations to train your network, you will need to configure following parameter (Source/Settings/arguments.lua):

```
params.train_data_count = 1000000
```

Just to make sure that you understand it:
 * You will need to have 200,000 files.
 * You will need to have 100,000 training samples.
 * You will need to have 1,000,000 poker situations.

```
200,000 files = 100,000 training samples = 1,000,000 poker situations
```

## Training your models

DeepHoldem is not a single neuronal network. DeepHoldem is composed of four networks:
 * River network
 * Turn network
 * Flop network
 * Preflop aux network

Remember that to generate training data for a network, you will need to have the precedent network trained.

The workflow should be:

```
River > Turn > Flop > Preflop aux
```

Preflop aux is already included, so you won't need to train it if you want.

In the future, if you want to train your neuronal network with more poker situations, you will have to generate again preflop, flop and turn samples. This is because turn poker situations depends on river model (and the same for flop because depends on turn model...).

## Abstraction actions

DeepStack uses following actions to generate poker situations:
 * Fold.
 * Call/Check.
 * Bet Half Pot (0.5p).
 * Bet Pot (1p).
 * All-in.

DeepHoldem uses following actions:
 * Fold.
 * Call/Check.
 * Bet Pot (1p).
 * All-in.

You can add more actions in parameter configuration file:

```
params.bet_sizing = {{1},{1},{1}}
```

You can't remove following actions:
 * Fold.
 * Call/Check.
 * All-in.

## Creating your own river, turn and flop models

Other than the preflop auxiliary network, the counterfactual value networks are not included as part of this release, you will need to generate them yourself. The model generation pipeline is a bit different from the Leduc-Holdem implementation in that the data generated is saved to disk as raw solutions rather than bucketed solutions. This makes it easier to experiment with different bucketing methods.

Here's a step by step guide to creating models:

1. `cd Source && th DataGeneration/main_data_generation.lua 4`.
2. Wait for enough data to be generated.
3. `th Training/raw_converter.lua 4`.
4. `th Training/main_train.lua 4`.
5. `th Training/pickup_best_model.lua 4`.
6. Repeat steps 1-5 for turn and flop by replacing `4` with `3` or `2`.

By default, data generation and model training uses GPU. If you want to disable it, just modify `Settings/arguments.lua` to `params.gpu = false`.

## Creating your own pre-flop model

Here's a step by step guide to creating pre-flop model:

1. `cd Source && th DataGeneration/main_aux_data_generation.lua 1`
2. Wait for enough data to be generated.
3. `th Training/raw_converter.lua 1`
4. `th Training/main_train.lua 1`
5. `th Training/pickup_best_model.lua 1`.

## Playing against DeepHoldem

`Player/manual_player.lua` is supplied so you can play against DeepHoldem for preflop situations. If you want
DeepHoldem to work for flop, turn and river, you will need to create your own models.

1. `cd ACPCServer && make`
2. `./dealer testMatch holdem.nolimit.2p.reverse_blinds.game 1000 0 Alice Bob`.
3. 2 ports will be output, note them down as port1 and port2.
4. Open a second terminal and `cd Source && th Player/manual_player.lua <port1>`.
5. Open a third terminal and `cd Source && th Player/deepstack.lua <port2>`.
6. Once the deepstack player is done loading, you can play against it using manual_player terminal. 

Possible actions:
* `f` = fold.
* `c` = check/call
* `450` = raise my total pot commitment to 450 chips.

## Playing vs Slum Bot

Slum bot is available to play online in this page: http://www.slumbot.com/

There is a script in Python that you can use to play versus Slum Bot.

1. Install Python 2.7
2. Install pip
3. Install Selenium using pip `pip install -U selenium`
3. Start the deepstack server using command `cd Source && th Player/deepstack_server.lua 16177`
4. `cd Source && python Player/slumbot_player.py localhost 16177`
5. Enjoy bot-battle!

## Testing deeper-stacker source code

There are a few tests defined:
1. Install `luarocks install busted`
2. `cd Source && busted --no-auto-insulate`

## Training details

Source: https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/

The batch size (params.train_batch_size) is a number of samples (poker situations) processed before the model is updated.

The number of epochs (params.epoch_count) is the number of complete passes through the training dataset.

The batch size (params.train_batch_size) must be more than or equal to one and less than or equal to the number of samples (poker situations) in the training dataset.

The number of epochs (params.epoch_count) can be set to an integer value between one and infinity.

They are both integer values and they are both hyperparameters for the learning algorithm, e.g. parameters for the learning process, not internal model parameters found by the learning process. You must specify the batch size and number of epochs for a learning algorithm.

There are no magic rules for how to configure these parameters. You must try different values and see what works best for your problem.

### Training example

Finally, let's make this concrete with an example taken from DeepStack paper (Suplement).

Assume you want to train a Turn network and you have a dataset with ten million samples (rows of data/poker situations) and you choose a batch size of 1,000 and 350 epochs.

This means that the dataset will be divided into 10,000 batches, each with 1,000 samples. The model weights will be updated after each batch of 1,000 samples (poker situations).

This also means that one epoch will involve 10,000 batches or 10,000 updates to the model.

With 350 epochs, the model will be exposed to or pass through the whole dataset 350 times. That is a total of 350,000 batches during the entire training process.

## Differences from the original paper

- A river model was used instead of solving directly from the turn.
- Different neural net architecture.
  - Batch normalization layers were added in between hidden layers because they were found to improve huber loss.
  - Only 3 hidden layers were used. Additional layers didn't improve huber loss, in agreement with the paper.
- Preflop solving was done with auxiliary network only, whereas paper used 20 iterations of flop network.
  - Because of this, the cfvs for a given flop must be calculated after seeing it by solving the preflop again with the current flop in mind.
- During re-solving, the opponent ranges were not warm started.

## Future work

- Warm start opponent ranges for re-solving.
- Speed up flop solving (use flop network during preflop solving?).
- Support LuaJIT.
- C++ implementation?

## References

DeepStack Paper & Supplements: https://www.deepstack.ai/downloads
