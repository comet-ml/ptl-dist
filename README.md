# Running Distributed Training with Comet + Pytorch Lightning

This example is intended to be run in a multi-node setup. Please ensure all nodes are able to communicate with each other using the relevant hosts/ports.

The example provided here has been tested on a two node system, where each node has a single GPU.

## Setup

### Install dependencies

```shell
pip install -r requirements.txt
```

### Set Comet Credentials

Ensure that Comet credentials are set on each node.

```shell
export COMET_API_KEY=<Your API Key>
export COMET_PROJECT_NAME=<Your Project Name>
```

### Set Distributed Training Variables

On each node set the following environment variables

```shell
export MASTER_ADDR=<IP of your main node>
export MASTER_PORT=<Port for your main node>
```

These variables will let Lightning know which machine is the main node, and which machines are worker nodes.

## Run Training

Run the following command on the main/master node.

```shell
env NODE_RANK="0" python ptl_train.py
```

This command will create a new Experiment using the Pytorch Lightning `CometLogger`. You will see a URL with an Experiment ID displayed in the terminal after running this command.

```
COMET INFO: Experiment is live on comet.com https://www.comet.com/team-comet-ml/ptl-dist/41741323f2674b46b18a272f39b62b68
```

On your worker machine, run the following command

```shell
env NODE_RANK="1" python ptl_train.py --experiment_id <Experiment ID created on the main node>
```

Based on the URL shown in this example, we would run the following command on the worker node

```
env NODE_RANK="1" python ptl_train.py \
--experiment_id 41741323f2674b46b18a272f39b62b68
```

This will create an Experiment object on the worker node to log system metrics without using the Lightning's `CometLogger`.

Lightning only allows rank 0 nodes to create logger objects. Metrics, and hyperparameters will be captured from the rank 0 node only.

The additional Experiment object created on the workers will only capture system level metrics (CPU/GPU usage, Memory Usage etc).

## Example Project

[Here is an example project](https://www.comet.com/team-comet-ml/ptl-dist?shareable=GJog5KxIv7mGGlxztyKyUlQ5A) with the results of a distributed run
