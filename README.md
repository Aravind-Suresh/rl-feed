# rl-feed
Implementation of a Q-learning based RL agent ( Fabber ) which learns to feed on snack in a grid world.

## Trial
First clone the repository. Then, to train the RL agent

```
$ python main.py --learn
```
To save the model, add ```--output /path/to/output/directory``` to the command like,

```
$ python main.py --learn --output models
```

To test the RL agent, add ```--test``` to the command. To load a pretrained model,
mention its path as ```--input /path/to/learned/model``` like,

```
$ python main.py --test --input models/model.pkl
```

## Dependencies
* Numpy ( Used v1.8.2 )
* OpenCV ( Used 3.0.0 ); Just for visualisation, can be removed

## Contribute
If you could think of any improvements to the project, please feel free to make a **pull request**.
