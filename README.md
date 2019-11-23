# Neural Augmented Simulator ++
Clone the repo and install the package: 

*  `git clone https://github.com/fgolemo/neural-augmented-simulator.git`
* `cd neural-augmented-simulator`
* `pip install -e .`

 Training ppo:

* `cd neural_augmented_simulator`
* `python experiments/ppo_train.py --approach 'testing' --variant 10 --seed 1 --env-name 'Nas-Pusher-3dof-Backlash01-v1'  --noise-type 'no-noise'`

 Testing ppo:

* `python experiments/ppo_test.py --approach 'testing' --variant 10 --seed 1 --env-name 'Nas-Pusher-3dof-Backlash01-v1'  --noise-type 'no-noise'`