# Neural Augmented Simulator ++
Clone the repo and install the package: 

*  `git clone https://github.com/fgolemo/neural-augmented-simulator.git`
* `cd neural-augmented-simulator`
* `pip install -e .`

 Training ppo:

* `cd neural_augmented_simulator`
* `python experiments/ppo_train.py --approach 'goal-babbling' --variant 10 --seed 3 --env-name 'Nas-ErgoReacherAugmented-Headless-MultiGoal-Halfdisk-Long-v2'  --task='reacher'`

 Testing ppo:

* `python experiments/ppo_test.py --approach 'goal-babbling' --variant 10 --seed 3 --env-name 'Nas-ErgoReacherAugmented-Headless-MultiGoal-Halfdisk-Long-v2'  --task='reacher'`

New Changes:
* Added augmented simulator code for reacher
* Included reacher trained models
* Restructured the code
* Changed the names of environments