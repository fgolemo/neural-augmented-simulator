# Density Estimator Model


## To train the model
```
python density_estimator.py
```

## To evaluate the model
```
python evaluate_policy_space.py
```

## Results

The results have been evaluated using the following trained models:
`ppo_ErgoReacherAugmented-Headless-MultiGoal-Halfdisk-Long-v2_1_goal-babbling_SEEDX`

Following are the results per seed :
- goal_1, seed_1
```
------------------------------
State based on probabilities
Mean:  0.36497828
Std:  0.16130003
Median:  0.30837628
------------------------------
------------------------------
State based on log probabilities
Mean:  0.7338402
Std:  0.16130003
Median:  0.71367365
------------------------------
```

- goal_1, seed_2
```
------------------------------
State based on probabilities
Mean:  0.4591099
Std:  0.16681255
Median:  0.44397837
------------------------------
------------------------------
State based on log probabilities
Mean:  0.84919393
Std:  0.16681255
Median:  0.8565335
------------------------------
```

- goal_1, seed_3
```
------------------------------
State based on probabilities
Mean:  0.69677716
Std:  0.2146249
Median:  0.71583724
------------------------------
------------------------------
State based on log probabilities
Mean:  0.83516127
Std:  0.2146249
Median:  0.864982
------------------------------

```

- goal_1, average across points collected by all seed models
```
------------------------------
State based on probabilities
Mean:  0.44629988
Std:  0.16845866
Median:  0.43259227
------------------------------
------------------------------
State based on log probabilities
Mean:  0.840951
Std:  0.16845866
Median:  0.84940326
------------------------------
```

- goal_2, seed_1
```
------------------------------
State based on probabilities
Mean:  0.5208745
Std:  0.21645766
Median:  0.52205336
------------------------------
------------------------------
State based on log probabilities
Mean:  0.82718146
Std:  0.21645766
Median:  0.85301983
------------------------------
```

- goal_2, seed_2
```
------------------------------
State based on probabilities
Mean:  0.58521515
Std:  0.2657814
Median:  0.58240914
------------------------------
------------------------------
State based on log probabilities
Mean:  0.8068505
Std:  0.2657814
Median:  0.84164906
------------------------------

```

- goal_2, seed_3
```
------------------------------
State based on probabilities
Mean:  0.560998
Std:  0.26034784
Median:  0.5729815
------------------------------
------------------------------
State based on log probabilities
Mean:  0.871466
Std:  0.26034784
Median:  0.90399003
------------------------------
```


- goal_2, average across points collected by all seed models
```
------------------------------
State based on probabilities
Mean:  0.53730494
Std:  0.24147533
Median:  0.52587765
------------------------------
------------------------------
State based on log probabilities
Mean:  0.9004066
Std:  0.24147533
Median:  0.9150627
------------------------------
```


- goal_10, seed_1
```
------------------------------
State based on probabilities
Mean:  0.4298055
Std:  0.24546608
Median:  0.32307172
------------------------------
------------------------------
State based on log probabilities
Mean:  0.6488242
Std:  0.24546608
Median:  0.5952127
------------------------------
```
- goal_10, seed_2
```
------------------------------
State based on probabilities
Mean:  0.4145616
Std:  0.25234944
Median:  0.3293498
------------------------------
------------------------------
State based on log probabilities
Mean:  0.6360527
Std:  0.25234944
Median:  0.60823154
------------------------------
```

- goal_10, seed_3
```
------------------------------
State based on probabilities
Mean:  0.3329864
Std:  0.16757837
Median:  0.32552642
------------------------------
------------------------------
State based on log probabilities
Mean:  0.5674193
Std:  0.16757837
Median:  0.5922605
------------------------------

```


- goal_10, average across points collected by all seed models
```
------------------------------
State based on probabilities
Mean:  0.39333272
Std:  0.22370377
Median:  0.33000696
------------------------------
------------------------------
State based on log probabilities
Mean:  0.61929834
Std:  0.22370377
Median:  0.60157704
------------------------------
```

- motor_1, seed_1
```
------------------------------
State based on probabilities
Mean:  0.35243282
Std:  0.19305143
Median:  0.32261533
------------------------------
------------------------------
State based on log probabilities
Mean:  0.63487923
Std:  0.19305143
Median:  0.65237296
------------------------------
```
- motor_1, seed_2
```
------------------------------
State based on probabilities
Mean:  0.3056588
Std:  0.19411077
Median:  0.2360453
------------------------------
------------------------------
State based on log probabilities
Mean:  0.5852926
Std:  0.19411077
Median:  0.5600414
------------------------------
```

- motor_1, seed_3
```
------------------------------
State based on probabilities
Mean:  0.43698153
Std:  0.22024086
Median:  0.45916212
------------------------------
------------------------------
State based on log probabilities
Mean:  0.7516232
Std:  0.22024086
Median:  0.8096886
------------------------------
```

- motor_1, average across points collected by all seed models

```
------------------------------
State based on probabilities
Mean:  0.3046052
Std:  0.19656284
Median:  0.2596686
------------------------------
------------------------------
State based on log probabilities
Mean:  0.6029147
Std:  0.19656284
Median:  0.61195314
------------------------------
```

- motor_2, seed_1
```
------------------------------
State based on probabilities
Mean:  0.40033457
Std:  0.20391731
Median:  0.3833567
------------------------------
------------------------------
State based on log probabilities
Mean:  0.6425891
Std:  0.20391731
Median:  0.6657437
------------------------------
```

- motor_2, seed_2
```
------------------------------
State based on probabilities
Mean:  0.5410984
Std:  0.19462773
Median:  0.5409332
------------------------------
------------------------------
State based on log probabilities
Mean:  0.80694187
Std:  0.19462773
Median:  0.8283788
------------------------------
```

- motor_2, seed_3
```
------------------------------
State based on probabilities
Mean:  0.45641798
Std:  0.092095025
Median:  0.4522641
------------------------------
------------------------------
State based on log probabilities
Mean:  0.7258316
Std:  0.092095025
Median:  0.72925234
------------------------------
```


- motor_2, average across points collected by all seed models
```
------------------------------
State based on probabilities
Mean:  0.50442946
Std:  0.16369754
Median:  0.49524134
------------------------------
------------------------------
State based on log probabilities
Mean:  0.81662273
Std:  0.16369754
Median:  0.8265799
------------------------------
```

- motor_10, seed_1
```
------------------------------
State based on probabilities
Mean:  0.47374552
Std:  0.2403321
Median:  0.4473767
------------------------------
------------------------------
State based on log probabilities
Mean:  0.8179735
Std:  0.2403321
Median:  0.83863753
------------------------------
```

- motor_10, seed_2
```
------------------------------
State based on probabilities
Mean:  0.4765349
Std:  0.228499
Median:  0.47359312
------------------------------
------------------------------
State based on log probabilities
Mean:  0.7778556
Std:  0.228499
Median:  0.81026006
------------------------------
```

- motor_10, seed_3
```
------------------------------
State based on probabilities
Mean:  0.33866757
Std:  0.29285938
Median:  0.20551166
------------------------------
------------------------------
State based on log probabilities
Mean:  0.6181864
Std:  0.29285938
Median:  0.59904885
------------------------------
```

- motor_10, average across points collected by all seed models
```
------------------------------
State based on probabilities
Mean:  0.4201263
Std:  0.26429686
Median:  0.39141497
------------------------------
------------------------------
State based on log probabilities
Mean:  0.72492784
Std:  0.26429686
Median:  0.7670014
------------------------------
```