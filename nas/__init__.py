from gym import register

register(
    id='Nas-Pusher-3dof-Vanilla-v1',
    entry_point='nas.envs:PusherVanillaEnv',
    max_episode_steps=100,
    reward_threshold=0,
    kwargs={})

register(
    id='Nas-Pusher-3dof-Backlash01-v1',
    entry_point='nas.envs:PusherVanillaEnv',
    max_episode_steps=100,
    reward_threshold=0,
    kwargs={
        "backlash": "01"
    })

register(
    id='Nas-Pusher-3dof-Backlash01-v2',
    entry_point='nas.envs:PusherVanillaEnv',
    max_episode_steps=100,
    reward_threshold=0,
    kwargs={
        "backlash": "01",
        "backlash_version": 2
    })

