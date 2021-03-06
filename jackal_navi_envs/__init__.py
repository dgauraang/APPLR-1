from gym.envs.registration import register
from jackal_navi_envs.gazebo_simulation import GazeboSimulation
from jackal_navi_envs.navigation_stack import NavigationStack
from jackal_navi_envs import jackal_env_wrapper
from jackal_navi_envs.jackal_env_continuous import range_dict

register(
    id='jackal_discrete-v0',
    entry_point='jackal_navi_envs.jackal_env_discrete:JackalEnvDiscrete',
)

register(
    id='jackal_continuous-v0',
    entry_point='jackal_navi_envs.jackal_env_continuous:JackalEnvContinuous',
)
