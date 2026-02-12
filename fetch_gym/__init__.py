import gym
from gym.envs.registration import register

env_name = 'FetchReachAL-v0'

registry = gym.envs.registry

# Handle both old gym (with .env_specs) and new gym (dict-like registry)
if hasattr(registry, "env_specs"):
    # Old API: registry.env_specs is a dict-like
    if env_name in registry.env_specs:
        del registry.env_specs[env_name]
else:
    # New API: registry itself is the dict-like
    if env_name in registry:
        del registry[env_name]

register(
    id=env_name,
    entry_point='fetch_gym.envs:' + env_name[:-3],
)
