from tensorforce import Agent, Environment

from tensorforce.execution import Runner

import DrawingGridEnv


# custom environment
environment = Environment.create(DrawingGridEnv.SimpleGrid, max_episode_timesteps=400)

# Instantiate a Tensorforce agent
agent = Agent.create(
    agent='ppo', environment=environment, batch_size=10, learning_rate=1e-3
)

# Train for 300 episodes
for _ in range(300):

    # Initialize episode
    states = environment.reset()
    terminal = False

    while not terminal:
        # Episode timestep
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

agent.save("./", format='numpy')
agent.close()
environment.close()