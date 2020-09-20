from tensorforce import Agent, Environment

from tensorforce.execution import Runner

import DrawingGridEnv


# custom environment
environment = Environment.create(DrawingGridEnv.SimpleGrid, max_episode_timesteps=500)

# Instantiate a Tensorforce agent
agent = Agent.create(
    agent='ppo', environment=environment, batch_size=10, learning_rate=1e-3, summarizer=dict(
        directory='results/summaries',
        # list of labels, or 'all'
        labels=['all']
    ),
)
counter = 0
# Train for 300 episodes
for _ in range(800):

    # Initialize episode
    states = environment.reset()
    terminal = False

    while not terminal:
        # print(f"timestep {counter}")
        # Episode timestep
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)
        counter+=1

agent.save("./", format='numpy')
agent.close()
environment.close()