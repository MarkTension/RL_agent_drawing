from tensorforce import Agent, Environment

from tensorforce.execution import Runner

import DrawingGridEnv

# dict(type='conv2d', window=3, stride=2, size=16, activation='elu')

# custom environment
environment = Environment.create(DrawingGridEnv.SimpleGrid, max_episode_timesteps=500)
# dict(type='conv2d', window=3, stride=2, size=16, activation='elu'),

# Instantiate a Tensorforce agent
agent = Agent.create(
    agent='ppo',
    environment=environment,
    network=[
        dict(type='conv2d', window=5, stride=3, size=8, activation='elu'),
        dict(type='flatten'),
        dict(type='dense', size=2),
        dict(type='flatten', name="out"),
    ],
    batch_size=10,
    learning_rate=1e-3,
    summarizer=dict(
        directory='results/summaries',
        # list of labels, or 'all'
        labels=['entropy', 'kl-divergence', 'loss', 'reward', 'update-norm']
    )
)
counter = 0
# Train for 300 episodes
for _ in range(10000):
    if counter % 1000 == 0:
        print(f"episode {counter}")
    # Initialize episode
    states = environment.reset()
    terminal = False

    while not terminal:
        # print(f"timestep {counter}")
        # Episode timestep
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)
    # print(f'rewaward = {agent.reward_spec}')
    counter+=1
    DrawingGridEnv.params.episode += 1

agent.save("./", format='numpy')
agent.close()
environment.close()