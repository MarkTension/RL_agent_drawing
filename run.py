from tensorforce import Agent, Environment
import DrawingGridEnv


load = False

# custom environment
environment = Environment.create(DrawingGridEnv.SimpleGrid, max_episode_timesteps=500)

if load:
    Agent.load(directory="./", filename="agent", format="numpy", environment=environment)
else:
    # Instantiate a Tensorforce agent
    agent = Agent.create(
        agent='ppo',
        environment=environment,
        network=[
            dict(type='conv2d', window=5, stride=3, size=8, activation='elu'),
            dict(type='flatten'),
            dict(type='dense', size=16),
            dict(type='flatten', name="out"),
        ],
        batch_size=10,
        learning_rate=1e-3,
        summarizer=dict(
            directory='results/summaries',
            labels=['entropy', 'kl-divergence', 'loss', 'reward', 'update-norm']
        )
    )

counter = 0

for _ in range(1000):
    if counter % 200 == 0:
        agent.save(f"./results/checkpoints/agent{counter}", format='numpy')
        print(f"episode {counter} and saved checkpoint")
    # Initialize episode
    states = environment.reset()
    terminal = False

    while not terminal:
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)
    counter+=1
    DrawingGridEnv.params.episode += 1


agent.save("./", format='numpy')
agent.close()
environment.close()