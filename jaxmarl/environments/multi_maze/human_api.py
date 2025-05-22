from jaxmarl.environments.multi_maze.multi_maze_v11 import WAIT, LEFT, RIGHT, UP, DOWN, MultiMazeLogWrapper
from jaxmarl import make
import jax
import jax.numpy as jnp
from jaxmarl.viz.window import Window

with jax.disable_jit():
    width = 7
    env = MultiMazeLogWrapper(make('multi_maze', num_agents=4, max_episode_len=50, map='Quadruple Trouble', difficulty='medium'), replace_info=True)
    key = jax.random.PRNGKey(0)
    key, key_act, key_act2 = jax.random.split(key, 3)
    obs, state = env.reset(key)

    _curr_view, render_images = env.render(state)
    height = (env.WINDOW_SIZE[1] / env.WINDOW_SIZE[0]) * width
    render_window = Window('Multi-Maze ', figsize=(width, height))
    render_window.show(block=False)
    render_window.show_img(_curr_view)

    comm_act = env.action_space('red_player_comm').sample(key_act)
    cont_act = env.action_space('red_player').sample(key_act2)
    comm_actions = {agent: jnp.ones_like(comm_act) for agent in env.comm_agents}
    for k, v in comm_actions.items():
        comm_actions[k] = v.at[6:].set(0)
    cont_actions = {agent: jnp.ones_like(cont_act) for agent in env.cont_agents}
    actions = {}
    while True:
        key, key_reset, key_act, key_step = jax.random.split(key, 4)
        _curr_view, _ = env.render(state, render_images=render_images)
        render_window.show_img(_curr_view)
        contin = False
        brk = False
        for k, v in cont_actions.items():
            val = input("Choose action for " + k + ":")
            if val == 'W' or val == 'w':
                actions[k] = WAIT * v
            elif val == 'L' or val == 'l':
                actions[k] = LEFT * v
            elif val == 'R' or val == 'r':
                actions[k] = RIGHT * v
            elif val == 'U' or val == 'u':
                actions[k] = UP * v
            elif val == 'D' or val == 'd':
                actions[k] = DOWN * v
            elif val == 'RESET' or val == 'reset':
                obs, state = env.reset(key_reset)
                contin = True
                break
            elif val == 'STOP' or val == 'stop' or val == 'EXIT' or val == 'exit':
                brk = True
                break
            elif val == '':
                continue
            else:
                actions[k] = WAIT * v

        if contin:
            continue
        if brk:
            break
        actions.update(comm_actions)
        obs, state, reward, done, infos = env.step(key_step, state, actions)

        print("reward:", reward["red_player"])


