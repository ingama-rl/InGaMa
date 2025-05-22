import numpy as np
from ingama_coordinate import load_model_parameters, initialize_render, run_n_episodes
from jaxmarl.viz.window import Window
import jax

def main(model_path, n_episodes, seed, deterministic):
    config, parameters = load_model_parameters(model_path)
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key, 2)
    env, env_state, render_data, render_images, render_window_kwargs, first_render = initialize_render(
        parameters, config, n_episodes, subkey, show=True
    )
    trajectories, cont_return, comm_return = run_n_episodes(
        env, env_state, render_data, n_episodes, render_images, key, deterministic=deterministic
    )
    print("mean cont return: ", cont_return)
    print("mean comm return: ", comm_return)
    render_window = Window(**render_window_kwargs)
    render_window.show(block=False)
    for jj, trajectory in enumerate(trajectories):
        render_window.show_img(np.array(first_render[jj]))
        for j, im in enumerate(trajectory):
            render_window.show_img(np.array(im))
    render_window.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--path", type=str)
    parser.add_argument("--deterministic", type=int)
    parser.add_argument("-n", type=int)
    args = parser.parse_args()
    # with jax.disable_jit():
    main(args.path, args.n, args.seed, args.deterministic > 0)



