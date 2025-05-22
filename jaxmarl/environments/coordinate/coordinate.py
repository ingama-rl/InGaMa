import jax
import jax.numpy as jnp
import numpy as np
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.wrappers.baselines import JaxMARLWrapper
from jaxmarl.environments.spaces import Discrete, MultiDiscrete, Box
import chex
from flax import struct
from functools import partial
from typing import Any, Optional, Tuple, Union, List, Dict
import os
import pygame

def blit_text(surface, words, pos, font, color):
    # taken from: https://stackoverflow.com/questions/42014195/rendering-text-with-multiple-lines-in-pygame
    space = font.size(' ')[0]  # The width of a space.
    max_width, _ = surface.get_size()
    x, y = pos
    for ii, word in enumerate(words):
        if ii < len(words) - 1:
            word += ','
        word_surface = font.render(word, 0, color)
        word_width, word_height = word_surface.get_size()
        if x + word_width >= max_width - 5:
            x = pos[0]  # Reset the x.
            y += word_height  # Start on new row.
        surface.blit(word_surface, (x, y))
        x += word_width + space

WAIT = 0
LEFT = 1
RIGHT = 2
CLAIM = 3
MARK = 4

@struct.dataclass
class ObsState:
    agent_locs: jnp.ndarray
    goal_locs: jnp.ndarray
    marks: jnp.ndarray
    goal_types: jnp.ndarray
    achieved: jnp.ndarray

@struct.dataclass
class State:
    n_steps: int
    agent_locs: jnp.ndarray
    goal_locs: jnp.ndarray
    marks: jnp.ndarray
    goal_types: jnp.ndarray
    achieved: jnp.ndarray
    observation: jnp.ndarray
    raw_comm: jnp.ndarray
    comm_actions: jnp.ndarray
    valid_comm: jnp.ndarray
    claims: jnp.ndarray

class Coordinate(MultiAgentEnv):
    def __init__(
            self,
            num_agents=2,
            grid_size=6,
            max_episode_len=50,
            window_height=800,
            comm_type='reg',  # 'reg'/'dense'/'no_comm',
            render_fps=2,
            render_mode='rgb_array',
            font_size=28,
    ):
        self.num_agents = num_agents
        self.grid_size = grid_size
        if max_episode_len is None:
            self.max_episode_len = jnp.inf
        else:
            self.max_episode_len = max_episode_len
        self.comm_type = comm_type
        self.render_fps = render_fps
        self.render_mode = render_mode
        self.font_size = font_size
        self.agent_range = jnp.arange(num_agents)
        self.cont_agents = ["player_" + str(i) for i in range(num_agents)]
        self.comm_agents = [agent + "_comm" for agent in self.cont_agents]
        self.agents = self.cont_agents + self.comm_agents
        self.a_to_i = {a: i for i, a in enumerate(self.cont_agents)}

        self.setup_spaces()
        self.render_init = False
        self.window_height = window_height

    def centralized_size(self):
        return self.num_agents * (4 * self.grid_size + 1)

    def communication_size(self):
        return self.num_agents * (4 * self.grid_size + 1 + len(self.vocab) + 1)

    def setup_rendering(self):
        Bhsize = 500
        csize = (Bhsize - 10) / self.grid_size
        self.cell_size = (csize, csize)
        self.BOARD_SIZE = (Bhsize, csize + 10)
        self.y_increase = 60
        self.WINDOW_SIZE = (self.BOARD_SIZE[0], self.num_agents * self.BOARD_SIZE[1] + self.y_increase * self.num_agents)

        self.vlineh = self.BOARD_SIZE[1]-10
        self.vlinew = int((16/296) * self.vlineh)
        render_images = {}
        render_images['vline'] = pygame.transform.scale(
            pygame.image.load(os.path.join(os.path.dirname(__file__), f"img/vline.png")), (self.vlinew, self.vlineh))
        self.bg_offset = int((208.0/320) * self.BOARD_SIZE[1])
        self._null_text_inputs = [("") for _ in range(self.num_agents)]
        self.window_surface = None
        self.clock = pygame.time.Clock()
        bg_name = os.path.join(os.path.dirname(__file__), "img/bg.png")
        render_images['bg'] = pygame.transform.scale(
            pygame.image.load(bg_name), (self.BOARD_SIZE[0] - 2 * self.bg_offset + 30, self.BOARD_SIZE[1])
        )
        rbg_name = os.path.join(os.path.dirname(__file__), "img/rbg.png")
        render_images['rbg'] = pygame.transform.scale(
            pygame.image.load(rbg_name), (self.bg_offset, self.BOARD_SIZE[1])
        )
        lbg_name = os.path.join(os.path.dirname(__file__), "img/lbg.png")
        render_images['lbg'] = pygame.transform.scale(
            pygame.image.load(lbg_name), (self.bg_offset, self.BOARD_SIZE[1])
        )
        claim_name = os.path.join(os.path.dirname(__file__), "img/claim.png")
        render_images['claim'] = pygame.transform.scale(
            pygame.image.load(claim_name), (self.BOARD_SIZE[0], self.BOARD_SIZE[1])
        )
        pygame.font.init()
        self._font = pygame.font.SysFont('Comic Sans MS', 40)
        self.font_render = lambda text: self._font.render(text, False, (255, 255, 255))
        obj_ratio = 0.9
        self.obj_offset = (
            int((self.BOARD_SIZE[1] - int(self.cell_size[0]*obj_ratio))/2),
            int((self.BOARD_SIZE[1] - int(self.cell_size[1]*obj_ratio))/2)
        )
        def load_piece(file_name):
            img_path = os.path.join(os.path.dirname(__file__), f"img/{file_name}.png")
            return pygame.transform.scale(
                pygame.image.load(img_path), (int(self.cell_size[0]*obj_ratio), int(self.cell_size[1]*obj_ratio))
            )

        render_images['shared_goal'] = load_piece("shared_goal")
        render_images['private_goal'] = load_piece("private_goal")
        render_images['agent'] = load_piece("agent")
        render_images['mark'] = load_piece("mark")
        self.render_init = True

        return render_images

    def setup_spaces(self):
        self.vocab = ('agent', 'goal', 'shared goal', 'mark', 'achieved')
        self.observation_spaces = {
            name: Box(
                low=0.,
                high=1.,
                shape=(4 * self.grid_size + 1 + self.num_agents,),
                dtype=jnp.float32,
            ) for i, name in enumerate(self.agents)
        }
        self.action_spaces = {
            name: Discrete(5) for i, name in enumerate(self.cont_agents)
        }
        self.action_spaces.update({name: MultiDiscrete([2] * (len(self.vocab) + 1)) for i, name in enumerate(self.comm_agents)})
        self.poses = {
            'agent': (0, self.grid_size),
            'goal': (self.grid_size, 2 * self.grid_size),
            'shared goal': (2 * self.grid_size, 3 * self.grid_size),
            'mark': (3 * self.grid_size, 4 * self.grid_size),
            'achieved': (4 * self.grid_size, 4 * self.grid_size + 1),
        }
        self.act2mask = jnp.zeros((len(self.vocab), 4 * self.grid_size + 1))
        for i, k in enumerate(self.vocab):
            self.act2mask = self.act2mask.at[i, self.poses[k][0]:self.poses[k][1]].set(1)

    @partial(jax.jit, static_argnums=[0])
    def observe(self, obs_state: ObsState):
        def _observe(agent_id: jnp.ndarray, state: ObsState):
            locf = jnp.zeros(self.grid_size, dtype=jnp.float32)
            glocf = jnp.zeros(self.grid_size,dtype=jnp.float32)
            sglocf = jnp.zeros(self.grid_size, dtype=jnp.float32)
            markf = jnp.zeros(self.grid_size, dtype=jnp.float32)

            aloc = state.agent_locs[agent_id]
            locf = locf.at[0].set(1)
            locf = jnp.roll(locf, aloc)
            gloc = state.goal_locs[agent_id]
            gtype = state.goal_types[agent_id]
            glocf = glocf.at[0].set(1)
            achieved = state.achieved[agent_id]
            glocf = jnp.roll(glocf, gloc) * (1 - gtype) * (1 - achieved)
            sglocf = sglocf.at[0].set(1)
            sglocf = jnp.roll(sglocf, gloc) * gtype * (1 - achieved)
            mloc = state.marks[agent_id]
            markf = markf.at[0].set(1)
            markf = jnp.roll(markf, mloc) * (mloc >= 0)

            return jnp.concatenate((locf.flatten(), glocf.flatten(), sglocf.flatten(), markf.flatten(), achieved.flatten()), axis=0)

        vmap_observe = jax.vmap(_observe, (0, None), 0)
        observations = vmap_observe(self.agent_range, obs_state)
        return observations

    @partial(jax.jit, static_argnums=[0])
    def valid_comm_act(self, observations):
        def _valid_comm_act(obs):
            comm_action = jnp.zeros((len(self.vocab) + 1,))
            comm_action = comm_action.at[1:].set(self.act2mask @ obs[..., None].squeeze(-1))
            comm_action = comm_action.at[0].set(jnp.any(comm_action[1:]))
            return (comm_action > 0).astype(jnp.int32)

        vmap_valid_comm_act = jax.vmap(_valid_comm_act, 0, 0)
        return vmap_valid_comm_act(observations)

    @partial(jax.jit, static_argnums=[0])
    def prep_comm(self, comm_actions, last_observations):
        def _prep_comm(comm_action, last_obs):
            mask = comm_action[1:].reshape((1, -1)) @ self.act2mask
            mask = jnp.clip(mask, 0, 1).astype(jnp.float32)
            comm_obs = last_obs * mask * comm_action[0]
            return comm_obs.flatten()

        vmap_prep_comm = jax.vmap(_prep_comm, (0, 0), 0)
        comm_array = vmap_prep_comm(comm_actions, last_observations)

        return comm_array

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        keys = jax.random.split(key, self.num_agents + 1)
        key = keys[0]
        perm = jax.vmap(lambda k: jax.random.permutation(k, self.grid_size), 0, 0)
        random_locs = perm(jnp.concatenate([k[None, ...] for k in keys[1:]], axis=0))
        agent_locs = random_locs[..., 0]
        goal_locs = random_locs[..., 1]
        key, subkey = jax.random.split(key)
        goal_types = jax.random.randint(subkey, (self.num_agents,), 0, 2)
        marks = (-1) * jnp.ones((self.num_agents,), dtype=jnp.float32)
        achieved = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        obs_state = ObsState(
            agent_locs=agent_locs,
            goal_locs=goal_locs,
            marks=marks,
            goal_types=goal_types,
            achieved=achieved,
        )
        obs_array = self.observe(obs_state)
        cent_obs = obs_array.flatten()[None, :].repeat(self.num_agents, axis=0)
        valid_comm = self.valid_comm_act(obs_array)

        comm_actions = jnp.zeros((self.num_agents, len(self.vocab) + 1)).astype(jnp.int32)
        raw_comm = self.prep_comm(comm_actions, jnp.zeros_like(obs_array))
        proc_comm = jnp.concatenate((raw_comm, comm_actions), axis=1).flatten()
        obss = jnp.concatenate((jnp.eye(self.num_agents), obs_array), axis=-1)
        observations = {agent: obss[i] for i, agent in enumerate(self.cont_agents)}
        observations.update({k + '_comm': v for k, v in observations.items()})
        observations.update({'communication': proc_comm, 'centralized': cent_obs, 'valid_comm': valid_comm})
        state = State(
            n_steps=0,
            agent_locs=agent_locs,
            goal_locs=goal_locs,
            marks=marks,
            goal_types=goal_types,
            achieved=achieved,
            observation=obs_array,
            raw_comm=raw_comm,
            comm_actions=comm_actions,
            valid_comm=valid_comm,
            claims=jnp.zeros((self.num_agents,), dtype=bool),
        )
        return observations, state

    @partial(jax.jit, static_argnums=[0])
    def validate_apply_action(self, actions, agent_locs, goal_locs, goal_types, marks, achieved):
        def _validate_apply_action(loc, mark, action):
            new_loc = loc - (action == LEFT) + (action == RIGHT)
            mark = mark * (action != MARK) + loc * (action == MARK)
            claim = action == CLAIM
            return jnp.clip(new_loc, 0, self.grid_size - 1), mark, claim

        vmap_validate_apply_action = jax.vmap(_validate_apply_action, (0, 0, 0), (0, 0, 0))
        new_locs, marks, claims = vmap_validate_apply_action(agent_locs, marks, actions)

        def _calc_reward(agent_id, act, loc, gloc, gtype, claim, mks, achvd):
            other_marks = jnp.roll(mks, -agent_id)[1:]
            global_mark = jnp.all(other_marks == gloc)
            reached = jnp.logical_or(((gtype > 0) * global_mark + (gtype <= 0)) * claim * (loc == gloc), achvd)
            reward = (100 * reached * (gtype > 0) + 10 * reached * (gtype <= 0) - 2 * (1 - reached)) * (1 - achvd) - 1 * achvd
            return reached, reward

        vmap_calc_reward = jax.vmap(_calc_reward, (0, 0, 0, 0, 0, 0, None, 0), (0, 0))
        achieved, reward = vmap_calc_reward(jnp.arange(self.num_agents), actions, agent_locs, goal_locs, goal_types, claims, marks, achieved)
        return new_locs, marks, achieved.astype(jnp.float32), reward, claims

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State, actions: dict):
        cont_actions = jnp.array([actions[i] for i in self.cont_agents])
        if self.comm_type == 'no_comm':
            comm_actions = jnp.concatenate([jnp.zeros_like(actions[ca])[None, ...] for ca in self.comm_agents], axis=0)
        elif self.comm_type == 'dense':
            comm_actions = jnp.concatenate([jnp.ones_like(actions[ca])[None, ...] for ca in self.comm_agents],axis=0)
        else:
            assert self.comm_type == 'reg'
            comm_actions = jnp.concatenate([actions[ca][None, ...] for ca in self.comm_agents], axis=0)
        n_steps = state.n_steps + 1
        truncated = n_steps >= self.max_episode_len

        new_agent_locs, marks, achieved, reward, claims = self.validate_apply_action(
            cont_actions,
            state.agent_locs,
            state.goal_locs,
            state.goal_types,
            state.marks,
            state.achieved,
        )
        reward = reward.mean()
        obs_state = ObsState(
            agent_locs=new_agent_locs,
            goal_locs=state.goal_locs,
            marks=marks,
            goal_types=state.goal_types,
            achieved=achieved,
        )
        obs_array = self.observe(obs_state)
        cent_obs = obs_array.flatten()[None, :].repeat(self.num_agents, axis=0)
        valid_comm = self.valid_comm_act(obs_array)
        raw_comm = self.prep_comm(comm_actions * state.valid_comm, state.observation)
        proc_comm = jnp.concatenate((raw_comm, comm_actions * state.valid_comm), axis=1).flatten()
        observations = {'communication': proc_comm, 'centralized': cent_obs, 'valid_comm': valid_comm}
        success = jnp.all(achieved)
        rewards, dones, infos = {}, {}, {}
        dones['__all__'] = jnp.logical_or(truncated, success).astype(bool).reshape(())
        infos['success'] = success * jnp.ones(self.num_agents, dtype=bool)
        infos['truncated'] = truncated * jnp.ones(self.num_agents, dtype=bool)
        yell_info, valid_yell_info = {}, {}
        vocab_info = {word: {} for word in self.vocab}
        valid_vocab = {word: {} for word in self.vocab}
        obss = jnp.concatenate((jnp.eye(self.num_agents), obs_array), axis=-1)
        for aid, agent in enumerate(self.cont_agents):
            dones[agent] = dones['__all__']
            dones[agent + '_comm'] = dones[agent]
            rewards[agent] = reward
            rewards[agent + '_comm'] = reward
            observations[agent] = obss[aid]
            observations[agent + '_comm'] = obss[aid]
            word_pos = comm_actions[aid, 1:] * comm_actions[aid, 0]
            valid_word_pos = valid_comm[aid, 1:] * valid_comm[aid, 0]
            yell_info[agent] = jnp.array(comm_actions[aid, 0]).astype(jnp.float32)
            valid_yell_info[agent] = jnp.array(valid_comm[aid, 0]).astype(jnp.float32)
            for j, word in enumerate(self.vocab):
                vocab_info[word][agent] = jnp.array(word_pos[j]).astype(jnp.float32)
                valid_vocab[word][agent] = jnp.array(valid_word_pos[j]).astype(jnp.float32)
        infos['yell'] = yell_info
        infos['vocab'] = vocab_info
        infos['valid_vocab'] = valid_vocab
        infos['valid_yell'] = valid_yell_info
        new_state = State(
            n_steps=n_steps,
            agent_locs=new_agent_locs,
            goal_locs=state.goal_locs,
            marks=marks,
            goal_types=state.goal_types,
            achieved=achieved,
            observation=obs_array,
            raw_comm=raw_comm,
            comm_actions=comm_actions,
            valid_comm=valid_comm,
            claims=claims,
        )
        return observations, new_state, rewards, dones, infos

    def render(self, state: State, render_images=None):
        if not self.render_init:
            render_images = self.setup_rendering()
        agent_locs, goal_locs, goal_types, marks, achieved, comm_actions, claims = (
            np.array(state.agent_locs),
            np.array(state.goal_locs),
            np.array(state.goal_types),
            np.array(state.marks),
            np.array(state.achieved),
            np.array(state.comm_actions),
            np.array(state.claims)
        )
        pic_offset = 14
        font_words = pygame.font.SysFont('Helvetica', self.font_size)
        if self.window_surface is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Coordinate")
                self.window_surface = pygame.display.set_mode(self.WINDOW_SIZE)
            elif self.render_mode == "rgb_array":
                self.window_surface = pygame.Surface(self.WINDOW_SIZE)

        x_ratio = (self.BOARD_SIZE[0] - 10) / self.grid_size
        piece_correct = [(0, 0) for _ in self.cont_agents]
        for i in range(len(self.cont_agents)):
            curr_words = []
            if comm_actions[i][0]:
                for word, vv in zip(self.vocab, comm_actions[i][1:]):
                    if vv:
                        curr_words.append(word)
            piece_correct[i] = (0, i * (self.BOARD_SIZE[1] + self.y_increase))
            self.window_surface.blit(render_images['bg'], (self.bg_offset - 15, i * (self.BOARD_SIZE[1] + self.y_increase)))
            self.window_surface.blit(render_images['lbg'], (0, i * (self.BOARD_SIZE[1] + self.y_increase)))
            self.window_surface.blit(render_images['rbg'], (self.BOARD_SIZE[0] - self.bg_offset, i * (self.BOARD_SIZE[1] + self.y_increase)))
            for j in range(self.grid_size - 1):
                self.window_surface.blit(
                    render_images['vline'], ((j + 1) * x_ratio + 5, i * (self.BOARD_SIZE[1] + self.y_increase) + 5)
                )
            pygame.draw.rect(
                self.window_surface,
                (0, 0, 0),
                pygame.Rect(
                    0,
                    i * (self.BOARD_SIZE[1] + self.y_increase) + self.BOARD_SIZE[1],
                    self.BOARD_SIZE[0],
                    self.y_increase
                )
            )

            blit_text(self.window_surface, curr_words, (2, i * (self.BOARD_SIZE[1] + self.y_increase) + self.BOARD_SIZE[1] + pic_offset), font_words, color=pygame.Color('white'))

        for j in range(len(self.cont_agents)):
            agent_loc = (agent_locs[j] * x_ratio + piece_correct[j][0] + self.obj_offset[0], piece_correct[j][1] + self.obj_offset[1])
            if not achieved[j]:
                goal_loc = (goal_locs[j] * x_ratio + piece_correct[j][0] + self.obj_offset[0], piece_correct[j][1] + self.obj_offset[1])
                goal_type = goal_types[j]
                gimage = render_images['shared_goal'] if goal_type > 0 else render_images['private_goal']
                self.window_surface.blit(gimage, goal_loc)
            if marks[j] >= 0:
                mark_loc = (marks[j] * x_ratio + piece_correct[j][0] + self.obj_offset[0], piece_correct[j][1] + self.obj_offset[1])
                self.window_surface.blit(render_images['mark'], mark_loc)
            self.window_surface.blit(render_images['agent'], agent_loc)
            if claims[j]:
                self.window_surface.blit(render_images['claim'], (0, j * (self.BOARD_SIZE[1] + self.y_increase)))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return render_images
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            ), render_images

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__


@struct.dataclass
class LogEnvState:
    env_state: State
    cont_episode_returns: jnp.ndarray
    comm_episode_returns: jnp.ndarray
    episode_lengths: int
    success_episode: jnp.ndarray
    returned_cont_episode_returns: jnp.ndarray
    returned_comm_episode_returns: jnp.ndarray
    returned_episode_lengths: jnp.ndarray
    returned_success_episode: jnp.ndarray
    episode_yells: jnp.ndarray
    episode_vocab: Dict[str, jnp.ndarray]
    returned_episode_yells: jnp.ndarray
    returned_episode_vocab: Dict[str, jnp.ndarray]
    episode_valid_yells: jnp.ndarray
    episode_valid_vocab: Dict[str, jnp.ndarray]
    returned_episode_valid_yells: jnp.ndarray
    returned_episode_valid_vocab: Dict[str, jnp.ndarray]


class CoordinateLogWrapper(JaxMARLWrapper):
    def __init__(self, env: MultiAgentEnv, replace_info: bool = False):
        super().__init__(env)
        self.replace_info = replace_info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        obs, env_state = self._env.reset(key)
        state = LogEnvState(
            env_state=env_state,
            cont_episode_returns=jnp.zeros((self._env.num_agents,)),
            comm_episode_returns=jnp.zeros((self._env.num_agents,)),
            episode_lengths=jnp.zeros((self._env.num_agents,)),
            success_episode=jnp.zeros((self._env.num_agents,), dtype=bool),
            returned_cont_episode_returns=jnp.zeros((self._env.num_agents,)),
            returned_comm_episode_returns=jnp.zeros((self._env.num_agents,)),
            returned_episode_lengths=jnp.zeros((self._env.num_agents,)),
            returned_success_episode=jnp.zeros((self._env.num_agents,)),
            episode_yells=jnp.zeros((self._env.num_agents,)),
            episode_vocab={k: jnp.zeros((self._env.num_agents,)) for k in self._env.vocab},
            returned_episode_yells=jnp.zeros((self._env.num_agents,)),
            returned_episode_vocab={k: jnp.zeros((self._env.num_agents,)) for k in self._env.vocab},
            episode_valid_yells=jnp.zeros((self._env.num_agents,)),
            episode_valid_vocab={k: jnp.zeros((self._env.num_agents,)) for k in self._env.vocab},
            returned_episode_valid_yells=jnp.zeros((self._env.num_agents,)),
            returned_episode_valid_vocab={k: jnp.zeros((self._env.num_agents,)) for k in self._env.vocab},
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
            self,
            key: chex.PRNGKey,
            state: LogEnvState,
            action: Union[int, float],
    ) -> Tuple[chex.Array, LogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        ep_done = done["__all__"]
        new_cont_episode_return = state.cont_episode_returns + self._batchify_floats(reward, self._env.cont_agents)
        new_comm_episode_return = state.comm_episode_returns + self._batchify_floats(reward, self._env.comm_agents)
        new_episode_length = state.episode_lengths + 1
        new_success_episode = info['success']
        new_episode_yells = state.episode_yells + self._batchify_floats(info['yell'], self._env.cont_agents)
        new_episode_vocab = {kk: state.episode_vocab[kk] + self._batchify_floats(info['vocab'][kk], self._env.cont_agents) for kk in self._env.vocab}
        new_episode_valid_yells = state.episode_valid_yells + self._batchify_floats(info['valid_yell'], self._env.cont_agents)
        new_episode_valid_vocab = {kk: state.episode_valid_vocab[kk] + self._batchify_floats(info['valid_vocab'][kk], self._env.cont_agents) for kk in self._env.vocab}

        new_state = LogEnvState(
            env_state=env_state,
            success_episode=(new_success_episode * (1 - ep_done)).astype(bool),
            cont_episode_returns=new_cont_episode_return * (1 - ep_done),
            comm_episode_returns=new_comm_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_cont_episode_returns=state.returned_cont_episode_returns * (1 - ep_done)
                                          + new_cont_episode_return * ep_done,
            returned_comm_episode_returns=state.returned_comm_episode_returns * (1 - ep_done)
                                          + new_comm_episode_return * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
                                     + new_episode_length * ep_done,
            returned_success_episode=state.returned_success_episode * (1 - ep_done)
                                     + new_success_episode * ep_done,
            episode_yells=new_episode_yells * (1 - ep_done),
            episode_vocab={k: v * (1 - ep_done) for k, v in new_episode_vocab.items()},
            returned_episode_yells=state.returned_episode_yells * (1 - ep_done)
                                   + new_episode_yells * ep_done,
            returned_episode_vocab={k: vp * (1 - ep_done) + vn * ep_done
                                    for k, vp, vn in
                                    zip(self._env.vocab,
                                        state.returned_episode_vocab.values(),
                                        new_episode_vocab.values())},
            episode_valid_yells=new_episode_valid_yells * (1 - ep_done),
            episode_valid_vocab={k: v * (1 - ep_done) for k, v in new_episode_valid_vocab.items()},
            returned_episode_valid_yells=state.returned_episode_valid_yells * (1 - ep_done)
                                      + new_episode_valid_yells * ep_done,
            returned_episode_valid_vocab={k: vp * (1 - ep_done) + vn * ep_done
                                       for k, vp, vn in
                                       zip(self._env.vocab,
                                           state.returned_episode_valid_vocab.values(),
                                           new_episode_valid_vocab.values())},
        )
        if self.replace_info:
            info = {}
        info["returned_cont_episode_returns"] = new_state.returned_cont_episode_returns
        info["returned_comm_episode_returns"] = new_state.returned_comm_episode_returns
        info["returned_episode_lengths"] = new_state.returned_episode_lengths
        info["returned_success_episode"] = new_state.returned_success_episode
        info["returned_episode"] = jnp.full((self._env.num_agents,), ep_done)
        info["returned_episode_yells"] = new_state.returned_episode_yells / new_state.returned_episode_lengths
        info["returned_episode_vocab"] = {
            k: v / new_state.returned_episode_lengths for k, v in new_state.returned_episode_vocab.items()
        }
        info["returned_episode_valid_yells"] = new_state.returned_episode_valid_yells / new_state.returned_episode_lengths
        info["returned_episode_valid_vocab"] = {
            k: v / new_state.returned_episode_lengths for k, v in new_state.returned_episode_valid_vocab.items()
        }
        return obs, new_state, reward, done, info

    def _batchify_floats(self, x: dict, names):
        return jnp.stack([x[a] for a in names])

    def _batchify_floats_dict(self, x: dict, names, key_list):
        def _get(_x, _kl):
            v = _x
            for k in _kl:
                v = v[k]
            return v
        return jnp.stack([_get(x[a], key_list) for a in names])

    def render(self, state, *args, **kwargs):
        return self._env.render(state.env_state, *args, **kwargs)
