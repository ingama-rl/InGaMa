import jax
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.wrappers.baselines import JaxMARLWrapper
from jaxmarl.environments.spaces import Discrete, MultiDiscrete, Box
import chex
from flax import struct
from functools import partial
from jaxmarl.environments.multi_maze.maps import maps
from jaxmarl.environments.multi_maze.utils import load_map
from jaxmarl.environments.multi_maze.rendering import initiate_rendering, _render
from typing import Any, Optional, Tuple, Union, List, Dict
import itertools

WAIT = 0
LEFT = 1
RIGHT = 2
UP = 3
DOWN = 4

WAIT_ACT = jnp.array((0,0), dtype=jnp.int32)
LEFT_ACT = jnp.array((-1,0), dtype=jnp.int32)
RIGHT_ACT = jnp.array((1,0), dtype=jnp.int32)
UP_ACT = jnp.array((0,-1), dtype=jnp.int32)
DOWN_ACT = jnp.array((0,1), dtype=jnp.int32)



@struct.dataclass
class ObsState:
    agent_locs: jnp.ndarray
    agent_map: jnp.ndarray
    door_map: jnp.ndarray
    key_map: jnp.ndarray
    goal_map: jnp.ndarray
    doors_state: jnp.ndarray

@struct.dataclass
class State:
    n_steps: int
    rgdist: float
    success: jnp.ndarray
    agent_locs: jnp.ndarray
    agent_map: jnp.ndarray
    doors: jnp.ndarray
    cl_doors: jnp.ndarray
    door_map: jnp.ndarray
    keys: jnp.ndarray
    decoy_keys: jnp.ndarray
    cl_keys: jnp.ndarray
    key_map: jnp.ndarray
    goal: jnp.ndarray
    goal_map: jnp.ndarray
    doors_state: jnp.ndarray
    observation: jnp.ndarray
    raw_comm: jnp.ndarray
    comm_actions: jnp.ndarray
    valid_comm: jnp.ndarray
    rndid: int
    ds_keys: jnp.ndarray

class MultiMaze(MultiAgentEnv):
    def __init__(
            self,
            gamma=0.99,
            num_agents=3,
            max_episode_len=100,
            window_height=800,
            comm_type='reg',  # 'reg'/'dense'/'no_comm',
            map='Triple Trouble',
            save_compilation=True,
            render_fps=2,
            render_mode='rgb_array',
    ):
        map_data = maps.get(map, None)
        assert map_data is not None, f"Map name (" + map + ") not found"

        self.gamma = gamma
        self.num_agents = num_agents
        if max_episode_len is None:
            self.max_episode_len = jnp.inf
        else:
            self.max_episode_len = max_episode_len
        self.comm_type = comm_type
        self.render_fps = render_fps
        self.render_mode = render_mode

        self.agent_range = jnp.arange(num_agents)
        self.cont_agents = [map_data['colors'][i] + "_player" for i in range(num_agents)]
        self.comm_agents = [agent + "_comm" for agent in self.cont_agents]
        self.agents = self.cont_agents + self.comm_agents
        self.a_to_i = {a: i for i, a in enumerate(self.cont_agents)}

        load_map(self, map_data, num_agents, save_compilation)
        self.setup_spaces(num_agents)
        self.render_init = False
        self.window_height = window_height

    def centralized_size(self):
        return self.cent_dim

    def communication_size(self):
        return self.comm_dim

    def setup_rendering(self):
        render_images = initiate_rendering(self, self.num_agents, self.window_height)
        self.render_init = True
        return render_images

    def setup_spaces(self, num_agents):
        self.observation_spaces = {
            name: Box(
                low=0.,
                high=num_agents + 1.,
                shape=(self.self_obs_dim,),
                dtype=jnp.float32,
            ) for i, name in enumerate(self.agents)
        }
        self.action_spaces = {
            name: Discrete(5) for i, name in enumerate(self.cont_agents)
        }
        self.action_spaces.update({name: MultiDiscrete([2] * (len(self.vocab) + 1)) for i, name in enumerate(self.comm_agents)})

    @partial(jax.jit, static_argnums=[0])
    def _random_agent_locs(self, key: chex.PRNGKey):
        key, subkey = jax.random.split(key)
        agent_locs = []
        for vloc in self.valid_locs:
            key, subkey = jax.random.split(key)
            agent_locs.append(jax.random.choice(subkey, vloc))
        agent_locs = jnp.array(agent_locs)

        return agent_locs, key

    @partial(jax.jit, static_argnums=[0])
    def _random_initial(self, key: chex.PRNGKey):
        key, subkey = jax.random.split(key)
        door_perm = jnp.arange(len(self.all_doors))
        dc_ind = jax.random.randint(subkey, (1,), 0, len(self.all_doors))
        door_perm = door_perm.at[len(self.all_doors) - 1].set(dc_ind[0])
        door_perm = jnp.roll(door_perm, -dc_ind[0])
        door_perm = door_perm.at[0].set(len(self.all_doors) - 1)
        door_perm = jnp.roll(door_perm, dc_ind[0])
        doors = self.all_doors[door_perm]
        keys = []
        decoy_keys = []
        cl_keys = []
        key, subkey = jax.random.split(key)
        rdid = jax.random.choice(subkey, jnp.arange(len(self.init_dist['probs'])), p=self.init_dist['probs'])
        rndid = self.init_dist['rndid'][rdid]
        for _i, kk in enumerate(self.keys_init):
            key, subkey = jax.random.split(key)
            assert _i == self.colored_ids[_i]
            i = self.colored_ids[_i]
            _kid = jax.random.randint(subkey, (1,), 0, self.init_dist['lens'][i][rdid])
            kid = self.init_dist['ids'][i][rdid][_kid]
            keys.append(kk[kid].flatten())

        for _i, clkk in enumerate(self.all_cl_keys):
            key, subkey = jax.random.split(key)
            assert _i + len(self.keys_init) == self.colorless_ids[_i]
            i = self.colorless_ids[_i]
            _clkid = jax.random.randint(subkey, (1,), 0, self.init_dist['lens'][i][rdid])
            clkid = self.init_dist['ids'][i][rdid][_clkid[0]]
            cl_keys.append(clkk[clkid].flatten())

        key, subkey = jax.random.split(key)
        _gid = jax.random.randint(subkey, (1,), 0, self.init_dist['lens'][-1][rdid])
        gid = self.init_dist['ids'][-1][rdid][_gid[0]]
        goal = self.all_goals[gid]

        for i, dkk in enumerate(self.all_decoy_keys):
            _ids = jnp.zeros(len(dkk), dtype=bool)
            for _kk in keys + decoy_keys:
                _ids = jnp.logical_or(_ids, jnp.logical_and(dkk[:, 0] == _kk[0], dkk[:, 1] == _kk[1]))
            key, subkey = jax.random.split(key)
            dkid = jnp.argmin(jax.random.permutation(subkey, len(dkk)) + (jnp.inf * _ids))
            decoy_keys.append(dkk[dkid].flatten())

        if self.decoy:
            keys.append(self.all_keys[-1])
            decoy_keys = jnp.stack(decoy_keys)
            decoy_id = jnp.sum(jnp.arange(len(door_perm)) * (door_perm == len(doors) - 1))
        else:
            decoy_keys = jnp.array([])
            decoy_id = None

        keys = jnp.stack(keys)
        cl_keys = jnp.stack(cl_keys)
        ds_keys = jnp.concatenate((keys[door_perm], cl_keys))
        return doors, self.all_cl_doors, keys[door_perm], decoy_keys, cl_keys, decoy_id, ds_keys, goal, rndid, key



    @partial(jax.jit, static_argnums=[0])
    def _random_doors_keys(self, key: chex.PRNGKey):
        key, subkey = jax.random.split(key)
        door_perm = jax.random.permutation(subkey, len(self.all_doors))
        doors = self.all_doors[door_perm]
        keys = []
        decoy_keys = []
        cl_keys = []
        ssid = 0
        for _i, kk in enumerate(self.keys_init):
            key, subkey = jax.random.split(key)
            assert _i == self.colored_ids[_i]
            i = self.colored_ids[_i]

            _kid = jax.random.randint(subkey, (1,), 0, self.seq_sample['len'][i][ssid])
            kid = self.seq_sample['loc'][i][ssid][_kid[0]]
            keys.append(kk[kid].flatten())
            kroom = self.keys_rooms[i][kid]
            ssid = self.seq_sample['ids'][i][ssid][kroom]

        for _i, clkk in enumerate(self.all_cl_keys):
            key, subkey = jax.random.split(key)
            assert _i + len(self.keys_init) == self.colorless_ids[_i]
            i = self.colorless_ids[_i]
            _clkid = jax.random.randint(subkey, (1,), 0, self.seq_sample['len'][i][ssid])
            clkid = self.seq_sample['loc'][i][ssid][_clkid[0]]
            cl_keys.append(clkk[clkid].flatten())
            clkroom = self.keys_rooms[i][clkid]
            ssid = self.seq_sample['ids'][i][ssid][clkroom]
        goal_ssm_len = self.seq_sample['len'][-1][ssid]
        goal_ssm_loc = self.seq_sample['loc'][-1][ssid]
        goal2rndid = self.seq_sample['ids'][-1][ssid]
        for i, dkk in enumerate(self.all_decoy_keys):
            _ids = jnp.zeros(len(dkk), dtype=bool)
            for _kk in keys:
                _ids = jnp.logical_or(_ids, jnp.logical_and(dkk[:, 0] == _kk[0], dkk[:, 1] == _kk[1]))
            key, subkey = jax.random.split(key)
            dkid = jnp.argmin(jax.random.permutation(subkey, len(dkk)) + (jnp.inf * _ids))
            decoy_keys.append(dkk[dkid].flatten())

        if self.decoy:
            keys.append(self.all_keys[-1])
            decoy_keys = jnp.stack(decoy_keys)
            decoy_id = jnp.sum(jnp.arange(len(door_perm)) * (door_perm == len(doors) - 1))
        else:
            decoy_keys = jnp.array([])
            decoy_id = None

        keys = jnp.stack(keys)
        cl_keys = jnp.stack(cl_keys)
        ds_keys = jnp.concatenate((keys[door_perm], cl_keys))
        return doors, self.all_cl_doors, keys[door_perm], decoy_keys, cl_keys, decoy_id, ds_keys, key, (goal_ssm_len, goal_ssm_loc, goal2rndid)

    @partial(jax.jit, static_argnums=[0])
    def _random_goal(self, key: chex.PRNGKey, goal_info):
        key, subkey = jax.random.split(key)
        _gid = jax.random.randint(subkey, (1,), 0, goal_info[0])
        gid = goal_info[1][_gid[0]]
        goal = self.all_goals[gid]
        rndid = goal_info[2][self.goal_rooms[gid]]
        return goal, rndid, key

    @partial(jax.jit, static_argnums=[0])
    def update_doors_state(self, agent_locs, keys):
        def _update_doors_state(agent_locs, key_):
            key_oc = jnp.logical_and(key_[..., 0] == agent_locs[..., 0], key_[..., 1] == agent_locs[..., 1])
            return jnp.any(key_oc) * jnp.ones((1,), dtype=jnp.float32)

        vmap_update_doors_state = jax.vmap(_update_doors_state, (None, 0), 0)
        return vmap_update_doors_state(agent_locs, keys)

    @partial(jax.jit, static_argnums=[0])
    def initiate_maps(self, doors, cl_doors, keys, decoy_keys, cl_keys, goal, agent_locs, decoy_id):
        agent_map =  jnp.zeros((self.num_agents, self.map_size, self.map_size), dtype=jnp.float32)
        door_map = jnp.zeros((self.door_l, self.map_size, self.map_size), dtype=jnp.float32)
        key_map = jnp.zeros((self.key_l, self.map_size, self.map_size), dtype=jnp.float32)
        goal_map =  jnp.zeros((1, self.map_size, self.map_size), dtype=jnp.float32)
        goal_map = goal_map.at[..., goal[0], goal[1]].set(1)
        key_ind = jnp.arange(len(keys))
        door_ind = jnp.arange(len(doors))
        if self.decoy:
            all_doors = jnp.roll(doors, -decoy_id, axis=0)[1:]
            all_doors = jnp.roll(all_doors, decoy_id, axis=0)
            door_ind = jnp.roll(door_ind, -decoy_id, axis=0)[1:]
            door_ind = jnp.roll(door_ind, decoy_id, axis=0)
            all_keys = jnp.roll(keys, -decoy_id, axis=0)
            all_keys = jnp.roll(jnp.concatenate((decoy_keys, all_keys[1:]), axis=0), -len(decoy_keys), axis=0)
            key_ind = jnp.roll(key_ind, -decoy_id, axis=0)
            key_ind = jnp.roll(jnp.concatenate((key_ind[0] * jnp.ones((len(decoy_keys),), dtype=int), key_ind[1:]), axis=0), -len(decoy_keys), axis=0)
        else:
            door_ind = jnp.arange(len(doors))
            all_doors = doors
            all_keys = keys
        for i, door in zip(door_ind, all_doors):
            door_map = door_map.at[i, door[0], door[1]].set(1)
        for i, key in zip(key_ind, all_keys):
            key_map = key_map.at[i, key[0], key[1]].set(1)

        if len(self.colorless_ids) > 0:
            for i in range(len(self.colorless_ids)):
                door, key = cl_doors[i], cl_keys[i]
                key_map = key_map.at[i - len(self.colorless_ids), key[0], key[1]].set(1)
                door_map = door_map.at[i - len(self.colorless_ids), door[0], door[1]].set(1)
        for i, aloc in enumerate(agent_locs):
            agent_map = agent_map.at[i, aloc[0], aloc[1]].set(1)
        return agent_map, door_map, key_map, goal_map

    @partial(jax.jit, static_argnums=[0])
    def observe_centralized(self, obs_state: ObsState):
        door_view = jnp.concatenate([jnp.logical_not(ds) * dm[None,...] for dm, ds in zip(obs_state.door_map, obs_state.doors_state)], axis=0)
        cl_door_view = door_view[-len(self.colorless_ids):].sum(axis=0)[None, ...]
        door_view = jnp.concatenate((door_view[:-len(self.colorless_ids)], cl_door_view), axis=0)
        key_view = obs_state.key_map
        cl_key_view = key_view[len(self.all_keys):].sum(axis=0)[None, ...]
        key_view = jnp.concatenate((key_view[:len(self.all_keys)], cl_key_view), axis=0)
        all_door_view = door_view.sum(axis=0)[None, ...]
        all_key_view = key_view.sum(axis=0)[None, ...]
        return jnp.concatenate(
            (
                self.wall_map,
                door_view,
                key_view,
                obs_state.goal_map,
                obs_state.agent_map,
                all_door_view,
                all_key_view,
                obs_state.agent_map.sum(axis=0)[None, ...],
            ), axis=0
        ).flatten()[None, :].repeat(self.num_agents, axis=0)

    @partial(jax.jit, static_argnums=[0])
    def observe(self, obs_state: ObsState):
        def _observe(agent_id: jnp.ndarray, state: ObsState, wall_map: jnp.ndarray):
            loc = state.agent_locs[agent_id]
            mask = jnp.zeros((1, self.map_size + 2 * self.view_range, self.map_size + 2 * self.view_range))
            mask = mask.at[..., :self.view_size, :self.view_size].set(1)
            mask = jnp.roll(mask, loc[0], axis=-2)
            mask = jnp.roll(mask, loc[1], axis=-1)
            mask = mask[..., self.view_range:-self.view_range, self.view_range:-self.view_range]
            wall_view = mask * wall_map
            door_view = state.door_map * (1 - state.doors_state.reshape((-1, 1, 1))
                                          .repeat(state.door_map.shape[-1], axis=-1)
                                          .repeat(state.door_map.shape[-2], axis=-2))
            cl_door_view = door_view[-len(self.colorless_ids):].sum(axis=0)[None, ...]
            door_view = jnp.concatenate((door_view[:-len(self.colorless_ids)], cl_door_view), axis=0)
            door_view = mask.repeat(door_view.shape[0], axis=0) * door_view

            key_view = state.key_map
            cl_key_view = key_view[len(self.all_keys):].sum(axis=0)[None, ...]
            key_view = jnp.concatenate((key_view[:len(self.all_keys)], cl_key_view), axis=0)
            key_view = mask.repeat(key_view.shape[0], axis=0) * key_view

            goal_view = mask * state.goal_map

            agent_view = mask.repeat(self.num_agents, axis=0) * state.agent_map

            all_door_view = door_view.sum(axis=0)[None, ...]
            all_key_view = key_view.sum(axis=0)[None, ...]
            all_agent_view = agent_view.sum(axis=0)[None, ...]

            return jnp.concatenate((wall_view, door_view, key_view, goal_view, agent_view, all_door_view, all_key_view, all_agent_view), axis=0).flatten()

        vmap_observe = jax.vmap(_observe, (0, None, None), 0)
        observations = vmap_observe(self.agent_range, obs_state, self.wall_map)

        return jnp.concatenate((jnp.eye(self.num_agents), observations), axis=-1)

    def obs_to_map(self, obs):
        agent_one_hot = obs[..., :self.object_locs['wall'][0]].flatten()
        agent_map = obs[..., self.object_locs['agent'][0]:self.object_locs['agent'][1]].reshape((self.num_agents, self.map_size, self.map_size)).swapaxes(-1, -2)
        door_map = obs[..., self.object_locs['door'][0]:self.object_locs['door'][1]].reshape((len(self.all_doors) + 1, self.map_size, self.map_size)).swapaxes(-1, -2)
        key_map = obs[..., self.object_locs['key'][0]:self.object_locs['key'][1]].reshape((len(self.all_keys) + 1, self.map_size, self.map_size)).swapaxes(-1, -2)
        wall_map = obs[..., self.object_locs['wall'][0]:self.object_locs['wall'][1]].reshape((1, self.map_size, self.map_size)).swapaxes(-1, -2)
        goal_map = obs[..., self.object_locs['goal'][0]:self.object_locs['goal'][1]].reshape((1, self.map_size, self.map_size)).swapaxes(-1, -2)

        all_door_map = obs[..., self.object_locs['__all__']['door'][0]:self.object_locs['__all__']['door'][1]].reshape((1, self.map_size, self.map_size)).swapaxes(-1, -2)
        all_key_map = obs[..., self.object_locs['__all__']['key'][0]:self.object_locs['__all__']['key'][1]].reshape((1, self.map_size, self.map_size)).swapaxes(-1, -2)
        all_agent_map = obs[..., self.object_locs['__all__']['agent'][0]:self.object_locs['__all__']['agent'][1]].reshape((1, self.map_size, self.map_size)).swapaxes(-1, -2)
        all_map = {'door': all_door_map, 'key': all_key_map, 'agent': all_agent_map}
        return {'agent_id': agent_one_hot, 'agent': agent_map, 'door': door_map, 'key': key_map, 'wall': wall_map, 'goal': goal_map, '__all__': all_map}

    def comm_to_map(self, communication, _lib):
        chunked = _lib.split(communication, self.num_agents)
        res = []
        for chunk in chunked:
            obs = chunk[..., :-(len(self.vocab) + 1)]
            agent_map = obs[..., self.object_locs['agent'][0]:self.object_locs['agent'][1]].reshape((self.num_agents, self.map_size, self.map_size)).swapaxes(-1, -2)
            door_map = obs[..., self.object_locs['door'][0]:self.object_locs['door'][1]].reshape((len(self.all_doors) + 1, self.map_size, self.map_size)).swapaxes(-1, -2)
            key_map = obs[..., self.object_locs['key'][0]:self.object_locs['key'][1]].reshape((len(self.all_keys) + 1, self.map_size, self.map_size)).swapaxes(-1, -2)
            wall_map = obs[..., self.object_locs['wall'][0]:self.object_locs['wall'][1]].reshape((1, self.map_size, self.map_size)).swapaxes(-1, -2)
            goal_map = obs[..., self.object_locs['goal'][0]:self.object_locs['goal'][1]].reshape((1, self.map_size, self.map_size)).swapaxes(-1, -2)
            all_door_map = obs[..., self.object_locs['__all__']['door'][0]:self.object_locs['__all__']['door'][1]].reshape((1, self.map_size, self.map_size)).swapaxes(-1, -2)
            all_key_map = obs[..., self.object_locs['__all__']['key'][0]:self.object_locs['__all__']['key'][1]].reshape((1, self.map_size, self.map_size)).swapaxes(-1, -2)
            all_agent_map = obs[..., self.object_locs['__all__']['agent'][0]:self.object_locs['__all__']['agent'][1]].reshape((1, self.map_size, self.map_size)).swapaxes(-1, -2)
            all_map = {'doors': all_door_map, 'keys': all_key_map, 'agents': all_agent_map}
            res.append({'agents': agent_map, 'doors': door_map, 'keys': key_map, 'wall': wall_map, 'goal': goal_map, '__all__': all_map})
        return res

    @partial(jax.jit, static_argnums=[0])
    def valid_comm_act(self, observations):
        def _valid_comm_act(obs, aid):
            comm_action = jnp.zeros((len(self.vocab) + 1,))
            agent_obs = obs[self.object_locs['agent'][0]: self.object_locs['agent'][1]]
            comm_action = comm_action.at[self.txt2pos['agent'] + 1].set(jnp.any(agent_obs))
            comm_action = comm_action.at[self.txt2pos['wall'] + 1].set(jnp.any(obs[self.object_locs['wall'][0]: self.object_locs['wall'][1]]))
            door_obs = obs[self.object_locs['door'][0]: self.object_locs['door'][1]].reshape((-1, self.map_size, self.map_size))
            comm_action = comm_action.at[self.txt2pos['door'] + 1].set(jnp.any(door_obs))
            for i, ad in enumerate(door_obs):
                color = self.colors[i]
                comm_action = comm_action.at[self.txt2pos[color] + 1].set(jnp.logical_or(jnp.any(ad), comm_action[self.txt2pos[color] + 1]))

            key_obs = obs[self.object_locs['key'][0]: self.object_locs['key'][1]].reshape((-1, self.map_size, self.map_size))
            comm_action = comm_action.at[self.txt2pos['key'] + 1].set(jnp.any(key_obs))
            for i, ak in enumerate(key_obs):
                color = self.colors[i]
                comm_action = comm_action.at[self.txt2pos[color] + 1].set(jnp.logical_or(jnp.any(ak), comm_action[self.txt2pos[color] + 1]))
            comm_action = comm_action.at[self.txt2pos['goal'] + 1].set(jnp.any(obs[self.object_locs['goal'][0]: self.object_locs['goal'][1]]))

            comm_action = comm_action.at[0].set(jnp.any(comm_action))

            return comm_action

        vmap_valid_comm_act = jax.vmap(_valid_comm_act, (0, 0), 0)
        return vmap_valid_comm_act(observations, self.agent_range)

    @partial(jax.jit, static_argnums=[0])
    def prep_comm(self, comm_actions, last_observations):
        def _prep_comm(comm_action, last_obs):
            obj_mask = comm_action[1:len(self.object_words) + 1].reshape((1, -1)) @ self.ca2mask[:len(self.object_words)]
            color_mask = comm_action[len(self.object_words) + 1:].reshape((1, -1)) @ self.ca2mask[len(self.object_words):-1]
            mask = jnp.clip(obj_mask * (color_mask + self.ca2mask[-1].reshape(color_mask.shape)), 0, 1).astype(jnp.float32)
            comm_obs = last_obs * mask * comm_action[0]
            return comm_obs.flatten()

        vmap_prep_comm = jax.vmap(_prep_comm, (0, 0), 0)
        comm_array = vmap_prep_comm(comm_actions, last_observations)

        return comm_array

    @partial(jax.jit, static_argnums=[0])
    def get_rgdist(self, rndid, agent_locs):
        dist = self.dists[rndid]
        agent_rooms = []
        for agentid in range(self.num_agents):
            loc = agent_locs[agentid]
            agent_rooms.append(self.id2room[:, loc[0], loc[1]])

        agents_poses = jnp.array(list(itertools.product(*agent_rooms)), dtype=int)
        def _get_rgdist(aposes, dist):
            rid = 0
            for i, ap in enumerate(aposes):
                nrs = self.ar2rgid[i][rid]
                rid = nrs[ap]
            return dist[rid][-1]

        vmap_get_rgid = jax.vmap(_get_rgdist, (0, None), 0)
        rgdist = vmap_get_rgid(agents_poses, dist).min()
        return rgdist.astype(jnp.float32)

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        doors, cl_doors, keys, decoy_keys, cl_keys, decoy_id, ds_keys, goal, rndid, key = self._random_initial(key)
        agent_locs, key = self._random_agent_locs(key)
        doors_state = self.update_doors_state(agent_locs, ds_keys)
        rgdist = self.get_rgdist(rndid, agent_locs)
        success = jnp.zeros((1,), dtype=jnp.float32)
        agent_map, door_map, key_map, goal_map = self.initiate_maps(doors, cl_doors, keys, decoy_keys, cl_keys, goal, agent_locs, decoy_id)
        obs_state = ObsState(
            agent_locs=agent_locs,
            agent_map=agent_map,
            door_map=door_map,
            key_map=key_map,
            goal_map=goal_map,
            doors_state=doors_state.astype(jnp.float32),
        )
        obs_array = self.observe(obs_state)
        cent_obs = self.observe_centralized(obs_state)
        valid_comm = self.valid_comm_act(obs_array)
        comm_actions = jnp.zeros((self.num_agents, len(self.vocab) + 1)).astype(jnp.int32)
        raw_comm = self.prep_comm(comm_actions * jnp.zeros_like(valid_comm), jnp.zeros_like(obs_array))
        proc_comm = jnp.concatenate((raw_comm, comm_actions), axis=1).flatten()
        observations = {agent: obs_array[i] for i, agent in enumerate(self.cont_agents)}
        observations.update({k + '_comm': v for k, v in observations.items()})
        observations.update({'communication': proc_comm, 'centralized': cent_obs, 'valid_comm': valid_comm})
        state = State(
            n_steps=0,
            rgdist=rgdist,
            success=success.astype(bool),
            agent_locs=obs_state.agent_locs,
            agent_map=obs_state.agent_map,
            doors=doors,
            cl_doors=cl_doors,
            door_map=obs_state.door_map,
            keys=keys,
            decoy_keys=decoy_keys,
            cl_keys=cl_keys,
            key_map=obs_state.key_map,
            goal=goal,
            goal_map=goal_map,
            doors_state=obs_state.doors_state,
            observation=obs_array,
            raw_comm=raw_comm,
            comm_actions=comm_actions,
            valid_comm=valid_comm,
            rndid=rndid,
            ds_keys=ds_keys,
        )
        return observations, state

    @partial(jax.jit, static_argnums=[0])
    def validate_apply_action(self, actions, agent_locs, door_map, doors_state):
        def _validate_apply_action(loc, action, door_map, doors_state):
            new_loc = (loc + (action == LEFT) * LEFT_ACT
                       + (action == RIGHT) * RIGHT_ACT + (action == UP) * UP_ACT
                       + (action == DOWN) * DOWN_ACT)
            is_wall = (jnp.any(new_loc >= self.map_size) + jnp.any(new_loc < 0) + self.wall_map[0, new_loc[0], new_loc[1]]) > 0
            door_oc = door_map[..., new_loc[0], new_loc[1]]
            door_oc = door_oc - doors_state.reshape(door_oc.shape) * door_oc
            invalid = is_wall + door_oc.sum() > 0
            return invalid * loc + (1 - invalid) * new_loc , door_oc, is_wall
        vmap_validate_apply_action = jax.vmap(_validate_apply_action, (0, 0, None, None), (0, 0, 0))
        new_locs, door_ocs, is_walls = vmap_validate_apply_action(agent_locs, actions, door_map, doors_state)
        return new_locs

    @partial(jax.jit, static_argnums=[0])
    def is_done(self, goal, agent_locs):
        return jnp.any(jnp.logical_and(agent_locs[:,0] == goal[0], agent_locs[:,1] == goal[1]))


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
        new_agent_locs = self.validate_apply_action(
            cont_actions,
            state.agent_locs,
            state.door_map,
            state.doors_state,
        )
        agent_map = jnp.zeros_like(state.agent_map)
        for i, loc in enumerate(new_agent_locs):
            agent_map = agent_map.at[i, loc[0], loc[1]].set(1)
        doors_state = self.update_doors_state(new_agent_locs, state.ds_keys)
        obs_state = ObsState(
            agent_locs=new_agent_locs,
            agent_map=agent_map,
            door_map=state.door_map,
            key_map=state.key_map,
            goal_map=state.goal_map,
            doors_state=doors_state.astype(jnp.float32),
        )
        obs_array = self.observe(obs_state)
        cent_obs = self.observe_centralized(obs_state)
        valid_comm = self.valid_comm_act(obs_array)
        raw_comm = self.prep_comm(comm_actions * state.valid_comm, state.observation)
        proc_comm = jnp.concatenate((raw_comm, comm_actions * state.valid_comm), axis=1).flatten()

        observations = {'communication': proc_comm, 'centralized': cent_obs, 'valid_comm': valid_comm}
        rgdist = self.get_rgdist(state.rndid, new_agent_locs)
        success = self.is_done(state.goal, new_agent_locs)
        term1 = success
        term2 = jnp.logical_and(rgdist < jnp.inf, jnp.logical_not(term1))
        term3 = jnp.logical_not(jnp.logical_or(term1, term2))
        reward = term1 * 1. + term2 * (1.0 - jnp.clip(rgdist,-1e8,1e8)) + term3 * (- state.rgdist)

        reward = reward - 1.

        all_success = jnp.logical_or(success, state.success)
        rewards, dones, infos = {}, {}, {}
        _dn = False
        dones['__all__'] = jnp.logical_or(jnp.logical_or(truncated, _dn), (self.max_episode_len >= jnp.inf) * all_success).astype(bool).reshape(())
        infos['success'] = all_success * jnp.ones((self.num_agents,))

        infos['truncated'] = truncated * jnp.ones(self.num_agents, dtype=bool)
        yell_info, valid_yell_info = {}, {}
        vocab_info = {word: {} for word in self.vocab}
        valid_vocab = {word: {} for word in self.vocab}
        for aid, agent in enumerate(self.cont_agents):
            dones[agent] = dones['__all__']
            dones[agent + '_comm'] = dones[agent]
            rewards[agent] = reward
            rewards[agent + '_comm'] = reward
            observations[agent] = obs_array[aid]
            observations[agent + '_comm'] = obs_array[aid]
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
            rgdist=jnp.clip(rgdist,0,1e8) * (rgdist < jnp.inf) + state.rgdist * (rgdist >= jnp.inf),
            success=all_success.astype(bool),
            agent_locs=new_agent_locs,
            agent_map=agent_map,
            doors=state.doors,
            cl_doors=state.cl_doors,
            door_map=state.door_map,
            keys=state.keys,
            decoy_keys=state.decoy_keys,
            cl_keys=state.cl_keys,
            key_map=state.key_map,
            goal=state.goal,
            goal_map=state.goal_map,
            doors_state=doors_state,
            observation=obs_array,
            raw_comm=raw_comm,
            comm_actions=comm_actions,
            valid_comm=valid_comm,
            rndid=state.rndid,
            ds_keys=state.ds_keys,
        )
        return observations, new_state, rewards, dones, infos

    def render(self, state: State, render_images=None):
        if not self.render_init:
            render_images = self.setup_rendering()
        if self.render_mode == 'human':
            _render(self, state, render_mode=self.render_mode, render_images=render_images)
            return render_images
        else:
            return _render(self, state, render_mode=self.render_mode, render_images=render_images), render_images

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


class MultiMazeLogWrapper(JaxMARLWrapper):
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
            success_episode=jnp.zeros((self._env.num_agents,)),
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
            success_episode= new_success_episode * (1 - ep_done),
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
