"""
Based on PureJaxRL Implementation of IPPO, with changes to give a centralised critic and communication policy.
"""
import os
import math
from datetime import datetime
from flax.serialization import to_bytes, from_bytes
from copy import deepcopy
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Tuple, Union, Dict
import wandb
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import DictConfig, OmegaConf
from functools import partial
from jaxmarl.environments.multi_maze.multi_categorical_distribution import MultiCategorical, \
    _kl_divergence_multicategorical
from distrax._src.distributions.categorical import _kl_divergence_categorical_categorical
from jaxmarl.environments.multi_maze import MultiMaze, MultiMazeLogWrapper
from jaxmarl.viz.window import Window
import pickle
from matplotlib import image
from flax.training import train_state

class ScannedRNN(nn.Module):
    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))

class PositionalEncoding(nn.Module):
    dim: int
    max_len: int
    def setup(self):
        pe = np.zeros((self.max_len, self.dim))
        pos = np.arange(0, self.max_len, dtype=np.float32)[:, None]
        _div = np.exp(np.arange(0, self.dim, 2) * (-math.log(10000.0) / self.dim))
        pe[:, 0::2] = np.sin(pos * _div)
        pe[:, 1::2] = np.cos(pos * _div)
        self.pe = jnp.array(pe[None])

    def __call__(self, x):
        return x + self.pe[:, :x.shape[1]]

class FeedForward(nn.Module):
    dim: int
    @nn.compact
    def __call__(self, x, training:bool):
        xn = nn.tanh(x)
        out = nn.Dense(
            self.dim, kernel_init=orthogonal(np.sqrt(2)),
        )(xn)
        out = nn.relu(out)
        out = out + x
        return out

class AttentionBlock(nn.Module):
    num_heads: int
    @nn.compact
    def __call__(self, q, kv, training:bool):
        qn = nn.tanh(q)
        kvn = nn.tanh(kv)
        att_out = nn.MultiHeadAttention(num_heads=self.num_heads)(qn, kvn)
        out = att_out + q
        return out

class Transformer(nn.Module):
    attention_len: int
    attention_dim: int
    attention_num_heads: int
    @nn.compact
    def __call__(self, carry, x, training:bool):

        carry, rnn_state = carry
        in1, in2, resets = x
        att_in1 = nn.Dense(
            self.attention_dim, kernel_init=orthogonal(np.sqrt(2)), use_bias=False
        )(in1)
        att_in2 = nn.Dense(
            self.attention_dim, kernel_init=orthogonal(np.sqrt(2)), use_bias=False
        )(in2)
        att_in = jnp.concatenate([att_in1, att_in2], axis=-1)
        carry = (carry * (1 - resets.reshape(list(resets.shape) + [1, 1]).repeat(carry.shape[-2], axis=-2))).at[..., 0, :].set(att_in)
        att_in1, att_in2 = jnp.split(carry, 2, axis=-1)
        att_in1 = PositionalEncoding(self.attention_dim, self.attention_len)(att_in1)
        att_in2 = PositionalEncoding(self.attention_dim, self.attention_len)(att_in2)
        att_out1 = AttentionBlock(self.attention_num_heads)(att_in1, att_in1, training)
        att_out2 = AttentionBlock(self.attention_num_heads)(att_in2, att_in2, training)
        att_out1 = FeedForward(self.attention_dim)(att_out1, training)
        att_out2 = FeedForward(self.attention_dim)(att_out2, training)
        att_out = AttentionBlock(self.attention_num_heads)(att_out1, att_out2, training)
        out = nn.tanh(att_out)
        carry = jnp.roll(carry, 1, axis=-2)
        out = out[..., -1, :]
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_rnn_carry(out.shape[0], out.shape[1]),
            rnn_state,
        )
        new_rnn_state, out = nn.GRUCell(features=out.shape[1])(rnn_state, out)

        return (carry, new_rnn_state), out + att_out[...,-1, :]

    @staticmethod
    def initialize_rnn_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ScannedTransformer(nn.Module):
    attention_len: int
    attention_dim: int
    attention_num_heads: int
    @nn.compact
    def __call__(self, carry, x, training:bool):
        scan = nn.scan(
            Transformer,
            variable_broadcast=["params"],
            in_axes=(0, nn.broadcast),
            out_axes=0,
            split_rngs={"params": False}
        )
        is_initializing = False
        if is_initializing:
            return Transformer(
                self.attention_len,
                self.attention_dim,
                self.attention_num_heads,
                name="Transformer"
            )(carry, tuple([y[0] for y in x]), training)
        else:
            return scan(
                self.attention_len,
                self.attention_dim,
                self.attention_num_heads,
                name="Transformer"
            )(carry, x, training)

    @staticmethod
    def initialize_carry(attention_len, batch_size, hidden_size, hidden_rnn_size):
        return (jnp.zeros((batch_size, attention_len, hidden_size)), Transformer.initialize_rnn_carry(batch_size, hidden_rnn_size))

class CommunicationNet(nn.Module):
    config: Dict
    num_agents: int

    @nn.compact
    def __call__(self, raw_comm):
        _comm = raw_comm.reshape(list(raw_comm.shape[:-1]) + [self.num_agents, -1])
        proc_comm = nn.Dense(
            self.config["FC_DIM_SIZE"], use_bias=False, kernel_init=orthogonal(np.sqrt(2)),
        )(_comm)
        proc_comm = nn.relu(proc_comm)

        proc_comm = nn.Dense(
            self.config["COMM_DIM"], use_bias=False, kernel_init=orthogonal(0.01)
        )(proc_comm)
        proc_comm = nn.tanh(proc_comm)
        return proc_comm.reshape(list(raw_comm.shape[:-1]) + [-1])

class CommActorAttention(nn.Module):
    action_dim: Sequence[int]
    num_categoricals: int
    num_agents: int
    config: Dict

    @nn.compact
    def __call__(self, hidden, x, training: bool):
        obs, raw_comm, dones, yell_ids, word_ids = x
        proc_comm = CommunicationNet(num_agents=self.num_agents, config=self.config)(raw_comm)
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), use_bias=False
        )(obs)
        embedding = nn.relu(embedding)
        embedding = nn.Dense(
            self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(np.sqrt(2)), use_bias=False
        )(embedding)
        embedding = nn.tanh(embedding)
        proc_comm = nn.Dense(
            self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(np.sqrt(2)), use_bias=False
        )(proc_comm)
        rnn_in_ = jnp.concatenate((embedding, proc_comm), axis=-1)
        rnn_in = (rnn_in_, dones)
        hidden_, att_out = ScannedRNN()(hidden[1], rnn_in)
        hidden = (hidden[0], hidden_)
        att_out = nn.relu(att_out) + rnn_in_
        actor_mean = nn.Dense(
            self.action_dim * self.num_categoricals, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(att_out)


        chunked = jnp.split(actor_mean, self.num_categoricals, axis=-1)
        yell_ids = yell_ids.reshape(list(actor_mean.shape[:-1]) + [1])
        word_ids = word_ids.reshape(list(actor_mean.shape[:-1]) + [1])
        valid_word_ids = jnp.logical_or(jnp.logical_not(yell_ids), word_ids)

        valid_ids = [valid_word_ids.repeat(self.action_dim, axis=-1)] + (self.num_categoricals - 1) * [yell_ids.repeat(self.action_dim, axis=-1)]
        chunked = [vi * ch + (1 - vi) * jax.lax.stop_gradient(ch)  for ch, vi in zip(chunked, valid_ids)]
        pi = MultiCategorical(chunked)

        return hidden, pi


class ContActorAttention(nn.Module):
    action_dim: Sequence[int]
    num_agents: int
    config: Dict

    @nn.compact
    def __call__(self, hidden, x, training: bool):
        obs, raw_comm, dones = x
        proc_comm = CommunicationNet(num_agents=self.num_agents, config=self.config)(raw_comm)
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), use_bias=False
        )(obs)
        embedding = nn.relu(embedding)
        embedding = nn.Dense(
            self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(np.sqrt(2)), use_bias=False
        )(embedding)
        embedding = nn.tanh(embedding)

        proc_comm = nn.Dense(
            self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(np.sqrt(2)), use_bias=False
        )(proc_comm)
        rnn_in_ = jnp.concatenate((embedding, proc_comm), axis=-1)
        rnn_in = (rnn_in_, dones)
        hidden_, att_out = ScannedRNN()(hidden[1], rnn_in)
        hidden = (hidden[0], hidden_)
        att_out = nn.relu(att_out) + rnn_in_
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(att_out)
        action_logits = actor_mean

        pi = distrax.Categorical(logits=action_logits)
        return hidden, pi


class Critic(nn.Module):
    num_agents: int
    config: Dict

    @nn.compact
    def __call__(self, hidden, x, training: bool):
        world_state, raw_comm, dones = x
        proc_comm = CommunicationNet(num_agents=self.num_agents, config=self.config)(raw_comm)
        embedding = nn.Dense(
            self.config["CRITIC_FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), use_bias=False
        )(world_state)
        embedding = nn.relu(embedding)
        embedding = nn.Dense(
            self.config["CRITIC_FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), use_bias=False
        )(embedding)
        embedding = nn.tanh(embedding)

        att_in = (embedding, proc_comm, dones)
        hidden, att_out = ScannedTransformer(
            self.config['ATTENTION_LEN'], self.config['ATTENTION_DIM'], self.config['ATTENTION_NUM_HEADS']
        )(hidden, att_in, training)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            att_out
        )
        return hidden, jnp.squeeze(critic, axis=-1)

class Transition(NamedTuple):
    global_done: jnp.ndarray
    cont_done: jnp.ndarray
    comm_done: jnp.ndarray
    cont_action: jnp.ndarray
    comm_action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    cont_log_prob: jnp.ndarray
    comm_log_prob: jnp.ndarray
    cont_obs: jnp.ndarray
    comm_obs: jnp.ndarray
    centralized: jnp.ndarray
    communication: jnp.ndarray
    valid_comm_act: jnp.ndarray
    info: jnp.ndarray

models = (ContActorAttention, CommActorAttention, Critic)
pnames = ("cont_actor", "comm_actor", "critic")

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def save_model_parameters(config, params, dir_path):
    os.mkdir(dir_path)
    config_path = os.path.join(dir_path, "config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(config, f)

    for name, state in params.items():
        file = os.path.join(dir_path, name + ".npz")
        with open(file, "wb") as f:
            pbytes = to_bytes(state)
            f.write(pbytes)

def load_model_parameters(path):
    config_path = os.path.join(path, "config.pkl")
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    parameters = {}
    for model, name in zip(models, pnames):
        file = os.path.join(path, name + ".npz")
        with open(file, "rb") as f:
            param = from_bytes(model, f.read())
            parameters[name] = jax.tree_map(jnp.array, param)
    return config, parameters

def init_actors(env, config, num_envs):
    cont_actor_network = ContActorAttention(env.action_space(env.cont_agents[0]).n, env.num_agents, config=config)
    comm_actor_network = CommActorAttention(2, len(env.vocab) + 1, env.num_agents, config=config)
    cont_ac_init_hstate = ScannedTransformer.initialize_carry(config['ATTENTION_LEN'], num_envs * env.num_agents, 2 * config["ATTENTION_DIM"], 2 * config["GRU_HIDDEN_DIM"])
    comm_ac_init_hstate = ScannedTransformer.initialize_carry(config['ATTENTION_LEN'], num_envs * env.num_agents, 2 * config["ATTENTION_DIM"], 2 * config["GRU_HIDDEN_DIM"])
    return {'cont': cont_actor_network, 'comm': comm_actor_network}, {'cont': cont_ac_init_hstate, 'comm': comm_ac_init_hstate}

def init_env_actors(config, num_envs):
    conf = deepcopy(config["ENV_KWARGS"])
    conf['render_mode'] = 'rgb_array'
    env = MultiMaze(**conf)
    render_images = env.setup_rendering()
    actors, hstates = init_actors(env, config, num_envs)
    return env, actors, hstates, render_images

@struct.dataclass
class RenderData:
    agents: Dict[str, jnp.ndarray]
    actors: Dict[str, nn.Module]
    hstates: Dict[str, jnp.ndarray]
    actor_params: Dict[str, jnp.ndarray]
    obs: Dict[str, jnp.ndarray]
    done: Dict[str, jnp.ndarray]

def initialize_render(parameters, config, num_envs, key, width=7, show=False, env_actors=None):
    actor_params = {'cont': parameters['cont_actor'], 'comm': parameters['comm_actor']}
    if env_actors is None:
        env, actors, hstates, render_images = init_env_actors(config, num_envs)
    else:
        env, actors, hstates, render_images = env_actors

    reset_rng = jax.random.split(key, num_envs)
    obs, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
    render_window_kwargs = None
    _curr_view = []
    for i in range(num_envs):
        es = jax.tree.map(lambda x: x[i], env_state)
        _cv, render_images = env.render(es, render_images)
        _curr_view.append(_cv)
    if show:
        height = (env.WINDOW_SIZE[1] / env.WINDOW_SIZE[0]) * width
        render_window_kwargs = {'title':'Multi-Maze', 'figsize':(width, height)}
    done = {a: jnp.zeros(num_envs, dtype=bool) for a in env.agents}
    done['__all__'] = jnp.zeros(num_envs, dtype=bool)
    agents = {'cont': env.cont_agents, 'comm': env.comm_agents}
    render_data = RenderData(
        agents=agents,
        actors=actors,
        hstates=hstates,
        actor_params=actor_params,
        obs=obs,
        done=done,
    )
    return env, env_state, render_data, render_images, render_window_kwargs, _curr_view

def run_single_step(x, unused, statics):
    env, n, actors, deterministic = statics
    def _aux(x):
        env_state, obs, done, cont_return, comm_return, actor_params, hstates, key, all_done, l = x
        key, key_act1, key_act2, key_step = jax.random.split(key, 4)
        communication = obs["communication"][:, None, ...].repeat(env.num_agents, axis=1).swapaxes(0, 1)
        communication = communication.reshape((env.num_agents * n, -1))
        cont_in = (
            batchify(obs, env.cont_agents, env.num_agents * n)[np.newaxis, :],
            communication[np.newaxis, :],
            batchify(done, env.cont_agents, env.num_agents * n).squeeze()[np.newaxis, :],
        )
        comm_in = (
            batchify(obs, env.comm_agents, env.num_agents * n)[np.newaxis, :],
            communication[np.newaxis, :],
            batchify(done, env.comm_agents, env.num_agents * n).squeeze()[np.newaxis, :],
            jnp.zeros_like(communication[..., :1])[np.newaxis, :],
            jnp.zeros_like(communication[..., :1])[np.newaxis, :],
        )
        cont_ac_hstate, cont_pi = actors['cont'].apply(actor_params['cont'], hstates['cont'], cont_in, False)
        comm_ac_hstate, comm_pi = actors['comm'].apply(actor_params['comm'], hstates['comm'], comm_in, False)
        if deterministic:
            cont_action = jnp.argmax(cont_pi.logits, axis=-1)
            comm_action = jnp.argmax(comm_pi.logits, axis=-1)
        else:
            cont_action = cont_pi.sample(seed=key_act1)
            comm_action = comm_pi.sample(seed=key_act2)
        cont_env_act = unbatchify(
            cont_action, env.cont_agents, n, env.num_agents
        )
        comm_env_act = unbatchify(
            comm_action, env.comm_agents, n, env.num_agents
        )
        actions = {k: v.squeeze() for k, v in cont_env_act.items()}
        actions.update({k: v.squeeze() for k, v in comm_env_act.items()})
        hstates = {'cont': cont_ac_hstate, 'comm': comm_ac_hstate}
        rng_step = jax.random.split(key_step, n)
        obs, env_state, reward, done, infos = jax.vmap(
            env.step, in_axes=(0, 0, 0)
        )(rng_step, env_state, actions)
        cont_return = cont_return + (1 - all_done).reshape(reward['red_player'].shape) * reward['red_player']
        comm_return = comm_return + (1 - all_done).reshape(reward['red_player_comm'].shape) * reward['red_player_comm']
        all_done = jnp.logical_or(all_done, done["__all__"])
        l = l + (1 - all_done)

        return (env_state, obs, done, cont_return, comm_return, actor_params, hstates, key, all_done, l), (cont_return, comm_return, env_state, l)
    return _aux(x)

def _n_step_scan(statics):
    single_step = partial(run_single_step, statics=statics)
    def scan_n_steps(carry, max_len):
        _, data = jax.lax.scan(single_step, carry, None, max_len)
        return data
    return jax.jit(scan_n_steps, static_argnums=[1])


def run_n_episodes(env, env_state, render_data, n, render_images, key, deterministic=False, scan_func=None):

    agents, actors, hstates, actor_params, obs, done = (
        render_data.agents,
        render_data.actors,
        render_data.hstates,
        render_data.actor_params,
        render_data.obs,
        render_data.done,
    )

    statics = (env, n, actors, deterministic)
    carry = (
        env_state, obs, done, jnp.zeros((n,)), jnp.zeros((n,)), actor_params, hstates, key, jnp.zeros_like(done['__all__']), jnp.zeros(n, dtype=int)
    )
    if scan_func is None:
        scan_func = _n_step_scan(statics)
    data = scan_func(carry, env.max_episode_len)
    trajectories = []
    for i in range(n):
        states = jax.tree.map(lambda x: x[:, i], data[2])
        traj = []
        for j in jnp.arange(data[3][-1, i]):
            es = jax.tree.map(lambda x: x[j], states)
            _cv, _ = env.render(es, render_images=render_images)
            traj.append(_cv)
        trajectories.append(traj)

    return trajectories, data[0][-1].mean(), data[1][-1].mean()

def make_train(config):
    env = MultiMaze(**config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
            2 * ((config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]) // 2)
    )
    config["MINIBATCH_SIZE"] = (
            config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["SAVE_INTERVAL"] = int(config["SAVE_FREQ"] * config["NUM_UPDATES"])
    config["CLIP_EPS_CONT"] = (
        config["CLIP_EPS_CONT"] / env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS_CONT"]
    )
    config["CLIP_EPS_COMM"] = (
        config["CLIP_EPS_COMM"] / env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS_COMM"]
    )
    env = MultiMazeLogWrapper(env, replace_info=True)
    render_env_actors = init_env_actors(config, config["NUM_EVAL_TRAJECTORIES"])
    statics = (render_env_actors[0], config["NUM_EVAL_TRAJECTORIES"], render_env_actors[1], False)
    render_scan_func = _n_step_scan(statics)

    def linear_schedule_comm(count):
        frac = (
                1.0
                - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
                / config["NUM_UPDATES"]
        )
        return config["LR_COMM"] * frac

    def linear_schedule_cont(count):
        frac = (
                1.0
                - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
                / config["NUM_UPDATES"]
        )
        return config["LR_CONT"] * frac

    def train(rng, dir_path):
        # INIT NETWORK
        cont_actor_network = ContActorAttention(env.action_space(env.cont_agents[0]).n, env.num_agents, config=config)
        comm_actor_network = CommActorAttention(2, len(env.vocab) + 1, env.num_agents, config=config)
        critic_network = Critic(env.num_agents, config=config)
        rng, _rng_cont_actor, _rng_comm_actor, _rng_cont_communication, _rng_comm_communication, _rng_critic = jax.random.split(
            rng, 6)
        ac_init_x = (
            jnp.zeros((1, env.num_agents * config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])),
            jnp.zeros((1, env.num_agents * config["NUM_ENVS"], env.communication_size())),
            jnp.zeros((1, env.num_agents * config["NUM_ENVS"])),
        )
        comm_ac_init_x = (
            jnp.zeros((1, env.num_agents * config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])),
            jnp.zeros((1, env.num_agents * config["NUM_ENVS"], env.communication_size())),
            jnp.zeros((1, env.num_agents * config["NUM_ENVS"])),
            jnp.zeros((1, env.num_agents * config["NUM_ENVS"])),
            jnp.zeros((1, env.num_agents * config["NUM_ENVS"])),
        )
        cont_ac_init_hstate = ScannedTransformer.initialize_carry(config['ATTENTION_LEN'], config["NUM_ENVS"] * env.num_agents, 2 * config["ATTENTION_DIM"], 2 * config["GRU_HIDDEN_DIM"])
        comm_ac_init_hstate = ScannedTransformer.initialize_carry(config['ATTENTION_LEN'], config["NUM_ENVS"] * env.num_agents, 2 * config["ATTENTION_DIM"], 2 * config["GRU_HIDDEN_DIM"])
        cont_actor_network_params = cont_actor_network.init(_rng_cont_actor, cont_ac_init_hstate, ac_init_x, False)
        comm_actor_network_params = comm_actor_network.init(_rng_comm_actor, comm_ac_init_hstate, comm_ac_init_x, False)
        cr_init_x = (
            jnp.zeros((1, env.num_agents * config["NUM_ENVS"], env.centralized_size(),)),
            jnp.zeros((1, env.num_agents * config["NUM_ENVS"], env.communication_size())),
            jnp.zeros((1, env.num_agents * config["NUM_ENVS"])),
        )
        cr_init_hstate = ScannedTransformer.initialize_carry(config['ATTENTION_LEN'], config["NUM_ENVS"] * env.num_agents, 2 * config["ATTENTION_DIM"], config["GRU_HIDDEN_DIM"])
        critic_network_params = critic_network.init(_rng_critic, cr_init_hstate, cr_init_x, False)

        if config["ANNEAL_LR"]:
            cont_actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM_AC"]),
                optax.adam(learning_rate=linear_schedule_cont, eps=1e-5),
            )
            comm_actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM_AC"]),
                optax.adam(learning_rate=linear_schedule_comm, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM_CR"]),
                optax.adam(learning_rate=linear_schedule_cont, eps=1e-5),
            )
        else:
            cont_actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM_AC"]),
                optax.adam(config["LR_CONT"], eps=1e-5),
            )
            comm_actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM_AC"]),
                optax.adam(config["LR_COMM"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM_CR"]),
                optax.adam(config["LR_CONT"], eps=1e-5),
            )
        cont_actor_train_state = TrainState.create(
            apply_fn=cont_actor_network.apply,
            params=cont_actor_network_params,
            tx=cont_actor_tx,
        )
        comm_actor_train_state = TrainState.create(
            apply_fn=comm_actor_network.apply,
            params=comm_actor_network_params,
            tx=comm_actor_tx,
        )
        critic_train_state = train_state.TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(update_runner_state, comm_update):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            old_train_states = runner_state[0]
            kl_coefs = runner_state[-2]
            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, _last_done, hstates, kl_coefs, rng = runner_state
                last_cont_done, last_comm_done = _last_done
                # SELECT ACTION
                rng, _rng_cont, _rng_comm = jax.random.split(rng, 3)
                cont_obs_batch = batchify(last_obs, env.cont_agents, config["NUM_ACTORS"])
                comm_obs_batch = batchify(last_obs, env.comm_agents, config["NUM_ACTORS"])
                communication = last_obs["communication"][:, None, ...].repeat(env.num_agents, axis=1).swapaxes(0, 1)
                communication = communication.reshape((config["NUM_ACTORS"], -1))
                cont_ac_in = (
                    cont_obs_batch[np.newaxis, :],
                    communication[np.newaxis, :],
                    last_cont_done[np.newaxis, :],
                )
                comm_ac_in = (
                    comm_obs_batch[np.newaxis, :],
                    communication[np.newaxis, :],
                    last_comm_done[np.newaxis, :],
                    jnp.zeros_like(last_comm_done[np.newaxis, :]),
                    jnp.zeros_like(last_comm_done[np.newaxis, :]),
                )
                cont_ac_hstate, cont_pi= cont_actor_network.apply(train_states[0].params, hstates[0], cont_ac_in, False)
                comm_ac_hstate, comm_pi= comm_actor_network.apply(train_states[1].params, hstates[1], comm_ac_in, False)
                cont_action = cont_pi.sample(seed=_rng_cont)
                comm_action = comm_pi.sample(seed=_rng_comm)
                cont_log_prob = cont_pi.log_prob(cont_action)
                comm_log_prob = comm_pi.log_prob(comm_action)
                cont_env_act = unbatchify(
                    cont_action, env.cont_agents, config["NUM_ENVS"], env.num_agents
                )
                comm_env_act = unbatchify(
                    comm_action, env.comm_agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in cont_env_act.items()}
                env_act.update({k: v.squeeze() for k, v in comm_env_act.items()})

                # VALUE
                # output of wrapper is (num_envs, num_agents, world_state_size)
                # swap axes to (num_agents, num_envs, world_state_size) before reshaping to (num_actors, world_state_size)
                centralized = last_obs["centralized"].swapaxes(0, 1)
                centralized = centralized.reshape((config["NUM_ACTORS"], -1))
                valid_comm_act = last_obs["valid_comm"].swapaxes(0, 1)
                valid_comm_act = valid_comm_act.reshape((config["NUM_ACTORS"], -1))

                cr_in = (
                    centralized[None, :],
                    communication[np.newaxis, :],
                    last_cont_done[np.newaxis, :],
                )
                cr_hstate, value = critic_network.apply(train_states[2].params, hstates[2], cr_in, False)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                cont_done_batch = batchify(done, env.cont_agents, config["NUM_ACTORS"]).squeeze()
                comm_done_batch = batchify(done, env.comm_agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    global_done=jnp.tile(done["__all__"], env.num_agents),
                    cont_done=last_cont_done,
                    comm_done=last_comm_done,
                    cont_action=cont_action.squeeze(),
                    comm_action=comm_action.squeeze(),
                    value=value.squeeze(),
                    reward=batchify(reward, env.cont_agents, config["NUM_ACTORS"]).squeeze(),
                    cont_log_prob=cont_log_prob.squeeze(),
                    comm_log_prob=comm_log_prob.squeeze(),
                    cont_obs=cont_obs_batch,
                    comm_obs=comm_obs_batch,
                    centralized=centralized,
                    communication=communication,
                    valid_comm_act=valid_comm_act,
                    info=info,
                )
                runner_state = (train_states, env_state, obsv, (cont_done_batch, comm_done_batch),
                                (cont_ac_hstate, comm_ac_hstate, cr_hstate), kl_coefs, rng)
                return runner_state, transition

            initial_hstates = runner_state[-3]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, _last_done, hstates, _, rng = runner_state
            last_cont_done, last_comm_done = _last_done
            last_centralized = last_obs["centralized"].swapaxes(0, 1)
            last_centralized = last_centralized.reshape((config["NUM_ACTORS"], -1))

            communication = last_obs["communication"][:, None, ...].repeat(env.num_agents, axis=1).swapaxes(0, 1)
            communication = communication.reshape((config["NUM_ACTORS"], -1))
            cr_in = (
                last_centralized[None, :],
                communication[np.newaxis, :],
                last_cont_done[np.newaxis, :],
            )
            _, last_val = critic_network.apply(train_states[2].params, hstates[2], cr_in, False)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                            delta
                            + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                kl_coefs = update_state[-3]
                def _update_minbatch(train_states, batch_info):
                    cont_actor_train_state, comm_actor_train_state, critic_train_state, old_train_states = train_states
                    cont_ac_init_hstate, comm_ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets, kl_coefs = batch_info
                    kl_coef_cont, kl_coef_comm = kl_coefs

                    def _cont_actor_loss_fn(actor_params, init_hstate, traj_batch, gae, kl_coef, old_params):
                        # RERUN NETWORK
                        _, pi = cont_actor_network.apply(
                            actor_params,
                            (init_hstate[0].squeeze(), init_hstate[1].squeeze()),
                            (traj_batch.cont_obs, traj_batch.communication, traj_batch.cont_done),
                            True,
                        )
                        log_prob = pi.log_prob(traj_batch.cont_action)

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.cont_log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config["CLIP_EPS_CONT"],
                                    1.0 + config["CLIP_EPS_CONT"],
                                    )
                                * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        _, old_pi = cont_actor_network.apply(
                            old_params,
                            (init_hstate[0].squeeze(), init_hstate[1].squeeze()),
                            (traj_batch.cont_obs, traj_batch.communication, traj_batch.cont_done),
                            False,
                        )
                        old_pi = jax.lax.stop_gradient(old_pi)
                        kl = jax.jit(_kl_divergence_categorical_categorical)(old_pi, pi).mean()
                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS_CONT"])

                        actor_loss = loss_actor - config["ENT_COEF_CONT"] * entropy + kl_coef * kl
                        return actor_loss, (loss_actor, entropy, ratio, kl, clip_frac)


                    def _critic_loss_fn(critic_params, init_hstate, traj_batch, targets):
                        # RERUN NETWORK
                        ca = traj_batch.communication
                        _, value = critic_network.apply(critic_params, (init_hstate[0].squeeze(), init_hstate[1].squeeze()),
                                                        (traj_batch.centralized, ca, traj_batch.cont_done), True)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                        ).clip(-config["CLIP_EPS_CONT"], config["CLIP_EPS_CONT"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                                0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, (value_loss)

                    cont_actor_grad_fn = jax.value_and_grad(_cont_actor_loss_fn, has_aux=True)
                    cont_actor_loss, cont_actor_grads = cont_actor_grad_fn(
                        cont_actor_train_state.params, cont_ac_init_hstate, traj_batch, advantages, kl_coef_cont, old_train_states[0].params
                    )
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params, cr_init_hstate, traj_batch, targets
                    )
                    _comm_update = comm_update.astype(jnp.bool)
                    cont_mask = jnp.logical_or(jnp.logical_not(_comm_update), config['SIMUL'] * jnp.ones_like(_comm_update))[0]

                    new_cont_actor_train_state = cont_actor_train_state.apply_gradients(grads=cont_actor_grads)
                    critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)

                    cont_actor_train_state = cont_actor_train_state.replace(
                        step=cont_mask * new_cont_actor_train_state.step + (
                                1 - cont_mask) * cont_actor_train_state.step,
                        params=jax.tree.map(lambda nx, ox: cont_mask * nx + (1 - cont_mask) * ox,
                                            new_cont_actor_train_state.params, cont_actor_train_state.params),
                        opt_state=jax.tree.map(lambda nx, ox: cont_mask * nx + (1 - cont_mask) * ox,
                                               new_cont_actor_train_state.opt_state, cont_actor_train_state.opt_state),
                    )

                    cont_loss = cont_actor_loss[0] + critic_loss[0]
                    comm_loss = cont_loss
                    total_loss = comm_loss + cont_loss
                    loss_info = {
                        "total_loss": total_loss,
                        "cont_loss": cont_loss,
                        "comm_loss": comm_loss,
                        "cont_actor_loss": cont_actor_loss[0],
                        "comm_actor_loss": cont_actor_loss[0],
                        "value_loss": critic_loss[0],
                        "cont_entropy": cont_actor_loss[1][1],
                        "comm_entropy": cont_actor_loss[1][1],
                        "cont_ratio": cont_actor_loss[1][2],
                        "comm_ratio": cont_actor_loss[1][2],
                        "approx_cont_kl": cont_actor_loss[1][3],
                        "approx_comm_kl": cont_actor_loss[1][3],
                        "cont_clip_frac": cont_actor_loss[1][4],
                        "comm_clip_frac": cont_actor_loss[1][4],
                    }

                    return (cont_actor_train_state,
                            comm_actor_train_state,
                            critic_train_state,
                            old_train_states), loss_info

                (
                    train_states,
                    init_hstates,
                    traj_batch,
                    advantages,
                    targets,
                    _,
                    old_train_states,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                init_hstates = jax.tree.map(lambda x: jnp.reshape(
                    x, (1, config["NUM_ACTORS"], config["ATTENTION_LEN"], -1)
                ), init_hstates)
                init_hstates = tuple([(ihs[0], ihs[1].reshape((1, config["NUM_ACTORS"], -1))) for ihs in init_hstates])
                batch = (
                    init_hstates[0],
                    init_hstates[1],
                    init_hstates[2],
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                            ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )
                minibatches = tuple(list(minibatches) + [(kl_coefs[0] * jnp.ones((config["NUM_MINIBATCHES"],)), kl_coefs[1] * jnp.ones((config["NUM_MINIBATCHES"],)))])

                train_states, loss_info = jax.lax.scan(
                    _update_minbatch, (train_states[0], train_states[1], train_states[2] ,old_train_states), minibatches
                )
                train_states = train_states[:-1]
                update_state = (
                    train_states,
                    jax.tree.map(lambda x: x.squeeze(), init_hstates),
                    traj_batch,
                    advantages,
                    targets,
                    kl_coefs,
                    old_train_states,
                    rng,
                )
                return update_state, loss_info

            update_state = (
                train_states,
                initial_hstates,
                traj_batch,
                advantages,
                targets,
                kl_coefs,
                old_train_states,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            loss_info["cont_ratio_0"] = loss_info["cont_ratio"].at[0, 0].get()
            loss_info["comm_ratio_0"] = loss_info["comm_ratio"].at[0, 0].get()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            kl_coef_cont = jnp.clip(((loss_info["approx_cont_kl"] > 2.0 * config["KL_TARGET_CONT"]) * 1.5 * kl_coefs[0] +
                                     (loss_info["approx_cont_kl"] < 0.5 * config["KL_TARGET_CONT"]) * 0.5 * kl_coefs[0]),config["MIN_KL_COEF"], config["MAX_KL_COEF"])
            kl_coef_comm = jnp.clip(((loss_info["approx_comm_kl"] > 2.0 * config["KL_TARGET_COMM"]) * 1.5 * kl_coefs[1] +
                                     (loss_info["approx_comm_kl"] < 0.5 * config["KL_TARGET_COMM"]) * 0.5 * kl_coefs[1]), config["MIN_KL_COEF"], config["MAX_KL_COEF"])
            kl_coefs = (kl_coef_cont, kl_coef_comm)
            train_states = update_state[0]
            metric = jax.tree.map(
                lambda x: x.reshape(
                    (config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)
                ),
                traj_batch.info,
            )

            metric["loss"] = loss_info
            metric["loss"]["kl_coef_cont"] = kl_coef_cont
            metric["loss"]["kl_coef_comm"] = kl_coef_comm
            rng = update_state[-1]

            def callback(metric, model_states):
                log_dict = {
                    # the metrics have an agent dimension, but this is identical
                    # for all agents so index into the 0th item of that dimension.
                    "episode_lengths": metric["returned_episode_lengths"][:, :, 0][
                        metric["returned_episode"][:, :, 0]
                    ].mean(),
                    "cont_returns": metric["returned_cont_episode_returns"][:, :, 0][
                        metric["returned_episode"][:, :, 0]
                    ].mean(),
                    "comm_returns": metric["returned_comm_episode_returns"][:, :, 0][
                        metric["returned_episode"][:, :, 0]
                    ].mean(),
                    "success": metric["returned_success_episode"][:, :, 0][
                        metric["returned_episode"][:, :, 0]
                    ].mean(),
                    "yells": metric["returned_episode_yells"][:, :, 0][
                        metric["returned_episode"][:, :, 0]
                    ].mean(),
                    "valid_yells": metric["returned_episode_valid_yells"][:, :, 0][
                        metric["returned_episode"][:, :, 0]
                    ].mean(),
                    "env_step": metric["update_steps"]
                                * config["NUM_ENVS"]
                                * config["NUM_STEPS"],
                    **metric["loss"],
                }
                _ids = metric["returned_episode"][:, :, 0]
                log_dict.update({
                    "vocab_" + k: metric["returned_episode_vocab"][k][:, :, 0][
                        metric["returned_episode"][:, :, 0]
                    ].mean() for k in env.vocab})
                log_dict.update({
                    "valid_vocab_" + k: metric["returned_episode_valid_vocab"][k][:, :, 0][
                        metric["returned_episode"][:, :, 0]
                    ].mean() for k in env.vocab})
                wandb.log(log_dict)
                return

            metric["update_steps"] = update_steps
            metric["rng"] = rng
            def callback_(metric, train_states):
                return jax.experimental.io_callback(callback, None, metric, train_states)

            def save_callback(metric, train_states):
                parameters = {name: state.params for name, state in zip(pnames, train_states)}
                strtime = datetime.now().strftime("%Y%m%d-%H%M%S")
                checkpoint_path = os.path.join(dir_path, "checkpoint_" + strtime)
                print('checkpoint_path: ', checkpoint_path)
                save_model_parameters(config, parameters, checkpoint_path)
                key, subkey = jax.random.split(metric["rng"])
                _env, env_state, render_data, render_images, render_window, first_render = initialize_render(
                    parameters, config, config["NUM_EVAL_TRAJECTORIES"], subkey, env_actors=render_env_actors,
                )
                trajectories, cont_return, comm_return = run_n_episodes(
                    _env, env_state, render_data, config["NUM_EVAL_TRAJECTORIES"], render_images, key, scan_func=render_scan_func
                )
                print("mean cont return: ", cont_return)
                print("mean comm return: ", comm_return)
                render_path = os.path.join(checkpoint_path, "renders")
                os.mkdir(render_path)
                for jj, trajectory in enumerate(trajectories):
                    trj_path = os.path.join(render_path, "trajectory_" + str(jj + 1))
                    os.mkdir(trj_path)
                    image.imsave(os.path.join(trj_path, "0.png"), first_render[jj])
                    for j, im in enumerate(trajectory):
                        image.imsave(os.path.join(trj_path, str(j + 1) + ".png"), im)
                return
            def no_callback(metric, train_states):
                return

            def callback_save_(metric, train_states):
                return jax.experimental.io_callback(save_callback, None, metric, train_states)

            jax.lax.cond(update_steps % config["SAVE_INTERVAL"] == 0, callback_save_, no_callback, metric, train_states)
            jax.lax.cond(update_steps % config["LOG_INTERVAL"] == 0, callback_, no_callback, metric, train_states)

            update_steps = update_steps + 1
            runner_state = (train_states, env_state, last_obs, (last_cont_done, last_comm_done), hstates, kl_coefs, rng)
            return (runner_state, update_steps), jnp.zeros(1)

        rng, _rng = jax.random.split(rng)
        runner_state = ((
                            (cont_actor_train_state, comm_actor_train_state, critic_train_state),
                            env_state,
                            obsv,
                            (jnp.zeros((config["NUM_ACTORS"]), dtype=bool), jnp.zeros((config["NUM_ACTORS"]), dtype=bool)),
                            (cont_ac_init_hstate, comm_ac_init_hstate, cr_init_hstate),
                            (config["KL_COEF_CONT"] * jnp.ones((1,)), config["KL_COEF_COMM"] * jnp.ones((1,))),
                            _rng,
                        ),
                        0
        )
        _comm_update_rate = jnp.concatenate((jnp.ones((int(config["NUM_UPDATES"] // (config["CONT_INTERVAL"] + config["COMM_INTERVAL"])), config["COMM_INTERVAL"])), jnp.zeros((int(config["NUM_UPDATES"] // (config["CONT_INTERVAL"] + config["COMM_INTERVAL"])), config["CONT_INTERVAL"]))), axis=1).reshape((-1, 1))
        comm_update_rate = jnp.zeros((int(config["NUM_UPDATES"]), 1))
        comm_update_rate = comm_update_rate.at[:len(_comm_update_rate)].set(_comm_update_rate)
        comm_update_rate = comm_update_rate.at[len(_comm_update_rate):].set(_comm_update_rate[:int(config["NUM_UPDATES"]) - len(_comm_update_rate)])
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, comm_update_rate, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}
    return train

@hydra.main(version_base=None, config_path="config", config_name="nocomm_multimaze")
def main(config):
    config = OmegaConf.to_container(config)

    wandb.init(
        project=config["PROJECT"],
        tags=["MAPPO", "ATTENTION"],
        config=config,
    )
    n_traj = 10
    strtime = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_path = os.path.join(os.path.dirname(__file__), config["SAVE_PATH"])
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    dir_path = os.path.join(dir_path, "model_" + strtime)
    os.mkdir(dir_path)

    rng = jax.random.PRNGKey(config["SEED"])
    out = jax.jit(make_train(config), static_argnums=[1])(rng, dir_path)
    parameters = {name: state.params for name, state in zip(pnames, out["runner_state"][0][0])}
    checkpoint_path = os.path.join(dir_path, "checkpoint_last")
    print('checkpoint_path: ', checkpoint_path)
    save_model_parameters(
        config,
        parameters,
        checkpoint_path
    )
    key, subkey = jax.random.split(rng)
    _env, env_state, render_data, render_images, render_window, first_render = initialize_render(
        parameters, config, config["NUM_EVAL_TRAJECTORIES"], subkey
    )
    trajectories, cont_return, comm_return = run_n_episodes(
        _env, env_state, render_data, config["NUM_EVAL_TRAJECTORIES"], render_images, key
    )
    print("mean cont return: ", cont_return)
    print("mean comm return: ", comm_return)
    render_path = os.path.join(checkpoint_path, "renders")
    os.mkdir(render_path)
    for jj, trajectory in enumerate(trajectories):
        trj_path = os.path.join(render_path, "trajectory_" + str(jj + 1))
        os.mkdir(trj_path)
        image.imsave(os.path.join(trj_path, "0.png"), first_render[jj])
        for j, im in enumerate(trajectory):
            image.imsave(os.path.join(trj_path, str(j + 1) + ".png"), im)

if __name__ == "__main__":
    main()
