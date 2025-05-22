import numpy as np
import pickle
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import itertools
from tqdm import tqdm
from os import path
from copy import deepcopy
import jax.numpy as jnp

colors = ('red', 'blue', 'green', 'orange', 'brown', 'pink', 'purple', 'yellow')

object_words = ('wall', 'door', 'key', 'goal', 'agent')

def initial_state_distribution(keys_rooms, goal_rooms, rndid_classes, balance=True):
    cls_prob = 1.0 / len(rndid_classes)
    probs = []
    rnd_rooms = []
    rnd_ids = []
    for rcls in rndid_classes:
        _clsp = cls_prob * (1.0 / len(rcls))
        for rcl in rcls:
            rndid, rndr = rcl
            rnd_rooms.append(rndr)
            probs.append(_clsp)
            rnd_ids.append(rndid)
    probs = np.array(probs)
    if not balance:
        probs = np.ones_like(probs) / len(probs)
    rnd_ids = np.array(rnd_ids, dtype=int)
    rnd_rooms = np.array(rnd_rooms, dtype=int)
    all_rooms = keys_rooms + goal_rooms
    room_ids = []
    room_lens = []
    for i, rooms in enumerate(all_rooms):
        _room_ids = (-1) * np.ones((len(rnd_rooms), len(rooms)), dtype=int)
        _room_lens = (-1) * np.ones((len(rnd_rooms),), dtype=int)
        for j, rndrs in enumerate(rnd_rooms):
            mid = np.arange(len(rooms))[rooms == rndrs[i]]
            _room_ids[j, :len(mid)] = mid
            _room_lens[j] = len(mid)
        room_ids.append(_room_ids)
        room_lens.append(_room_lens)


    return {'probs': probs, 'rndid': rnd_ids, 'ids': room_ids, 'lens': room_lens}

def make_seq_sample(keys_rooms, goal_rooms, solvable, n_rooms, rs2rndid):
    all_loc_map = [np.arange(len(keys_rooms[0])).reshape(1, -1)]
    all_len_map = [len(keys_rooms[0]) * np.ones((1,))]
    map_ids = [(-1) * np.ones((1, n_rooms), dtype=int)]
    prevs = 1
    curr_seqs = [[tuple()]]
    i = 0
    all_rooms = keys_rooms + goal_rooms
    for kr in keys_rooms:
        prev_sqs = [list(cs) for cs in curr_seqs[-1]]
        ukrs = list(set(kr))
        new_sqs = prev_sqs * len(ukrs)
        currs = len(ukrs) * prevs
        next_rooms = all_rooms[i + 1]
        loc_map = (-1) * np.ones((currs, len(next_rooms)), dtype=int)
        len_map = (-1) * np.ones((currs,), dtype=int)
        for j in range(prevs):
            for jj, _kr  in enumerate(ukrs):
                nid = jj * len(prev_sqs) + j
                map_ids[-1][j, _kr] = nid
                new_sqs[nid] = tuple(new_sqs[nid] + [_kr])
                tmp = solvable[:,:len(new_sqs[nid])] == np.array(new_sqs[nid], dtype=int)
                tmp = solvable[np.all(tmp, axis=1)][:,len(new_sqs[nid])]
                valid_next_rooms = np.unique(tmp.flatten())
                valid_next_id = []
                for ri, nr in enumerate(next_rooms):
                    if np.any((valid_next_rooms == nr)):
                        valid_next_id.append(ri)

                loc_map[nid, :len(valid_next_id)] = np.array(valid_next_id, dtype=int)
                len_map[nid] = len(valid_next_id)
        curr_seqs.append(new_sqs)
        map_ids.append((-1) * np.ones((currs, n_rooms), dtype=int))
        all_loc_map.append(loc_map)
        all_len_map.append(len_map)
        for lom, lem in zip(loc_map, len_map):
            assert np.all(lom[:lem] >= 0)

        prevs = currs
        i += 1
    grs = list(set(goal_rooms[0]))
    prev_sqs = [list(cs) for cs in curr_seqs[-1]]
    for j in range(prevs):
        for _kr in grs:
            map_ids[-1][j, _kr] = rs2rndid[tuple(prev_sqs[j] + [_kr])]

    return all_loc_map, all_len_map, map_ids

def numpify_dict(d, key_length, key_support):
    k_arr = [[] for _ in range(key_length)]
    for k in d.keys():
        for i in range(key_length):
            k_arr[i].append(k[i])

    arr_list = [(key_support + 1) * np.ones((1, key_support), dtype=int)]
    prevs = 1
    curr_seqs = [[tuple()]]
    arrs_dims = [1]
    for i, ks in enumerate(k_arr):
        _ks = list(set(ks))
        prev_sqs = [list(cs) for cs in curr_seqs[-1]]
        if i < len(k_arr) - 1:
            new_sqs = prev_sqs * len(_ks)
            for j in range(prevs):
                for jj, kk in enumerate(_ks):
                    nid = jj * len(prev_sqs) + j
                    arr_list[-1][j, kk] = nid

                    new_sqs[nid] = tuple(new_sqs[nid] + [kk])
            curr_seqs.append(new_sqs)
            arrs_dims.append(arrs_dims[-1] * len(_ks))
            prevs = arrs_dims[-1]
            arr_list.append((-1) * np.ones((arrs_dims[-1], key_support), dtype=int))
        else:
            for j in range(prevs):
                for g in _ks:
                    arr_list[-1][j, g] = d[tuple(prev_sqs[j] + [g])]
    max_val = max(arr_list[-1].shape[0], len(d))
    arr_list = [(max_val + 2) * (ar < 0) + ar * (ar >= 0) for ar in arr_list]
    return arr_list

def get_graph_representation(map_size, wall_map, door_map):
    graph = np.zeros((map_size**2, map_size**2))
    for i in range(map_size):
        for j in range(map_size):
            graph[i * map_size + j, i * map_size + j] = 0.1
            if not (wall_map[i,j] or door_map[:,i,j].any()):
                if i > 0:
                    graph[(i - 1) * map_size + j, i * map_size + j] = 1
                if i < map_size - 1:
                    graph[(i + 1) * map_size + j, i * map_size + j] = 1
                if j > 0:
                    graph[i * map_size + j - 1, i * map_size + j] = 1
                if j < map_size - 1:
                    graph[i * map_size + j + 1, i * map_size + j] = 1
    return graph


def get_rooms(map_size, doors, walls, valid_locs):
    door_map = np.zeros((len(doors), map_size, map_size), dtype=np.float32)
    for i, door in enumerate(doors):
        door_map[i, door[0], door[1]] = 1
    wall_map =  np.zeros((map_size, map_size), dtype=np.float32)
    for wall in walls:
        wall_map[wall[0], wall[1]] = 1
    dist = dijkstra(csr_matrix(get_graph_representation(map_size, wall_map, door_map)))
    id2room = {}
    id2room_agents = {}
    rooms = []
    pos_ids = [(i, j) for i in range(map_size) for j in range(map_size) if not wall_map[i,j] and not door_map[:,i,j].any()]
    while len(pos_ids) > 0:
        pos_id = pos_ids[0]
        ids = np.where(dist[pos_id[0] * map_size + pos_id[1], :] < np.inf)[0]
        ids_0 = ids // map_size
        ids_1 = ids % map_size
        ids = [(int(id0), int(id1)) for id0, id1 in zip(ids_0, ids_1)]
        rooms.append({'locs': np.array(ids, dtype=int), 'neighbors': [], 'eq_rooms': []})
        for idx in ids:
            id2room[idx] = len(rooms) - 1
            id2room_agents[idx] = len(rooms) - 1
            pos_ids.remove(idx)
    for doorid, door in enumerate(doors):
        ids = np.where(dist[door[0] * map_size + door[1], :] == 1)[0]
        ids_0 = ids // map_size
        ids_1 = ids % map_size
        ids = [(int(id0), int(id1)) for id0, id1 in zip(ids_0, ids_1)]
        neighbors = []
        id2room[tuple(door)] = []
        id2room_agents[tuple(door)] = []
        for roomid, room in enumerate(rooms):
            for idx in ids:
                if np.logical_and(room['locs'][:,0] == idx[0], room['locs'][:,1] == idx[1]).any():
                    neighbors.append(roomid)
                    id2room[tuple(door)].append(roomid)
                    id2room_agents[tuple(door)].append(roomid)
                    break
        if len(neighbors) == 2:
            rooms[neighbors[0]]['neighbors'].append({'room': neighbors[1], 'door': doorid})
            rooms[neighbors[1]]['neighbors'].append({'room': neighbors[0], 'door': doorid})
        if len(neighbors) == 1:
            rooms.append({'locs': door.reshape((1,2)).astype(int), 'neighbors': [{'room': neighbors[0], 'door': doorid}], 'eq_rooms': []})
            rooms[neighbors[0]]['neighbors'].append({'room': len(rooms) - 1, 'door': doorid})
            rooms[neighbors[0]]['eq_rooms'].append(len(rooms) - 1)
            id2room[tuple(door)] = len(rooms) - 1
            id2room_agents[tuple(door)] = [neighbors[0], len(rooms) - 1]

    reach_dist = dijkstra(csr_matrix(get_graph_representation(map_size, wall_map, np.zeros_like(door_map))))
    valid_rooms = []
    start_rooms = []
    for i, vl in enumerate(valid_locs):
        vr_ = []
        sr_ = []
        for _vl in vl:
            sr_.append(id2room[tuple(_vl)])
            for j, room in enumerate(rooms):
                if reach_dist[_vl[0] * map_size + _vl[1], room['locs'][0, 0] * map_size + room['locs'][0, 1]] < np.inf:
                    vr_.append(j)
        valid_rooms.append(list(set(vr_)))
        start_rooms.append(list(set(sr_)))

    return rooms, id2room, id2room_agents, wall_map, valid_rooms, start_rooms

def get_room_graph(num_agents, rooms, doors, keys, goals, id2room, one_way_ids, valid_rooms, start_rooms, difficulty, next_difficulty):
    goal_rooms = list(set([id2room[tuple(ag)] for ag in goals]))
    keys_rooms = []
    kr_ref = []
    for key in keys[:len(doors)]:
        keys_rooms.append(list(set([id2room[tuple(kk)] for kk in key])))
        kr_ref.append(np.array([id2room[tuple(kk)] for kk in key], dtype=int))
    all_random = keys_rooms + [goal_rooms]
    rndid2rooms = list(itertools.product(*all_random))
    rooms2rndid = {rms: _ir for _ir, rms in enumerate(rndid2rooms)}
    al = 1
    for vl in valid_rooms:
        al *= len(vl)
    rgid2agent_room = list(itertools.product(*valid_rooms))
    agent_rooms2rgid = {arms: _i for _i, arms in enumerate(rgid2agent_room)}
    solutions = []
    solvables = []
    all_starts = list(itertools.product(*start_rooms))
    preds = []
    sol_classes = []
    n_unique = 0
    for (rndid, rnd_rooms) in tqdm(enumerate(rndid2rooms)):
        room_graph = 0.1 * np.eye(al + 1)
        key_rooms = list(rnd_rooms[:-1])
        goal_room = rnd_rooms[-1]
        for i, i_agents_poses in enumerate(rgid2agent_room):
            i_held_keys = [[] for _ in range(len(doors))]
            i_valid_next = []
            i_needed_doors = []
            for agentid, i_agent_room in enumerate(i_agents_poses):
                if i_agent_room in key_rooms:
                    for iar in [_i for _i, _kr in enumerate(key_rooms) if (_kr == i_agent_room) or (i_agent_room in rooms[_kr]['eq_rooms'])]:
                        i_held_keys[iar].append(agentid)

                i_valid_next.append([nb['room'] for nb in rooms[i_agent_room]['neighbors']])
                i_needed_doors.append([nb['door'] for nb in rooms[i_agent_room]['neighbors']])
                if i_agent_room == goal_room:
                    room_graph[i, -1] = 1
            for j, j_agents_poses in enumerate(rgid2agent_room):
                if i == j:
                    continue
                j_held_keys = [[] for _ in range(len(doors))]
                needed_keys = []
                _valid_trans = True
                for agentid, j_agent_room in enumerate(j_agents_poses):
                    if (i_agents_poses[agentid] == j_agent_room):
                        needed_keys.append(None)
                    else:
                        if j_agent_room in i_valid_next[agentid]:
                            did = i_valid_next[agentid].index(j_agent_room)
                            needed_keys.append(i_needed_doors[agentid][did])
                        else:
                            _valid_trans = False
                            break
                    if j_agent_room in key_rooms:
                        for jar in [_j for _j, _kr in enumerate(key_rooms) if (_kr == j_agent_room) or (j_agent_room in rooms[_kr]['eq_rooms'])]:
                            j_held_keys[jar].append(agentid)
                if not _valid_trans:
                    continue
                assigned_agents = []
                assigned_keys = []
                n_kr = [0] * len(rooms)
                for agentid in np.arange(num_agents):
                    nk = needed_keys[agentid]
                    ar = i_agents_poses[agentid]
                    fake_room = rooms[ar]['locs'].size == 2 and np.all(rooms[ar]['locs'].flatten() == doors[nk])
                    if nk in one_way_ids:
                        if i_agents_poses[agentid] == key_rooms[nk]:
                            continue
                    if nk is not None and nk not in assigned_keys and not fake_room:
                        intersecting = list(set(i_held_keys[nk]).intersection(set(j_held_keys[nk])) - set(assigned_agents))


                        if len(intersecting) - n_kr[key_rooms[nk]] <= 0:
                            _valid_trans = False
                            break
                        else:
                            n_kr[key_rooms[nk]] += 1
                            assigned_agents.append(agentid)
                            assigned_keys.append(nk)


                if _valid_trans:
                    room_graph[i, j] = 1
        sol, pred = dijkstra(csr_matrix(room_graph), return_predecessors=True)


        solvable = 1
        min_len_sol = np.inf
        max_len_sol = 0
        for sp_ in all_starts:
            srgid_ = agent_rooms2rgid[tuple(sp_)]
            solvable = solvable * (sol[srgid_,-1] < np.inf)
            min_len_sol = min(min_len_sol, sol[srgid_,-1])
            max_len_sol = max(max_len_sol, sol[srgid_,-1])
        solutions.append(sol)
        if min_len_sol < difficulty or max_len_sol >= next_difficulty:
            solvable = False
        if solvable:
            for ci, ck in enumerate(key_rooms[:-1]):
                for cin, ckn in enumerate(key_rooms[ci+1:]):
                    _sol = True
                    if ck == ckn:
                        _sol = False
                        ckk = keys[ci][kr_ref[ci] == ck]
                        cknk = keys[cin + ci + 1][kr_ref[cin + ci + 1] == ckn]
                        for ckkk in ckk:
                            if np.any(cknk != ckkk):
                                _sol = True
                                break
                    solvable = solvable and _sol
                    if not solvable:
                        break


        if solvable:
            n_unique += len(np.unique(key_rooms[:3])) == 3
            solvables.append(np.array(rnd_rooms, dtype=int))
            equiv = False
            for ip, pp in enumerate(preds):
                equiv = True
                for sp_ in all_starts:
                    srgid_ = agent_rooms2rgid[tuple(sp_)]
                    curr = -1
                    _equiv = True
                    while curr != srgid_:
                        if pred[srgid_, curr] == pp[srgid_, curr]:
                            curr = pred[srgid_, curr]
                        else:
                            _equiv = False
                            break
                    equiv = equiv and _equiv
                if equiv:
                    sol_classes[ip].append((rndid, np.array(rnd_rooms, dtype=int)))
                    break
            if not equiv:
                preds.append(pred)
                sol_classes.append([(rndid, np.array(rnd_rooms, dtype=int))])

    print("Total of " + str(len(solvables)) + " solvable scenarios. # of unique solution paths: " + str(n_unique))
    return np.stack(solutions), np.stack(solvables), rooms2rndid, agent_rooms2rgid, sol_classes

def compile_environment(params, num_agents, difficulty, balanced):
    if difficulty not in params['difficulties'].keys():
        raise ValueError("Chosen difficulty is not available in this map!")
    map_size, walls, doors, keys, goal, view_range, valid_locs = (
        params['map_size'],
        params['walls'],
        params['doors'],
        params['keys'],
        params['goal'],
        params['view_range'],
        params["valid_locs"]
    )
    print('Compiling ' + params['name'] + ' - ' + difficulty + ' mode...')
    diff = params['difficulties'][difficulty]
    if difficulty == 'easy':
        next_diff = params['difficulties'].get('medium', np.inf)
    elif difficulty == 'medium':
        next_diff = params['difficulties'].get('hard', np.inf)
    elif difficulty == 'hard':
        next_diff = np.inf
    one_way_ids = params.get('one_way_ids', tuple())
    colorless_ids = params.get('colorless_ids', tuple())
    rooms, _id2room, id2room_agents, wall_map, valid_rooms, start_rooms = get_rooms(map_size, doors, walls, valid_locs)
    room_solutions, solvable, rs2rndid, ar2rgid, rndid_classes = get_room_graph(num_agents, rooms, doors, keys, goal, _id2room, one_way_ids, valid_rooms, start_rooms, diff, next_diff)
    keys_rooms = tuple([tuple([_id2room[tuple(k)] for k in vk]) for vk in keys[:len(doors)]])

    goal_rooms = tuple([_id2room[tuple(g)] for g in goal])
    init_dist = initial_state_distribution(
        [np.array(ak, dtype=int) for ak in keys_rooms],
        [np.array(goal_rooms, dtype=int)],
        rndid_classes,
        balance=balanced,
    )
    room_solutions = room_solutions[init_dist['rndid']]
    init_dist['rndid'] = np.arange(len(init_dist['rndid']))
    ar2rgid = numpify_dict(ar2rgid, key_length=num_agents, key_support=len(rooms))
    id2room = (-1) * np.ones((2, map_size, map_size), dtype=int)
    for k, v in id2room_agents.items():
        if isinstance(v, list):
            id2room[0, k[0], k[1]] = v[0]
            id2room[1, k[0], k[1]] = v[1]
        else:
            id2room[0, k[0], k[1]] = v
            id2room[1, k[0], k[1]] = v
    all_keys = keys[:len(doors)]
    all_decoy_keys = keys[len(doors):]
    decoy = len(all_decoy_keys) > 0
    all_doors = doors
    colored_ids = np.array([i for i in range(len(all_keys)) if i not in colorless_ids], dtype=int)
    keys_init = deepcopy(all_keys)
    if len(all_decoy_keys) > 0:
        all_keys = list(all_keys) + [(-1) * np.ones((2), dtype=int)]
        all_doors = np.concatenate((all_doors, (-1) * np.ones((1, 2), dtype=int)), axis=0)
    _colored_ids = np.array([i for i in range(len(all_keys)) if i not in colorless_ids], dtype=int)
    if len(colorless_ids) > 0:
        subs_cl = len(colorless_ids) - 1
        all_cl_keys = [kk for i, kk in enumerate(all_keys) if i in colorless_ids]
        all_keys = [kk for i, kk in enumerate(all_keys) if i in _colored_ids]
        keys_init = [kk for i, kk in enumerate(keys_init) if i in colored_ids]
        all_cl_doors = [dd for i, dd in enumerate(all_doors) if i in colorless_ids]
        all_doors = [dd for i, dd in enumerate(all_doors) if i in _colored_ids]
    else:
        subs_cl = 0
        all_keys = all_keys
        all_doors = all_doors
        all_cl_keys = tuple()
        all_cl_doors = tuple()

    door_l = len(all_doors) + len(all_cl_doors)
    key_l = len(all_keys) + len(all_cl_keys)
    color2id = {
        params['colors'][i]: i
        for i in range(
            max(door_l - subs_cl,
                key_l - subs_cl,
                num_agents)
        )
    }
    cl_colorid = len(color2id) - 1
    color_words = list(params['colors'][:max(num_agents, len(color2id))])
    vocab = tuple(list(object_words) + color_words)
    pos2txt = {i: k for i, k in enumerate(vocab)}
    txt2pos = {v: k for k, v in pos2txt.items()}
    view_size = 2 * view_range + 1

    _owall_r = (num_agents, num_agents + (map_size ** 2))
    _odoor_r = (_owall_r[1], _owall_r[1] + ((door_l - subs_cl) * (map_size ** 2)))
    _okey_r = (_odoor_r[1], _odoor_r[1] + ((key_l - subs_cl) * (map_size ** 2)))
    _ogoal_r = (_okey_r[1], _okey_r[1] + (map_size ** 2))
    _oagent_r = (_ogoal_r[1], _ogoal_r[1] + (num_agents * (map_size ** 2)))
    _oall_doors_r = (_oagent_r[1], _oagent_r[1] + (map_size ** 2))
    _oall_keys_r = (_oall_doors_r[1], _oall_doors_r[1] + (map_size ** 2))
    _oall_agents_r = (_oall_keys_r[1], _oall_keys_r[1] + (map_size ** 2))
    object_locs = {
        'wall': _owall_r,
        'door': _odoor_r,
        'key': _okey_r,
        'goal': _ogoal_r,
        'agent': _oagent_r,
        '__all__':{'door':_oall_doors_r, 'key':_oall_keys_r, 'agent':_oall_agents_r},
    }

    self_obs_dim = (
            num_agents +
            (door_l + key_l - (2 * subs_cl) + num_agents + 2 + 3) * (map_size ** 2)
    )
    cent_dim = (
            (door_l + key_l - (2 * subs_cl) + num_agents + 2 + 3) * (map_size ** 2)
    )
    comm_dim = num_agents * (self_obs_dim + len(vocab) + 1)

    ca2mask = np.ones((len(vocab) + 1, self_obs_dim))
    for k, v in txt2pos.items():
        if k in object_words:
            for ow in object_words:
                if ow == k:
                    continue
                rng = object_locs[ow]
                ca2mask[v, rng[0]: rng[1]] = 0
                if ow in object_locs['__all__']:
                    rng = object_locs['__all__'][ow]
                    ca2mask[v, rng[0]: rng[1]] = 0

        else:
            _v = v - len(object_words)
            _u = 0
            done = False
            while not done:
                if _u == _v:
                    _u += 1
                    continue
                done = True
                if (_u + 1) * (map_size ** 2) + _odoor_r[0] <= _odoor_r[1]:
                    ca2mask[v, _odoor_r[0] + _u * (map_size ** 2):
                               _odoor_r[0] + (_u + 1) * (map_size ** 2)] = 0
                    done = False
                if (_u + 1) * (map_size ** 2) + _okey_r[0] <= _okey_r[1]:
                    ca2mask[v, _okey_r[0] + _u * (map_size ** 2):
                               _okey_r[0] + (_u + 1) * (map_size ** 2)] = 0
                    done = False
                if (_u + 1) * (map_size ** 2) + _oagent_r[0] <= _oagent_r[1]:
                    ca2mask[v, _oagent_r[0] + _u * (map_size ** 2):
                               _oagent_r[0] + (_u + 1) * (map_size ** 2)] = 0
                    done = False
                _u += 1
    ca2mask[-1] = np.logical_not(np.logical_not(ca2mask[len(object_words):-1]).sum(axis=0))

    compiled = {
        'params': params,
        'data':{
            'map_size': map_size,
            'id2room': id2room,
            'init_dist': init_dist,
            'ar2rgid': ar2rgid,
            'keys_rooms': keys_rooms,
            'goal_rooms': goal_rooms,
            'dists': room_solutions,
            'wall_map': tuple([tuple(wm) for wm in wall_map]),
            'valid_locs': params['valid_locs'],
            'all_doors': all_doors,
            'all_cl_doors': all_cl_doors,
            'keys_init': keys_init,
            'all_keys': all_keys,
            'all_cl_keys': all_cl_keys,
            'all_decoy_keys': all_decoy_keys,
            'all_goals': goal,
            'colored_ids': tuple(colored_ids),
            'colorless_ids': tuple(colorless_ids),
            'ca2mask': ca2mask,
            'object_locs': object_locs,
            'self_obs_dim': self_obs_dim,
            'cent_dim': cent_dim,
            'comm_dim': comm_dim,
            'cl_colorid': cl_colorid,
            'object_words': object_words,
            'vocab': vocab,
            'txt2pos': txt2pos,
            'pos2txt': pos2txt,
            'view_size': view_size,
            'view_range': view_range,
            'decoy': decoy,
            'colors': tuple(color_words),
            'door_l': door_l,
            'key_l': key_l,
        }
    }
    return compiled


def load_map(env_inst, params, num_agents, save_compilation, difficulty='hard', balanced=True):
    assert num_agents >= params['num_agents'], "The minimum number of agents for this environment is " + str(
        params['num_agents'])
    file_name = params['name'] + '_' + str(num_agents) + '_' + difficulty
    if balanced:
        file_name = file_name + '_agents_balanced.pkl'
    else:
        file_name = file_name + '_agents.pkl'
    comp_path = path.join(path.dirname(__file__), f"comp/{file_name}")
    try:
        with open(comp_path, 'rb') as f:
            compiled = pickle.load(f)
    except:
        compiled = compile_environment(params, num_agents, difficulty, balanced)
        if save_compilation:
            file_name = params['name'] + '_' + str(num_agents) + '_' + difficulty
            if balanced:
                file_name = file_name + '_agents_balanced.pkl'
            else:
                file_name = file_name + '_agents.pkl'
            comp_path = path.join(path.dirname(__file__), f"comp/{file_name}")
            with open(comp_path, 'wb') as f:
                pickle.dump(compiled, f)
    data = compiled['data']
    data['id2room'] = jnp.array(data['id2room'], dtype=int)
    data['init_dist']['probs'] = jnp.array(data['init_dist']['probs'])
    data['init_dist']['rndid'] = jnp.array(data['init_dist']['rndid'])
    data['init_dist']['ids'] = [jnp.array(dd) for dd in data['init_dist']['ids']]
    data['init_dist']['lens'] = [jnp.array(dd) for dd in data['init_dist']['lens']]
    data['ar2rgid'] = [jnp.array(r2r) for r2r in data['ar2rgid']]
    data['keys_rooms'] = [jnp.array(ak, dtype=int) for ak in data['keys_rooms']]
    data['goal_rooms'] = jnp.array(data['goal_rooms'], dtype=int)
    data['dists'] = jnp.array(data['dists'])
    data['wall_map'] = jnp.array(data['wall_map'])[None, ...]
    data['valid_locs'] = [jnp.array(vl, dtype=int) for vl in data['valid_locs']]
    data['all_doors'] = jnp.array(data['all_doors'], dtype=int)
    data['all_cl_doors'] = jnp.array(data['all_cl_doors'], dtype=int)
    data['all_keys'] = [jnp.array(ak, dtype=int) for ak in data['all_keys']]
    data['keys_init'] = [jnp.array(ak, dtype=int) for ak in data['keys_init']]
    data['all_cl_keys'] = [jnp.array(ak, dtype=int) for ak in data['all_cl_keys']]
    data['all_decoy_keys'] = [jnp.array(ak, dtype=int) for ak in data['all_decoy_keys']]
    data['all_goals'] = jnp.array(data['all_goals'], dtype=int)
    data['ca2mask'] = jnp.array(data['ca2mask'])

    env_inst.__dict__.update(data)

