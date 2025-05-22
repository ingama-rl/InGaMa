import numpy as np

maps_list = []

# Triple Trouble:
#    012345678
#   ###########
# 0 #k     #g #
# 1 #d# k  #KD#
# 2 # D#d  ## #
# 3 ##K#K## kd#
# 4 #g ##k    #
# 5 ####Kd    #
# 6 #   #######
# 7 ##    k#Kg#
# 8 #Kdk  d D #
#   ###########

_map_size = 9
_walls = np.concatenate(
    (
        np.concatenate((np.arange(6, 8).reshape((-1,1)), 2 * np.ones(2).reshape((-1,1))), axis=1),
        np.concatenate((np.arange(4, 6).reshape((-1,1)), 3 * np.ones(2).reshape((-1,1))), axis=1),
        np.concatenate((np.arange(2, 4).reshape((-1,1)), 4 * np.ones(2).reshape((-1,1))), axis=1),
        np.concatenate((np.arange(3).reshape((-1,1)), 5 * np.ones(3).reshape((-1,1))), axis=1),
        np.concatenate((np.arange(3, 9).reshape((-1,1)), 6 * np.ones(6).reshape((-1,1))), axis=1),
        np.concatenate((6 * np.ones(2).reshape((-1,1)), np.arange(2).reshape((-1,1))), axis=1),
        np.concatenate((2 * np.ones(2).reshape((-1,1)), np.arange(2, 4).reshape((-1,1))), axis=1),
        np.array([[1, 1], [0, 3], [0, 7], [6, 7]])
    ), axis=0).astype(int)
_doors = np.array([[1, 2], [8, 1], [7, 8], [0, 1], [3, 2], [8, 3], [4, 5], [5, 8], [1, 8]], dtype=int)
_goal = np.array([[0, 4], [7, 0], [8, 7]], dtype=int)
_valid_locs1 = np.array([[5, 0], [5, 1], [5, 2]], dtype=int)
_valid_locs2 = np.array([[6, 5], [7, 5], [8, 5]], dtype=int)
_valid_locs3 = np.array([[0, 6], [1, 6], [2, 6]], dtype=int)

_keys = (
    np.array([[1, 3], [3, 3], [7, 1], [3, 5], [7, 7], [0, 8]], dtype=int),
    np.array([[1, 3], [3, 3], [7, 1], [3, 5], [7, 7], [0, 8]], dtype=int),
    np.array([[1, 3], [3, 3], [7, 1], [3, 5], [7, 7], [0, 8]], dtype=int),
    np.array([[0, 0]], dtype=int),
    np.array([[3, 1]], dtype=int),
    np.array([[7, 3]], dtype=int),
    np.array([[4, 4]], dtype=int),
    np.array([[5, 7]], dtype=int),
    np.array([[2, 8]], dtype=int),
    np.array([[1, 3], [3, 3], [7, 1], [3, 5], [7, 7], [0, 8]], dtype=int),
    np.array([[1, 3], [3, 3], [7, 1], [3, 5], [7, 7], [0, 8]], dtype=int),
    np.array([[1, 3], [3, 3], [7, 1], [3, 5], [7, 7], [0, 8]], dtype=int),
)
one_way_ids = np.arange(3,9)

maps_list.append(
    {
        'name': 'Triple Trouble',
        'map_size':_map_size,
        'walls': _walls,
        'keys': _keys,
        'doors': _doors,
        'goal': _goal,
        'colorless_ids': tuple(one_way_ids),
        'one_way_ids': tuple(one_way_ids),
        'valid_locs': [_valid_locs1, _valid_locs2, _valid_locs3],
        'num_agents':3,
        'view_range': 2,
        'colors': ('red', 'blue', 'green', 'orange', 'brown'),
        'difficulties': {'easy': 3, 'medium': 4},
    }
)
# Quadruple Trouble:
#    0123456789
#   ############
# 0 #d D G#  kd#
# 1 #k##K #k # #
# 2 #  ####d##D#
# 3 # kdK##K#K #
# 4 ######### G#
# 5 #G #########
# 6 # K#K##Kdk #
# 7 #D##d####  #
# 8 # # k# K##k#
# 9 #dk  #G D d#
#   ############

_map_size = 10
_walls = np.concatenate(
    (
        np.concatenate((np.arange(5).reshape((-1,1)), 4 * np.ones(5).reshape((-1,1))), axis=1),
        np.concatenate((2 * np.ones(3).reshape((-1, 1)), np.arange(5, 8).reshape((-1, 1))), axis=1),
        np.concatenate((np.ones(2).reshape((-1, 1)), np.arange(7, 9).reshape((-1, 1))), axis=1),
        np.concatenate((np.arange(5, 10).reshape((-1,1)), 5 * np.ones(5).reshape((-1,1))), axis=1),
        np.concatenate((7 * np.ones(3).reshape((-1, 1)), np.arange(2, 5).reshape((-1, 1))), axis=1),
        np.concatenate((8 * np.ones(2).reshape((-1, 1)), np.arange(1, 3).reshape((-1, 1))), axis=1),
        np.concatenate((5 * np.ones(5).reshape((-1,1)), np.arange(5).reshape((-1,1))), axis=1),
        np.concatenate((np.arange(2, 5).reshape((-1, 1)), 2 * np.ones(3).reshape((-1, 1))), axis=1),
        np.concatenate((np.arange(1, 3).reshape((-1, 1)), np.ones(2).reshape((-1, 1))), axis=1),
        np.concatenate((4 * np.ones(5).reshape((-1,1)), np.arange(5, 10).reshape((-1,1))), axis=1),
        np.concatenate((np.arange(5, 8).reshape((-1, 1)), 7 * np.ones(3).reshape((-1, 1))), axis=1),
        np.concatenate((np.arange(7, 9).reshape((-1, 1)), 8 * np.ones(2).reshape((-1, 1))), axis=1),
        np.array([[4, 3], [6, 4], [5, 6], [3, 5]])
    ), axis=0).astype(int)
_doors = np.array([[2, 0], [9, 2], [7, 9], [0, 7], [0, 0], [2, 3], [9, 0], [6, 2], [9, 9], [7, 6], [0, 9], [3, 7]], dtype=int)
_goal = np.array([[4, 0], [9, 4], [5, 9], [0, 5]], dtype=int)
_valid_locs1 = np.array([[0, 2], [1, 2], [0, 3], [1, 3]], dtype=int)
_valid_locs2 = np.array([[6, 0], [7, 0], [6, 1], [7, 1]], dtype=int)
_valid_locs3 = np.array([[8, 6], [9, 6], [8, 7], [9, 7]], dtype=int)
_valid_locs4 = np.array([[2, 8], [3, 8], [2, 9], [3, 9]], dtype=int)

_keys = (
    # Reg
    np.array([[3, 1], [3, 3], [8, 3], [6, 3], [6, 8], [6, 6], [1, 6], [3, 6]], dtype=int),
    np.array([[3, 1], [3, 3], [8, 3], [6, 3], [6, 8], [6, 6], [1, 6], [3, 6]], dtype=int),
    np.array([[3, 1], [3, 3], [8, 3], [6, 3], [6, 8], [6, 6], [1, 6], [3, 6]], dtype=int),
    np.array([[3, 1], [3, 3], [8, 3], [6, 3], [6, 8], [6, 6], [1, 6], [3, 6]], dtype=int),
    # One way
    np.array([[0, 1]], dtype=int),
    np.array([[1, 3]], dtype=int),
    np.array([[8, 0]], dtype=int),
    np.array([[6, 1]], dtype=int),
    np.array([[9, 8]], dtype=int),
    np.array([[8, 6]], dtype=int),
    np.array([[1, 9]], dtype=int),
    np.array([[3, 8]], dtype=int),
    # Decoy
    np.array([[3, 1], [3, 3], [8, 3], [6, 3], [6, 8], [6, 6], [1, 6], [3, 6]], dtype=int),
    np.array([[3, 1], [3, 3], [8, 3], [6, 3], [6, 8], [6, 6], [1, 6], [3, 6]], dtype=int),
    np.array([[3, 1], [3, 3], [8, 3], [6, 3], [6, 8], [6, 6], [1, 6], [3, 6]], dtype=int),
    np.array([[3, 1], [3, 3], [8, 3], [6, 3], [6, 8], [6, 6], [1, 6], [3, 6]], dtype=int),
)
one_way_ids = np.arange(4, 12)

maps_list.append(
    {
        'name': 'Quadruple Trouble',
        'map_size':_map_size,
        'walls': _walls,
        'keys': _keys,
        'doors': _doors,
        'goal': _goal,
        'colorless_ids': tuple(one_way_ids),
        'one_way_ids': tuple(one_way_ids),
        'valid_locs': [_valid_locs1, _valid_locs2, _valid_locs3, _valid_locs4],
        'num_agents': 4,
        'view_range': 2,
        'colors': ('red', 'blue', 'green', 'orange', 'yellow', 'brown'),
        'difficulties': {'easy': 3, 'medium': 4, 'hard': 5},
    }
)

maps = {m['name']: m for m in maps_list}
