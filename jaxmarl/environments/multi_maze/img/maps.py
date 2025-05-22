import numpy as np

maps_list = []

# 2 agent basic map:
#    01234567
#   ##########
# 0 #    #D  #
# 1 #  g #   #
# 2 # D  #  k#
# 3 #    #  k#
# 4 ###D##D###
# 5 #     a  #
# 6 #   a    #
# 7 #  k   k #
#   ##########

_map_size = 8
_walls = np.concatenate(
    (
        np.concatenate((np.arange(2).reshape((-1,1)), 4 * np.ones(2).reshape((-1,1))), axis=1),
        np.concatenate((np.arange(3,5).reshape((-1,1)), 4 * np.ones(2).reshape((-1,1))), axis=1),
        np.concatenate((np.arange(6, _map_size).reshape((-1, 1)), 4 * np.ones(2).reshape((-1, 1))), axis=1),
        np.concatenate((4 * np.ones(4).reshape((-1,1)), np.arange(4).reshape((-1, 1))), axis=1)
    ), axis=0).astype(int)
_doors = np.array([[2, 4],[5, 0], [5, 4], [1, 1]], dtype=int)
_goal_locs = np.concatenate(
    [
        np.concatenate((np.arange(4).reshape((-1,1)), i * np.ones(4).reshape((-1,1))), axis=1)
        for i in range(3)], axis=0).astype(int)
_goal = np.array(_goal_locs, dtype=int)
_valid_locs = np.concatenate(
    [
        np.concatenate((np.arange(_map_size).reshape((-1,1)), i * np.ones(_map_size).reshape((-1,1))), axis=1)
        for i in range(5, 8)], axis=0).astype(int)
_validk1 = np.concatenate(
    [
        np.concatenate((np.arange(5,8).reshape((-1,1)), i * np.ones(3).reshape((-1,1))), axis=1)
        for i in range(3)], axis=0).astype(int)

_validk2 = np.concatenate((np.arange(_map_size).reshape((-1,1)), 7 * np.ones(_map_size).reshape((-1,1))), axis=1).astype(int)
_validk3 = np.concatenate(
    [
        np.concatenate((np.arange(5,8).reshape((-1,1)), i * np.ones(3).reshape((-1,1))), axis=1)
        for i in range(1,3)] + [np.concatenate((np.arange(6,8).reshape((-1,1)), np.zeros(2).reshape((-1,1))), axis=1)], axis=0).astype(int)
_keys = (_validk1.reshape((-1,2)).astype(int), _validk2.reshape((-1,2)).astype(int), _validk2.reshape((-1,2)).astype(int), _validk3.reshape((-1,2)).astype(int))

maps_list.append(
    {
        'name': 'Basic2_v1',
        'map_size':_map_size,
        'walls':tuple([tuple(wall) for wall in _walls]),
        'keys':tuple([tuple([tuple(kk) for kk in key]) for key in _keys]),
        'doors':tuple([tuple(door) for door in _doors]),
        'goal':tuple([tuple(goal) for goal in _goal]),
        'valid_locs':(tuple([tuple(vl) for vl in _valid_locs]), tuple([tuple(vl) for vl in _valid_locs])),
        'num_agents':2,
        'view_range': 1,
    }
)

# 2 agent basic map v2:
#    01234567
#   ##########
# 0 #g#    #g#
# 1 # #    #D#
# 2 # #  k # #
# 3 # #  k D #
# 4 #D###D####
# 5 # k   a  #
# 6 #   a    #
# 7 #  k   k #
#   ##########

_map_size = 8
_walls = np.concatenate(
    (
        np.concatenate((np.arange(1,4).reshape((-1,1)), 4 * np.ones(3).reshape((-1,1))), axis=1),
        np.concatenate((np.arange(5,8).reshape((-1,1)), 4 * np.ones(3).reshape((-1,1))), axis=1),
        np.concatenate((np.ones(4).reshape((-1,1)), np.arange(4).reshape((-1, 1))), axis=1),
        np.concatenate((6 * np.ones(3).reshape((-1,1)), np.arange(3).reshape((-1, 1))), axis=1),
    ), axis=0).astype(int)
_doors = np.array([[4, 4],[0, 4], [6, 3], [7, 1]], dtype=int)
_goal = np.array([[0, 0], [7, 0]], dtype=int)
_valid_locs = np.concatenate(
    [
        np.concatenate((np.arange(_map_size).reshape((-1,1)), i * np.ones(_map_size).reshape((-1,1))), axis=1)
        for i in range(5, 8)], axis=0).astype(int)
_validk1 = [
    np.concatenate((np.arange(_map_size).reshape((-1,1)), i * np.ones(_map_size).reshape((-1,1))), axis=1) for i in range(6, 8)
]
_validk1 =_validk1 + [np.concatenate((np.arange(4).reshape((-1,1)), 5 * np.ones(4).reshape((-1,1))), axis=1)]
_validk1 =_validk1 + [np.concatenate((np.arange(5,8).reshape((-1,1)), 5 * np.ones(3).reshape((-1,1))), axis=1)]
_validk1 = np.concatenate(_validk1, axis=0).astype(int)
_validk2 = np.concatenate(
    [
        np.concatenate((np.arange(2,6).reshape((-1,1)), i * np.ones(4).reshape((-1,1))), axis=1)
        for i in range(4)], axis=0).astype(int)
_keys = (_validk1.reshape((-1,2)).astype(int), _validk2.reshape((-1,2)).astype(int), _validk1.reshape((-1,2)).astype(int), _validk1.reshape((-1,2)).astype(int), _validk2.reshape((-1,2)))

maps_list.append(
    {
        'name': 'Basic2_v2',
        'map_size':_map_size,
        'walls': tuple([tuple(wall) for wall in _walls]),
        'keys': tuple([tuple([tuple(kk) for kk in key]) for key in _keys]),
        'doors': tuple([tuple(door) for door in _doors]),
        'goal': tuple([tuple(goal) for goal in _goal]),
        'valid_locs': (tuple([tuple(vl) for vl in _valid_locs]), tuple([tuple(vl) for vl in _valid_locs])),
        'num_agents':2,
        'view_range': 1,
    }
)

# 3 agent basic map:
#    012345678
#   ###########
# 0 #g   #  k #
# 1 #D   D##D##
# 2 ####D#    #
# 3 #   k# kk #
# 4 ###D##  k #
# 5 #  aa####D#
# 6 #   a# k  #
# 7 #    #    #
# 8 #k   D   k#
#   ###########

_map_size = 9
_walls = np.concatenate(
    (
        np.concatenate((np.arange(5,7).reshape((-1,1)), 1 * np.ones(2).reshape((-1,1))), axis=1),
        np.concatenate((np.arange(3).reshape((-1,1)), 2 * np.ones(3).reshape((-1,1))), axis=1),
        np.concatenate((np.arange(2).reshape((-1,1)), 4 * np.ones(2).reshape((-1,1))), axis=1),
        np.concatenate((np.arange(5,8).reshape((-1,1)), 5 * np.ones(3).reshape((-1,1))), axis=1),
        np.concatenate((4 * np.ones(8).reshape((-1,1)), np.arange(8).reshape((-1, 1))), axis=1),
        np.array([[8, 1], [3, 4]])
    ), axis=0).astype(int)
_doors = np.array([[4, 8], [2, 4], [8, 5], [3, 2], [7, 1], [0, 1]], dtype=int)
_goal = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0]], dtype=int)
_valid_locs = np.concatenate(
    [
        np.concatenate((np.arange(4).reshape((-1,1)), i * np.ones(4).reshape((-1,1))), axis=1)
        for i in range(5, 9)], axis=0).astype(int)
_validk1 = np.concatenate(
    [
        np.concatenate((np.arange(4).reshape((-1,1)), i * np.ones(4).reshape((-1,1))), axis=1)
        for i in range(5, 8)], axis=0).astype(int).reshape((-1,2))
_validk1 = np.concatenate((_validk1, np.concatenate((np.arange(3).reshape((-1,1)), 8 * np.ones(3).reshape((-1,1))), axis=1)), axis=0).astype(int)
_validk2 = np.concatenate(
    [
        np.concatenate((np.arange(5, 9).reshape((-1,1)), i * np.ones(4).reshape((-1,1))), axis=1)
        for i in range(6, 9)], axis=0).astype(int).reshape((-1,2))
_validk3 = np.concatenate((np.arange(4).reshape((-1,1)), 3 * np.ones(4).reshape((-1,1))), axis=1).astype(int).reshape((-1,2))
_validk4 = np.concatenate(
    [
        np.concatenate((np.arange(5,9).reshape((-1,1)), i * np.ones(4).reshape((-1,1))), axis=1)
        for i in range(2,5)], axis=0).astype(int).reshape((-1,2))
_validk5 = np.concatenate(
    [
        np.concatenate((np.arange(5,9).reshape((-1,1)), i * np.ones(4).reshape((-1,1))), axis=1)
        for i in range(3,5)], axis=0).astype(int).reshape((-1,2))
_validk5 = np.concatenate((_validk5, np.array([[5, 2], [6, 2], [8, 2]])), axis=0).astype(int)
_validk6 = np.concatenate((np.arange(5,9).reshape((-1,1)), np.zeros(4).reshape((-1,1))), axis=1).astype(int).reshape((-1,2))

_keys = (_validk1, _validk2, _validk3, _validk4, _validk5, _validk6, _validk2, _validk4)

maps_list.append(
    {
        'name': 'Basic3_v1',
        'map_size':_map_size,
        'walls': tuple([tuple(wall) for wall in _walls]),
        'keys': tuple([tuple([tuple(kk) for kk in key]) for key in _keys]),
        'doors': tuple([tuple(door) for door in _doors]),
        'goal': tuple([tuple(goal) for goal in _goal]),
        'valid_locs': (
            tuple([tuple(vl) for vl in _valid_locs]),
            tuple([tuple(vl) for vl in _valid_locs]),
            tuple([tuple(vl) for vl in _valid_locs])
        ),
        'num_agents':3,
        'view_range': 1,
    }
)

# 3 agent basic map v2:
#    012345678
#   ###########
# 0 #  dk  kdk#
# 1 #D##   ####
# 2 #  #   #g #
# 3 #g #####  #
# 4 #########D#
# 5 #  #k#kd# #
# 6 #  kd##k#d#
# 7 #  ####  k#
# 8 # kd Dg#  #
#   ###########

_map_size = 9
_walls = np.concatenate(
    (
        np.concatenate((np.arange(1,3).reshape((-1,1)), np.ones(2).reshape((-1,1))), axis=1),
        np.concatenate((np.arange(6,9).reshape((-1,1)), np.ones(3).reshape((-1,1))), axis=1),
        np.concatenate((np.arange(3,6).reshape((-1,1)), 3 * np.ones(3).reshape((-1,1))), axis=1),
        np.concatenate((np.arange(8).reshape((-1,1)), 4 * np.ones(8).reshape((-1,1))), axis=1),
        np.concatenate((np.arange(4,6).reshape((-1,1)), 6 * np.ones(2).reshape((-1,1))), axis=1),
        np.concatenate((np.arange(2,6).reshape((-1,1)), 7 * np.ones(4).reshape((-1,1))), axis=1),
        np.array([[6, 2], [2, 3], [6, 3], [2, 5], [4, 5], [7, 5], [6, 8], [7, 6], [2, 2]])
    ), axis=0).astype(int)
_doors = np.array([[0, 1], [8, 4], [4, 8], [2, 0], [7, 0], [8, 6], [6, 5], [2, 8], [3, 6]], dtype=int)
_goal = np.array([[0, 3], [7, 2], [5, 8]], dtype=int)
_valid_locs1 = [np.concatenate((5 * np.ones(3).reshape((-1,1)), np.arange(3).reshape((-1,1))), axis=1)]
_valid_locs1 = _valid_locs1 + [np.array([[6, 0]], dtype=int)]
_valid_locs1 = np.concatenate(_valid_locs1, axis=0).astype(int).reshape((-1,2))
_valid_locs2 = np.array([[6, 6], [6, 7], [7, 7], [7, 8]], dtype=int)

_valid_locs3 = np.array([[0, 5], [1, 5]], dtype=int)

_keys = (
    np.array([[5, 5], [8, 5]], dtype=int),
    np.array([[3, 5], [3, 8]], dtype=int),
    np.array([[8, 0], [1, 0]], dtype=int),
    np.array([[3, 0]], dtype=int),
    np.array([[6, 0]], dtype=int),
    np.array([[8, 7]], dtype=int),
    np.array([[6, 6]], dtype=int),
    np.array([[1, 8]], dtype=int),
    np.array([[2, 6]], dtype=int),
    np.array([[5, 5], [8, 5]], dtype=int),
    np.array([[3, 5], [3, 8]], dtype=int),
    np.array([[8, 0], [1, 0]], dtype=int),

)
one_way_ids = np.arange(3,9)

maps_list.append(
    {
        'name': 'Basic3_v2',
        'map_size':_map_size,
        'walls': tuple([tuple(wall) for wall in _walls]),
        'keys': tuple([tuple([tuple(kk) for kk in key]) for key in _keys]),
        'doors': tuple([tuple(door) for door in _doors]),
        'goal': tuple([tuple(goal) for goal in _goal]),
        'colorless_ids': tuple(one_way_ids),
        'one_way_ids':tuple(one_way_ids),
        'valid_locs': (
            tuple([tuple(vl) for vl in _valid_locs1]),
            tuple([tuple(vl) for vl in _valid_locs2]),
            tuple([tuple(vl) for vl in _valid_locs3]),
        ),
        'num_agents':3,
        'view_range': 2,
    }
)

# Triple Trouble:
#    012345678
#   ###########
# 0 #  dk  kdk#
# 1 #D##   ####
# 2 #  #   #g #
# 3 #g #   #  #
# 4 #########D#
# 5 #  #k#kd# #
# 6 #  kd##k#d#
# 7 #  ####  k#
# 8 # kd Dg#  #
#   ###########

_map_size = 9
_walls = np.concatenate(
    (
        np.concatenate((np.arange(1,3).reshape((-1,1)), np.ones(2).reshape((-1,1))), axis=1),
        np.concatenate((np.arange(6,9).reshape((-1,1)), np.ones(3).reshape((-1,1))), axis=1),
        np.concatenate((np.arange(8).reshape((-1,1)), 4 * np.ones(8).reshape((-1,1))), axis=1),
        np.concatenate((np.arange(4,6).reshape((-1,1)), 6 * np.ones(2).reshape((-1,1))), axis=1),
        np.concatenate((np.arange(2,6).reshape((-1,1)), 7 * np.ones(4).reshape((-1,1))), axis=1),
        np.array([[6, 2], [2, 3], [6, 3], [2, 5], [4, 5], [7, 5], [6, 8], [7, 6], [2, 2]])
    ), axis=0).astype(int)
_doors = np.array([[0, 1], [8, 4], [4, 8], [2, 0], [7, 0], [8, 6], [6, 5], [2, 8], [3, 6]], dtype=int)
_goal = np.array([[0, 3], [7, 2], [5, 8]], dtype=int)
_valid_locs1 = [np.concatenate((5 * np.ones(4).reshape((-1,1)), np.arange(4).reshape((-1,1))), axis=1)]
_valid_locs1 = _valid_locs1 + [np.array([[6, 0]], dtype=int)]
_valid_locs1 = np.concatenate(_valid_locs1, axis=0).astype(int).reshape((-1,2))
_valid_locs2 = np.array([[6, 6], [6, 7], [7, 7], [7, 8]], dtype=int)

_valid_locs3 = np.array([[0, 5], [1, 5]], dtype=int)

_keys = (
    np.array([[5, 5]], dtype=int),
    np.array([[3, 5]], dtype=int),
    np.array([[8, 0]], dtype=int),
    np.array([[3, 0]], dtype=int),
    np.array([[6, 0]], dtype=int),
    np.array([[8, 7]], dtype=int),
    np.array([[6, 6]], dtype=int),
    np.array([[1, 8]], dtype=int),
    np.array([[2, 6]], dtype=int),
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
    }
)

# Triple Trouble v2:
#    0123456789
#   ############
# 0 #  dk   kdk#
# 1 #D##k    ###
# 2 #  #d#   #g#
# 3 #g #k#   # #
# 4 ##########D#
# 5 #  #k#kd#  #
# 6 #  kd##k#d##
# 7 #  kdk#  k #
# 8 # kd Dg# ###
# 9 #  #  ##kdk#
#   ###########

_map_size = 10
_walls = np.concatenate(
    (
        np.concatenate((np.arange(1,3).reshape((-1,1)), np.ones(2).reshape((-1,1))), axis=1),
        np.concatenate((np.arange(8,10).reshape((-1,1)), np.ones(2).reshape((-1,1))), axis=1),
        np.concatenate((np.arange(9).reshape((-1,1)), 4 * np.ones(9).reshape((-1,1))), axis=1),
        np.concatenate((np.arange(4,6).reshape((-1,1)), 6 * np.ones(2).reshape((-1,1))), axis=1),
        np.concatenate((np.arange(8,10).reshape((-1,1)), 8 * np.ones(2).reshape((-1,1))), axis=1),
        np.concatenate((np.arange(5,7).reshape((-1,1)), 9 * np.ones(2).reshape((-1,1))), axis=1),
        np.concatenate((2 * np.ones(2).reshape((-1, 1)), np.arange(2,4).reshape((-1, 1))), axis=1),
        np.concatenate((4 * np.ones(2).reshape((-1, 1)), np.arange(2, 4).reshape((-1, 1))), axis=1),
        np.concatenate((8 * np.ones(2).reshape((-1, 1)), np.arange(2, 4).reshape((-1, 1))), axis=1),
        np.array([[2, 5], [4, 5], [7, 5], [7, 6], [9, 6], [5, 7], [2, 9], [6, 8]])
    ), axis=0).astype(int)
_doors = np.array([[0, 1], [9, 4], [4, 8], [2, 0], [8, 0], [3, 2], [8, 6], [6, 5], [8, 9], [2, 8], [3, 6], [3, 7]], dtype=int)
_goal = np.array([[0, 3], [9, 2], [5, 8]], dtype=int)
_valid_locs1 = np.concatenate((7 * np.ones(4).reshape((-1,1)), np.arange(4).reshape((-1,1))), axis=1)
_valid_locs2 = np.array([[6, 6], [6, 7], [7, 7], [7, 8], [7, 9]], dtype=int)

_valid_locs3 = np.array([[0, 5], [1, 5]], dtype=int)

_keys = (
    np.array([[5, 5], [9, 9]], dtype=int),
    np.array([[3, 5], [4, 7]], dtype=int),
    np.array([[9, 0], [3, 3]], dtype=int),
    np.array([[3, 0]], dtype=int),
    np.array([[7, 0]], dtype=int),
    np.array([[3, 1]], dtype=int),
    np.array([[8, 7]], dtype=int),
    np.array([[6, 6]], dtype=int),
    np.array([[7, 9]], dtype=int),
    np.array([[1, 8]], dtype=int),
    np.array([[2, 6]], dtype=int),
    np.array([[2, 7]], dtype=int),
    np.array([[5, 5], [9, 9]], dtype=int),
    np.array([[3, 5], [4, 7]], dtype=int),
    np.array([[9, 0], [3, 3]], dtype=int),
)
one_way_ids = np.arange(3,12)

maps_list.append(
    {
        'name': 'Triple Trouble v2',
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
    }
)


# maps_list.append(
#     {
#         'name': 'Triple Trouble',
#         'map_size':_map_size,
#         'walls': tuple([tuple(wall) for wall in _walls]),
#         'keys': tuple([tuple([tuple(kk) for kk in key]) for key in _keys]),
#         'doors': tuple([tuple(door) for door in _doors]),
#         'goal': tuple([tuple(goal) for goal in _goal]),
#         'colorless_ids': tuple(one_way_ids),
#         'one_way_ids':tuple(one_way_ids),
#         'valid_locs': (
#             tuple([tuple(vl) for vl in _valid_locs1]),
#             tuple([tuple(vl) for vl in _valid_locs2]),
#             tuple([tuple(vl) for vl in _valid_locs3]),
#         ),
#         'num_agents':3,
#         'view_range': 2,
#     }
# )



# 4 agent basic map:
#    0123456789
#   ############
# 0 #g D   # k #
# 1 #####D##   #
# 2 #      ##D##
# 3 # k    #   #
# 4 #    k #   #
# 5 ###D####k  #
# 6 #      ###D#
# 7 # a a  #   #
# 8 # a a  #k  #
# 9 # k    D   #
#   ############

_map_size = 10
_walls = np.concatenate(
    (
        np.concatenate((np.arange(4).reshape((-1,1)), 1 * np.ones(4).reshape((-1,1))), axis=1),
        np.concatenate((np.arange(2).reshape((-1,1)), 5 * np.ones(2).reshape((-1,1))), axis=1),
        np.concatenate((np.arange(3,6).reshape((-1,1)), 5 * np.ones(3).reshape((-1,1))), axis=1),
        np.concatenate((6 * np.ones(9).reshape((-1,1)), np.arange(9).reshape((-1, 1))), axis=1),
        np.concatenate((np.arange(7,9).reshape((-1,1)), 6 * np.ones(2).reshape((-1,1))), axis=1),
        np.array([[7,2], [9,2], [5, 1]])
    ), axis=0).astype(int)
_doors = np.array([[9, 6], [4, 1], [2, 0], [8, 2], [2, 5], [6, 9]], dtype=int)
_goal_locs = np.concatenate((np.arange(2).reshape((-1,1)), np.zeros(2).reshape((-1,1))), axis=1).astype(int)
_goal = np.array(_goal_locs, dtype=int)
_valid_locs = np.concatenate(
    [
        np.concatenate((np.arange(6).reshape((-1,1)), i * np.ones(6).reshape((-1,1))), axis=1)
        for i in range(6, 10)], axis=0).astype(int)
_validk1 = np.concatenate(
    [
        np.concatenate((np.arange(6).reshape((-1,1)), i * np.ones(6).reshape((-1,1))), axis=1)
        for i in range(3,5)], axis=0).astype(int).reshape((-1,2))
_validk2 = np.concatenate(
    [
        np.concatenate((np.arange(7,10).reshape((-1,1)), i * np.ones(3).reshape((-1,1))), axis=1)
        for i in range(2)], axis=0).astype(int).reshape((-1,2))
_validk3 = np.concatenate(
    [
        np.concatenate((np.arange(7,10).reshape((-1,1)), i * np.ones(3).reshape((-1,1))), axis=1)
        for i in range(4,6)], axis=0).astype(int).reshape((-1,2))
_validk4 = np.concatenate(
    [
        np.concatenate((np.arange(7,10).reshape((-1,1)), i * np.ones(3).reshape((-1,1))), axis=1)
        for i in range(7,10)], axis=0).astype(int).reshape((-1,2))
_validk5 = np.concatenate(
    [
        np.concatenate((np.arange(5).reshape((-1,1)), i * np.ones(5).reshape((-1,1))), axis=1)
        for i in range(7,10)], axis=0).astype(int).reshape((-1,2))

_keys = (_validk1, _validk1, _validk2, _validk3, _validk4, _validk5)
maps_list.append(
    {
        'name': 'Basic4',
        'map_size':_map_size,
        'walls': tuple([tuple(wall) for wall in _walls]),
        'keys': tuple([tuple([tuple(kk) for kk in key]) for key in _keys]),
        'doors': tuple([tuple(door) for door in _doors]),
        'goal': tuple([tuple(goal) for goal in _goal]),
        'valid_locs': (
            tuple([tuple(vl) for vl in _valid_locs]),
            tuple([tuple(vl) for vl in _valid_locs]),
            tuple([tuple(vl) for vl in _valid_locs]),
            tuple([tuple(vl) for vl in _valid_locs])
        ),
        'num_agents':4,
        'view_range':1,
    }
)

maps = {m['name']: m for m in maps_list}