import pygame
import numpy as np
from os import path


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

def find_font_size(num_agents, words, max_font_size, height):
    done = False
    font_size = int(max_font_size)
    while not done:
        font = pygame.font.SysFont('Helvetica', font_size)
        space = font.size(' ')[0]
        x = 5
        wh = 0
        for ii, word in enumerate(words):
            if ii < len(words) - 1:
                word += ','
            word_surface = font.render(word, 0, pygame.Color('black'))
            word_width, word_height = word_surface.get_size()
            wh = max(wh, word_height)
            x += word_width + space * (ii < len(words) - 1)
        dummy_surface = pygame.Surface((height - (1.25 * wh * num_agents), height))
        max_width, _ = dummy_surface.get_size()
        if x >= max_width - 5:
            font_size -= 1
            continue
        else:
            done = True
    return font_size, wh


def load_piece(file_name, cell_size, rat=None):
    if rat is None:
        rat = cell_size
    img_path = path.join(path.dirname(__file__), f"img/{file_name}.png")
    return pygame.transform.scale(
        pygame.image.load(img_path), (int(rat[0]), int(rat[1]))
    )

def load_render_images(env_inst):
    comm_rat = env_inst._comm_rat
    csize = env_inst.cell_size[0]
    overlap_csize = max(1.025 * csize, csize + 2)
    overlap_cell_size = (overlap_csize, overlap_csize)
    view_area_size = csize * env_inst.view_size + max(0.025 * csize, 2)
    view_area = pygame.Surface((view_area_size, view_area_size), pygame.SRCALPHA)
    view_area.fill((255, 204, 0, 80))  # notice the alpha value in the color
    render_images = {
        'wall': load_piece("wall", env_inst.cell_size, overlap_cell_size),
        'goal': load_piece("goal", env_inst.cell_size, (0.529375 * csize, 0.875 * csize)),
        'bgs': tuple([load_piece("bg_" + str(i), env_inst.cell_size, overlap_cell_size) for i in (1, 2)]),
        'open_door': load_piece("open_door", env_inst.cell_size, overlap_cell_size),
        'doors': tuple([load_piece(env_inst.colors[i] + "_door", env_inst.cell_size, overlap_cell_size) for i in
                        range(len(env_inst.colors))]),
        'keys': tuple(
            [load_piece(env_inst.colors[i] + "_key", env_inst.cell_size, (0.455 * csize, 0.875 * csize)) for i in
             range(len(env_inst.colors))]),
        'agents': tuple(
            [load_piece(env_inst.colors[i] + "_agent", env_inst.cell_size, (0.8 * csize, 0.8 * csize)) for i in
             range(len(env_inst.colors))]),
        'wall_comm': load_piece("wall", env_inst.cell_size, (0.9 * comm_rat * csize, 0.9 * comm_rat * csize)),
        'goal_comm': load_piece("goal", env_inst.cell_size, (0.605 * 0.9 * comm_rat * csize, 0.9 * comm_rat * csize)),
        'doors_comm': tuple([load_piece(env_inst.colors[i] + "_door", env_inst.cell_size,
                                        (0.9 * comm_rat * csize, 0.9 * comm_rat * csize)) for
                             i in range(len(env_inst.colors))]),
        'keys_comm': tuple([load_piece(env_inst.colors[i] + "_key", env_inst.cell_size,
                                       (0.52 * 0.9 * comm_rat * csize, 0.9 * comm_rat * csize))
                            for i in range(len(env_inst.colors))]),
        'agents_comm': tuple([load_piece(env_inst.colors[i] + "_agent", env_inst.cell_size,
                                         (0.9 * comm_rat * csize, 0.9 * comm_rat * csize))
                              for i in range(len(env_inst.colors))]),
        'doors_comm_all': load_piece("any_door", env_inst.cell_size,(0.9 * comm_rat * csize, 0.9 * comm_rat * csize)),
        'keys_comm_all': load_piece("any_key", env_inst.cell_size, (0.52 * 0.9 * comm_rat * csize, 0.9 * comm_rat * csize)),
        'agents_comm_all': load_piece("any_agent", env_inst.cell_size,(0.9 * comm_rat * csize, 0.9 * comm_rat * csize)),
        'comm_view': tuple([load_piece("comm1", env_inst.cell_size, (0.9 * 0.8 * 0.84625 * csize, 0.9 * 1.175 * 0.8 * csize)),
                            load_piece("comm2", env_inst.cell_size, (0.9 * 0.8 * 1.56875 * csize, 0.9 * 1.175 * 0.8 * csize)),
                            load_piece("comm3", env_inst.cell_size, (0.9 * 0.8 * 2.29 * csize, 0.9 * 1.175 * 0.8 * csize)),
                            load_piece("comm4", env_inst.cell_size, (0.9 * 0.8 * 1.56875 * csize, 0.9 * 1.8875 * 0.8 * csize)),
                            load_piece("comm6", env_inst.cell_size, (0.9 * 0.8 * 2.29 * csize, 0.9 * 1.8875 * 0.8 * csize)),
                            load_piece("comm9", env_inst.cell_size, (0.9 * 0.8 * 2.29 * csize, 0.9 * 2.6 * 0.8 * csize))]),
        'view_area': view_area,
    }
    for kk in ('wall_comm', 'goal_comm', 'doors_comm', 'keys_comm', 'agents_comm'):
        if isinstance(render_images[kk], tuple):
            for img in render_images[kk]:
                img.set_alpha(140)
        else:
            render_images[kk].set_alpha(140)

    return render_images


def initiate_rendering(env_inst, num_agents, window_height):
    pygame.init()
    pygame.font.init()
    # search text size:
    font_size, font_height = find_font_size(num_agents, env_inst.vocab, 0.036 * window_height, window_height)
    env_inst.text_slot = 1.25 * font_height
    text_height = num_agents * env_inst.text_slot
    env_inst.font_size = font_size
    env_inst.pg_clock = None
    env_inst.window_surface = None

    env_inst.WINDOW_SIZE = (window_height - text_height, window_height)
    csize = env_inst.WINDOW_SIZE[0] / (env_inst.map_size + 2)
    env_inst.cell_size = (csize, csize)
    env_inst.render_view_size = csize * (2 * env_inst.view_range + 1) + max(0.025 * csize, 2)

    comm_rat = 0.9 * 0.52
    env_inst._comm_view_size = comm_rat * csize
    env_inst._key_comm_shift = 0.24 * 0.9 * comm_rat * csize
    env_inst._goal_comm_shift = 0.1975 * 0.9 * comm_rat * csize
    env_inst._space_comm = 0.1 * comm_rat * csize
    env_inst._comm_rat = comm_rat
    render_images = load_render_images(env_inst)

    env_inst.render_offs = {
        'goal': (0.2353125 * csize, 0.0625 * csize),
        'keys': (0.2725 * csize, 0.0625 * csize),
        'agents': (0.1 * csize, 0.1 * csize)
    }

    return render_images

def _render(env_inst, state, render_mode='rgb_array', render_images=None):
    assert env_inst.render_init
    if render_images is None:
        render_images = load_render_images(env_inst)
    wall_map = np.array(env_inst.wall_map).squeeze()
    doors = np.array(state.doors)
    cl_doors = np.array(state.cl_doors)
    keys = np.array(state.keys)
    decoy_keys = np.array(state.decoy_keys)
    cl_keys = np.array(state.cl_keys)
    goal = np.array(state.goal)
    agent_locs = np.array(state.agent_locs)
    doors_state = np.array(state.doors_state)
    raw_comm = np.array(state.raw_comm)
    comm_actions = np.array(state.comm_actions)

    if render_mode == "human":
        if env_inst.window_surface is None:
            if env_inst.pg_clock is None:
                env_inst.pg_clock = pygame.time.Clock()
            pygame.display.init()
            pygame.display.set_caption("Multi Maze")
            env_inst.window_surface = pygame.display.set_mode(env_inst.WINDOW_SIZE)
        window_surface = env_inst.window_surface
    elif render_mode == "rgb_array":
        window_surface = pygame.Surface(env_inst.WINDOW_SIZE)
    cell_size = env_inst.cell_size
    cl_colorid = env_inst.cl_colorid
    render_offs = env_inst.render_offs
    wall_img = render_images['wall']
    bg_imgs = render_images['bgs']
    for i in range(env_inst.map_size + 2):
        for j in range(env_inst.map_size + 2):
            if i == 0 or j == 0 or i == env_inst.map_size + 1 or j == env_inst.map_size + 1:
                window_surface.blit(wall_img, (i * cell_size[0], j * cell_size[1]))
            else:
                if wall_map[i - 1][j - 1]:
                    window_surface.blit(wall_img, (i * cell_size[0], j * cell_size[1]))
                elif (i % 2 == 0) ^ (j % 2 == 0):
                    window_surface.blit(bg_imgs[0],
                                             (i * cell_size[0], j * cell_size[1]))
                else:
                    window_surface.blit(bg_imgs[1],
                                             (i * cell_size[0], j * cell_size[1]))
    open_door_img = render_images['open_door']
    door_imgs = render_images['doors']
    for i, loc in enumerate(doors):
        if np.any(loc < 0):
            continue
        if doors_state[i]:
            window_surface.blit(open_door_img,
                                     ((loc[0] + 1) * cell_size[0], (loc[1] + 1) * cell_size[1]))
        else:
            window_surface.blit(door_imgs[i],
                                     ((loc[0] + 1) * cell_size[0], (loc[1] + 1) * cell_size[1]))
    for i, loc in enumerate(cl_doors):
        if doors_state[i + len(doors)]:
            window_surface.blit(open_door_img,
                                ((loc[0] + 1) * cell_size[0], (loc[1] + 1) * cell_size[1]))
        else:
            window_surface.blit(door_imgs[cl_colorid],
                                ((loc[0] + 1) * cell_size[0], (loc[1] + 1) * cell_size[1]))
    key_imgs = render_images['keys']
    for i, loc in enumerate(keys):
        if np.any(loc < 0):
            for lc in decoy_keys:
                window_surface.blit(key_imgs[i],
                                    ((lc[0] + 1) * cell_size[0] + render_offs['keys'][0],
                                     (lc[1] + 1) * cell_size[1] + render_offs['keys'][1]))
        else:
            window_surface.blit(key_imgs[i],
                                ((loc[0] + 1) * cell_size[0] + render_offs['keys'][0],
                                 (loc[1] + 1) * cell_size[1] + render_offs['keys'][1]))
    for i, loc in enumerate(cl_keys):
        window_surface.blit(key_imgs[cl_colorid],
                            ((loc[0] + 1) * cell_size[0] + render_offs['keys'][0],
                             (loc[1] + 1) * cell_size[1] + render_offs['keys'][1]))

    goal_img = render_images['goal']
    window_surface.blit(goal_img,
                             ((goal[0] + 1) * cell_size[0] + render_offs['goal'][0],
                              (goal[1] + 1) * cell_size[1] + render_offs['goal'][1]))

    comm_renders = []
    agent_imgs = render_images['agents']

    font_words = pygame.font.SysFont('Helvetica', env_inst.font_size)
    curr_words = []
    all_comm_objects = env_inst.comm_to_map(np.concatenate((raw_comm, comm_actions), axis=1).flatten(), np)
    for i, loc in enumerate(agent_locs):
        window_surface.blit(agent_imgs[i],
                                 ((loc[0] + 1) * cell_size[0] + render_offs['agents'][0],
                                  (loc[1] + 1) * cell_size[1] + render_offs['agents'][1]))
        idxs = np.arange(len(env_inst.pos2txt))[np.logical_and(comm_actions[i][0], comm_actions[i][1:])]
        curr_cw = []
        curr_ow = []
        for idx in idxs:
            if idx < len(env_inst.object_words):
                curr_ow.append(env_inst.pos2txt[idx])
            else:
                curr_cw.append(env_inst.pos2txt[idx])
        curr_words.append(curr_ow + curr_cw)
        ocount = 0
        olist = []
        otype = []
        n_obj_dict = {}
        window_surface.blit(render_images['view_area'], (
        (loc[0] + 1 - env_inst.view_range) * cell_size[0], (loc[1] + 1 - env_inst.view_range) * cell_size[1]))
        for k, _v in all_comm_objects[i].items():
            if k == '__all__':
                continue
            v = _v.sum(axis=-1).sum(axis=-1).flatten() > 0
            n_obj = np.sum(v)
            n_obj_dict[k] = n_obj
            ocount += n_obj
            if n_obj > 0:
                if k in ['wall', 'goal']:
                    olist.append(render_images[k + '_comm'])
                    otype.append(k)
                else:
                    for j in np.arange(len(v))[v]:
                        otype.append(k)
                        if (k == 'doors' and j >= len(doors)) or (k == 'keys' and j >= len(keys)):
                            olist.append(render_images[k + '_comm'][cl_colorid])
                        else:
                            olist.append(render_images[k + '_comm'][j])
        for k, _v in all_comm_objects[i]['__all__'].items():

            if n_obj_dict[k] <= 0:
                v = _v.sum() > 0
                n_obj = v
                ocount += n_obj
                if n_obj > 0:
                    otype.append(k)
                    olist.append(render_images[k + '_comm_all'])

        _comm_renders = []
        if ocount > 0:
            if ocount == 1:
                img_id = 0
                cmod = ocount
                comm_loc = ((loc[0] + 1.648 + 0.0677) * cell_size[0] + render_offs['agents'][0],
                            (loc[1] + 0.35 + 0.094) * cell_size[1] + render_offs['agents'][1])
            elif ocount == 2:
                img_id = 1
                cmod = ocount
                comm_loc = ((loc[0] + 1.083 + 0.1255) * cell_size[0] + render_offs['agents'][0],
                            (loc[1] + 0.35 + 0.094) * cell_size[1] + render_offs['agents'][1])
            elif ocount == 3:
                img_id = 2
                cmod = ocount
                comm_loc = ((loc[0] + 0.507 + 0.1832) * cell_size[0] + render_offs['agents'][0],
                            (loc[1] + 0.35 + 0.094) * cell_size[1] + render_offs['agents'][1])
            elif ocount == 4:
                img_id = 3
                cmod = 2
                comm_loc = ((loc[0] + 1.083 + 0.1255) * cell_size[0] + render_offs['agents'][0],
                            (loc[1] - 0.21 + 0.151) * cell_size[1] + render_offs['agents'][1])
            elif ocount <= 6:
                img_id = 4
                cmod = 3
                comm_loc = ((loc[0] + 0.507 + 0.1832) * cell_size[0] + render_offs['agents'][0],
                            (loc[1] - 0.21 + 0.151) * cell_size[1] + render_offs['agents'][1])
            else:
                img_id = 5
                cmod = 3
                comm_loc = ((loc[0] + 0.507 + 0.1832) * cell_size[0] + render_offs['agents'][0],
                            (loc[1] - 0.77 + 0.208) * cell_size[1] + render_offs['agents'][1])

            _comm_renders.append((render_images['comm_view'][img_id], comm_loc))

            for jj, oo, tt in zip(np.arange(ocount), olist, otype):
                _comm_renders.append((oo, (
                comm_loc[0] + 2.3 * env_inst._space_comm + (jj % cmod) * (env_inst._comm_view_size + env_inst._space_comm) + (
                            tt == 'keys') * env_inst._key_comm_shift + (tt == 'goal') * env_inst._goal_comm_shift,
                comm_loc[1] + 1.7 * env_inst._space_comm + (jj // cmod) * (env_inst._comm_view_size + env_inst._space_comm))))
            comm_renders.append(_comm_renders)

        for acomm in comm_renders:
            for ocomm in acomm:
                window_surface.blit(*ocomm)

    pygame.draw.rect(
        window_surface,
        (221, 235, 255),
        pygame.Rect(
            0,
            env_inst.WINDOW_SIZE[0],
            env_inst.WINDOW_SIZE[0],
            env_inst.WINDOW_SIZE[1] - env_inst.WINDOW_SIZE[0]
        ),
    )

    for i in range(len(agent_locs)):
        blit_text(window_surface, curr_words[i],
                  (5, i * env_inst.text_slot + env_inst.WINDOW_SIZE[0] + 0.1 * env_inst.text_slot), font_words,
                  color=pygame.Color(env_inst.colors[i]))

    if render_mode == "human":
        pygame.event.pump()
        pygame.display.update()
        env_inst.pg_clock.tick(env_inst.render_fps)
    elif render_mode == "rgb_array":
        return np.transpose(np.array(pygame.surfarray.pixels3d(window_surface)), axes=(1, 0, 2))
