from lux.kit import obs_to_game_state, GameState
from lux.config import EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
import numpy as np
import sys
class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):

        def find_zero_rubble_tiles(x, y, rubble_board, orig_x=None, orig_y=None, explored_tiles=None, zero_rubble_tiles=None, factory_tiles=None): #ChatGPT
            if orig_x is None:
                orig_x = x 
            if orig_y is None:
                orig_y = y
            if explored_tiles is None:
                explored_tiles = {}
            if zero_rubble_tiles is None:
                zero_rubble_tiles = {}
            if factory_tiles is None:
                factory_tiles = {}
                for dx in [-1,0,1]:
                    for dy in [-1,0,1]:
                        factory_tiles[(x + dx, y + dy)] = True

            if x < 0 or x >= 48 or y < 0 or y >= 48 or (x, y) in explored_tiles:
                return zero_rubble_tiles
            if (rubble_board[x][y] != 0) and ((x,y) not in factory_tiles):
                return zero_rubble_tiles
            explored_tiles[(x, y)] = True
            if (x,y) not in factory_tiles:
                zero_rubble_tiles[(x, y)] = True
            for dx, dy in [[-1, 0], [0, 1], [1, 0], [0, -1]]:
                find_zero_rubble_tiles(x+dx, y+dy, rubble_board, orig_x, orig_y, explored_tiles, zero_rubble_tiles, factory_tiles)
            return zero_rubble_tiles

        if step == 0:
            # bid 0 to not waste resources bidding and declare as the default faction
            return dict(faction="AlphaStrike", bid=0)
        else:
            game_state = obs_to_game_state(step, self.env_cfg, obs)
            # factory placement period

            # how much water and metal you have in your starting pool to give to new factories
            water_left = game_state.teams[self.player].water
            metal_left = game_state.teams[self.player].metal

            # how many factories you have left to place
            factories_to_place = game_state.teams[self.player].factories_to_place
            # whether it is your turn to place a factory
            my_turn_to_place = my_turn_to_place_factory(game_state.teams[self.player].place_first, step)
            if factories_to_place > 0 and my_turn_to_place:

                
                potential_factory_tiles = list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1)))
                smallest_distance = np.inf
                best_factory_tiles = []
                ice_tiles = np.argwhere(game_state.board.ice==1)
                for potential_factory_tile in potential_factory_tiles:
                    ice_tile_distances = np.mean((ice_tiles - potential_factory_tile)**2,1) # NOTE this implementation only chooses tiles orthogonal to the center of the factory, not the knight position to factory, which is OK for this purpose. imrpve later.
                    closest_ice_distance = np.min(ice_tile_distances)
                    #closest_ore_tile = np.argmin(ore_tile_distances) don't actually use these
                    #closest_ore_tile_pos = ore_tiles[closest_ore_tile]
                    if(closest_ice_distance < smallest_distance):
                        smallest_distance = closest_ice_distance
                        best_factory_tiles = [potential_factory_tile]
                    elif(closest_ice_distance == smallest_distance):
                        best_factory_tiles.append(potential_factory_tile)


                #instead of random selection, get one with most non-rubble tiles
                rubble_board = game_state.board.rubble
                best_tiles = []
                most_0_rubble_spots = 0
                for fac_tile in best_factory_tiles:
                    zero_rubble_tiles_found = find_zero_rubble_tiles(fac_tile[0], fac_tile[1], rubble_board)
                    zero_rubble_spots=len(zero_rubble_tiles_found)
                    if(zero_rubble_spots > most_0_rubble_spots):
                        best_tiles=[fac_tile]
                        most_0_rubble_spots=zero_rubble_spots
                    elif(zero_rubble_spots==most_0_rubble_spots):
                        best_tiles.append(fac_tile)


                random_selection = np.random.randint(0,len(best_tiles))
                spawn_loc = best_tiles[random_selection]
                return dict(spawn=spawn_loc, metal=150, water=150)
            return dict()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = dict()
        game_state = obs_to_game_state(step, self.env_cfg, obs)

        factories = game_state.factories[self.player]
        factory_tiles, factory_units = [], []
        for unit_id, factory in factories.items():
            if factory.power >= self.env_cfg.ROBOTS["HEAVY"].POWER_COST and factory.cargo.metal >= self.env_cfg.ROBOTS["HEAVY"].METAL_COST:
                actions[unit_id] = factory.build_heavy()
            elif factory.power >= self.env_cfg.ROBOTS["LIGHT"].POWER_COST and factory.cargo.metal >= self.env_cfg.ROBOTS["LIGHT"].METAL_COST and factory.cargo.metal>40:
                actions[unit_id] = factory.build_light()
            if game_state.real_env_steps > 600:
                if (factory.water_cost(game_state)+20) <= factory.cargo.water:
                    actions[unit_id] = factory.water()
            factory_tiles += [factory.pos]
            factory_units += [factory]
        factory_tiles = np.array(factory_tiles)

        units = game_state.units[self.player]
        ore_map = game_state.board.ore
        ore_tile_locations = np.argwhere(ore_map == 1)
        ice_map = game_state.board.ice
        ice_tile_locations = np.argwhere(ice_map == 1)
        for unit_id, unit in units.items():

            # track the closest factory
            closest_factory = None
            adjacent_to_factory = False
            factory_distances = np.mean((factory_tiles - unit.pos) ** 2, 1)
            closest_factory_tile = factory_tiles[np.argmin(factory_distances)]
            closest_factory = factory_units[np.argmin(factory_distances)]
            adjacent_to_factory = np.mean((closest_factory_tile - unit.pos) ** 2) == 2 # change from 0 to 2. Will be adjacent and can "transfer" if distance is 2. This only transfers to center of factory though
            in_outer_edge_of_factory = np.mean((closest_factory_tile - unit.pos) ** 2) == 0.5
            on_factory = np.mean((closest_factory_tile - unit.pos) ** 2) == 0
            in_knight_formation = np.mean((closest_factory_tile - unit.pos) ** 2) == 2.5



            #find closest ice tile
            ice_tile_distances = np.mean((ice_tile_locations - unit.pos) ** 2, 1)
            closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]

            if(unit.unit_type=='LIGHT'):
                #find spot where unit can transfer power
                ice_location = np.array(closest_ice_tile)
                factory_location = np.array(closest_factory_tile)
                for dx,dy in [[-1,0],[0,1],[1,0],[0,-1]]: # find transfer spot for lil bot
                    spot = ice_location+np.array([dx,dy]) # sum the matricies together to form new spot
                    distance_to_fac = np.mean((factory_location - spot) ** 2)
                    if(distance_to_fac<2): # spot is inside the factory borders
                        transfer_spot = spot
                        break
                # if np.all(transfer_spot==unit.pos):
                #     # start pickup and transfer cycle
                #     if unit.power<150:
                #         direction = direction_to(unit.pos, ice_location)
                #         actions[unit_id] = [unit.pickup(4, 150,repeat=True)]
                #     else:
                #         direction = direction_to(unit.pos, ice_location)
                #         actions[unit_id] = [unit.transfer(direction,4,unit.power)]
                # else:
                #     direction = direction_to(unit.pos, transfer_spot)
                #     move_cost = unit.move_cost(game_state, direction)
                #     if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                #         actions[unit_id] = [unit.move(direction, repeat=0)]
                if np.all(transfer_spot==unit.pos):
                    if unit.action_queue.size==0:
                        direction = direction_to(unit.pos, ice_location)
                        actions[unit_id] = [unit.pickup(4, 150,repeat=1),unit.transfer(direction,4,150,repeat=1)]


                else:
                    direction = direction_to(unit.pos, transfer_spot)
                    move_cost = unit.move_cost(game_state, direction)
                    if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                        actions[unit_id] = [unit.move(direction, repeat=0)]



            elif(unit.unit_type=='HEAVY'):
                # if(unit.power < 150):
                #     if(on_factory or in_outer_edge_of_factory):
                #         actions[unit_id] = [unit.pickup(4, 700)]
                #     else:
                #         direction = direction_to(unit.pos, closest_factory_tile)
                #         move_cost = unit.move_cost(game_state, direction)
                #         if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                #             actions[unit_id] = [unit.move(direction, repeat=0)]

                #1. go to ice
                #2. dig and transfer forever
                if np.all(closest_ice_tile == unit.pos): # unit is on ice                    
                    if unit.action_queue.size==0:
                        direction = direction_to(unit.pos, closest_factory_tile)
                        actions[unit_id] = [unit.dig(n=20,repeat=10),unit.transfer(direction, 0, 3000, repeat=1)]
                else:
                    direction = direction_to(unit.pos, closest_ice_tile)
                    move_cost = unit.move_cost(game_state, direction)
                    if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                        actions[unit_id] = [unit.move(direction)]


                # if (unit.cargo.ice < 100):
                #     if np.all(closest_ice_tile == unit.pos): # unit is on ice
                #         if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                #             if(unit_id not in actions):
                #                 if not unit_id in actions: # don't update if the bot already has dig action
                #                     actions[unit_id] = [unit.dig(repeat=50)] # repeat forever
                                
                #     else:
                #         direction = direction_to(unit.pos, closest_ice_tile)
                #         move_cost = unit.move_cost(game_state, direction)
                #         if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                #             actions[unit_id] = [unit.move(direction, repeat=0)]
                # # else if we have enough ice, we go back to the factory and dump it.
                # elif (unit.cargo.ice >= 200):
                #     direction = direction_to(unit.pos, closest_factory_tile)
                #     if adjacent_to_factory or on_factory or in_outer_edge_of_factory or in_knight_formation:
                #         if unit.power >= unit.action_queue_cost(game_state):
                #             actions[unit_id] = [unit.transfer(direction, 0, unit.cargo.ice, repeat=0)]
                #     else:
                #         move_cost = unit.move_cost(game_state, direction)
                #         if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                #             actions[unit_id] = [unit.move(direction, repeat=0)]
        return actions
                        

