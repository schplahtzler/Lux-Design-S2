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

        def find_0_rubble(x, y, rubble_board, already_explored, journey_count, limit):
            if journey_count == limit:
                return
            for i in [-1,0,1]:
                for j in [-1,0,1]:
                    new_x = x + i 
                    new_y = y + j
                    if((new_x)<0 or (new_x)>=48 or (new_y)<0 or (new_y)>=48):
                        continue 
                    if [new_x, new_y] in already_explored or rubble_board[new_x, new_y]!=0 or (new_x==0 and new_y==0) or (new_x==1 and new_y==1) or (new_x==-1 and new_y==-1) or (new_x==1 and new_y==-1) or (new_x==-1 and new_y==1):
                        continue 
                    else:
                        zero_rubble_tiles_found.append((new_x,new_y))
                        already_explored.append((new_x,new_y))
                        find_0_rubble(new_x, new_y, rubble_board, already_explored, journey_count+1, limit)

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
                    ice_tile_distances = np.mean((ice_tiles - potential_factory_tile)**2,1)
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
                    zero_rubble_tiles_found=[]
                    zero_rubble_spots = find_0_rubble(fac_tile[0],fac_tile[1],rubble_board,already_explored=[],journey_count=0,limit=3)
                    zero_rubble_spots=len(set(zero_rubble_tiles_found))
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
            if factory.power >= self.env_cfg.ROBOTS["HEAVY"].POWER_COST and \
            factory.cargo.metal >= self.env_cfg.ROBOTS["HEAVY"].METAL_COST:
                actions[unit_id] = factory.build_heavy()
            if game_state.real_env_steps > 700:
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

            if(unit.power < 150):
                if(on_factory or in_outer_edge_of_factory):
                    actions[unit_id] = [unit.pickup(4, 700)]
                else:
                    direction = direction_to(unit.pos, closest_factory_tile)
                    move_cost = unit.move_cost(game_state, direction)
                    if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                        actions[unit_id] = [unit.move(direction, repeat=0)]

            elif unit.cargo.ice < 100:
                ice_tile_distances = np.mean((ice_tile_locations - unit.pos) ** 2, 1)
                closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]
                if np.all(closest_ice_tile == unit.pos): # unit is on ice
                    if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                        if(unit_id not in actions):
                            actions[unit_id] = [unit.dig(repeat=50)] # repeat forever
                            
                else:
                    direction = direction_to(unit.pos, closest_ice_tile)
                    move_cost = unit.move_cost(game_state, direction)
                    if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                        actions[unit_id] = [unit.move(direction, repeat=0)]
            # else if we have enough ice, we go back to the factory and dump it.
            elif unit.cargo.ice >= 100:
                direction = direction_to(unit.pos, closest_factory_tile)
                if adjacent_to_factory or on_factory or in_outer_edge_of_factory:
                    if unit.power >= unit.action_queue_cost(game_state):
                        actions[unit_id] = [unit.transfer(direction, 0, unit.cargo.ice, repeat=0)]
                else:
                    move_cost = unit.move_cost(game_state, direction)
                    if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                        actions[unit_id] = [unit.move(direction, repeat=0)]
        return actions
                        

