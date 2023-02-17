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
        self.factory_assignments = {}
        self.heavy_assignments = {}
        self.mini_assignments = {}

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
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
                ore_tiles = np.argwhere(game_state.board.ore==1)
                for potential_factory_tile in potential_factory_tiles:
                    ore_tile_distances = np.mean((ore_tiles - potential_factory_tile)**2,1)
                    closest_ore_distance = np.min(ore_tile_distances)
                    #closest_ore_tile = np.argmin(ore_tile_distances) don't actually use these
                    #closest_ore_tile_pos = ore_tiles[closest_ore_tile]
                    if(closest_ore_distance < smallest_distance):
                        smallest_distance = closest_ore_distance
                        best_factory_tiles = [potential_factory_tile]
                    elif(closest_ore_distance == smallest_distance):
                        best_factory_tiles.append(potential_factory_tile)

                random_selection = np.random.randint(0,len(best_factory_tiles))
                spawn_loc = best_factory_tiles[random_selection]
                return dict(spawn=spawn_loc, metal=150, water=150)
            return dict()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = dict()
        game_state = obs_to_game_state(step, self.env_cfg, obs)

        #####functions##############################
        def mine(unit_id, unit, actions):
            if(unit.power < 350):
                if(on_factory or in_outer_edge_of_factory):
                    actions[unit_id] = [unit.pickup(4, 700)]
                else:
                    direction = direction_to(unit.pos, closest_factory_tile)
                    move_cost = unit.move_cost(game_state, direction)
                    if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                        actions[unit_id] = [unit.move(direction, repeat=0)]

            elif unit.cargo.ore < 100:
                ore_tile_distances = np.mean((ore_tile_locations - unit.pos) ** 2, 1)
                closest_ore_tile = ore_tile_locations[np.argmin(ore_tile_distances)]
                if np.all(closest_ore_tile == unit.pos): # unit is on ore
                    if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                        actions[unit_id] = [unit.dig(repeat=0)]
                else:
                    direction = direction_to(unit.pos, closest_ore_tile)
                    move_cost = unit.move_cost(game_state, direction)
                    if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                        actions[unit_id] = [unit.move(direction, repeat=0)]
            # else if we have enough ice, we go back to the factory and dump it.
            elif unit.cargo.ore >= 100:
                direction = direction_to(unit.pos, closest_factory_tile)
                if adjacent_to_factory or on_factory or in_outer_edge_of_factory:
                    if unit.power >= unit.action_queue_cost(game_state):
                        actions[unit_id] = [unit.transfer(direction, 1, unit.cargo.ore, repeat=0)]
                else:
                    move_cost = unit.move_cost(game_state, direction)
                    if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                        actions[unit_id] = [unit.move(direction, repeat=0)]
            return actions

        def ice(unit_id, unit, actions):
            actions[unit_id] = [unit.recharge(3000, repeat=0)]
            return actions
        ####################################################

        if not self.factory_assignments: #empty dict that has not been populated yet.
            for factory_id in game_state.factories[self.player]:
                self.factory_assignments[factory_id] = {'miners':[],
                                                        'icers':[]}

        factories = game_state.factories[self.player]
        factory_tiles, factory_units = [], []
        for unit_id, factory in factories.items():
            if factory.power >= self.env_cfg.ROBOTS["HEAVY"].POWER_COST and \
            factory.cargo.metal >= self.env_cfg.ROBOTS["HEAVY"].METAL_COST:
                actions[unit_id] = factory.build_heavy()
            if self.env_cfg.max_episode_length - game_state.real_env_steps < 50:
                if factory.water_cost(game_state) <= factory.cargo.water:
                    actions[unit_id] = factory.water()
            factory_tiles += [factory.pos]
            factory_units += [factory]
        factory_tiles = np.array(factory_tiles)

        units = game_state.units[self.player]
        ore_map = game_state.board.ore
        ore_tile_locations = np.argwhere(ore_map == 1)
        for unit_id, unit in units.items():

            # track the closest factory
            closest_factory = None
            adjacent_to_factory = False
            if len(factory_tiles) > 0:
                factory_distances = np.mean((factory_tiles - unit.pos) ** 2, 1)
                closest_factory_tile = factory_tiles[np.argmin(factory_distances)]
                closest_factory = factory_units[np.argmin(factory_distances)]
                adjacent_to_factory = np.mean((closest_factory_tile - unit.pos) ** 2) == 2 # change from 0 to 2. Will be adjacent and can "transfer" if distance is 2. This only transfers to center of factory though
                in_outer_edge_of_factory = np.mean((closest_factory_tile - unit.pos) ** 2) == 0.5
                on_factory = np.mean((closest_factory_tile - unit.pos) ** 2) == 0

                # determine unit job first
                # every factory needs one ore miner, then one water miner, then rest are ore miners?
                if(unit_id not in self.heavy_assignments and unit.unit_type == 'HEAVY'):
                    #find a job
                    closest_factory_id = closest_factory.unit_id
                    if(len(self.factory_assignments[closest_factory_id]['miners']) < 1): # every factory has at least 1 miner, then at least 1 icer, than rest miners
                        self.heavy_assignments[unit_id] = 'miner'
                        self.factory_assignments[closest_factory_id]['miners'].append(unit_id) # need an "update ledgers" function to update factory and heavy assignments and keep them consistent
                    elif(len(self.factory_assignments[closest_factory_id]['icers']) < 1):
                        self.heavy_assignments[unit_id] = 'icer'
                        self.factory_assignments[closest_factory_id]['icers'].append(unit_id)
                    else:
                        self.heavy_assignments[unit_id] = 'miner' # rest are miners for now
                        self.factory_assignments[closest_factory_id]['miners'].append(unit_id)

                if(self.heavy_assignments[unit_id]=='miner'):
                    actions = mine(unit_id, unit, actions)
                elif(self.heavy_assignments[unit_id]=='icer'):
                    actions = ice(unit_id, unit, actions)
                else:
                    actions[unit_id] = [unit.move(2, repeat=0)]
                    

        return actions