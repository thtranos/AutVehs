total_episodes = 2000
max_steps = 1200
train_steps = 1500

#FLAGS
PRINT_STATS = True
SAVE_AGENTS = True
TEST = False
gui = False
BUILD = True
random = True

ENSEMBLE = False
SYN = False

SCENARIO_AGENT = [0,1]
NETWORK = "train3"

GAMMA = 0.99
ENTROPY = 0.01
BATCH_SIZE = 250
BUFFER_SIZE = 200
LR_ACTOR = 0.00001
LR_CRITIC = 0.00001
WD_ACTOR = 1E-4
WD_CRITIC = 1E-4
LEARN_NUM = 100
NUMBER_OF_NEURONS = 256
EPS_CLIP = 0.1
ACTION_DIM = 1
STATE_DIM = 5
MIN_ACTION_STD = 0.01
ACTION_STD_INIT = 0.3
ACTION_STD_UPDATE = 25
ACTION_STD_DECAY_RATE = (ACTION_STD_INIT*ACTION_STD_UPDATE)/train_steps
SAVE_EVERY = 25
TRAIN_EVERY = 1
EVALUATION_EVERY = 25

max_speed = 20.0
min_speed = 0.0
safe_distance = 20.0

if NETWORK == 'small_network1':
	#test small_network1
	sumoBinary = "/opt/homebrew/Cellar/sumo/1.16.0/share/sumo/bin/sumo"
	sumoBinaryGui = "/opt/homebrew/Cellar/sumo/1.16.0/share/sumo/bin/sumo-gui"
	path = "networks/1_lane/small_network1/2hour_test"
	net_path = "/randomSmall.net.xml"
	trips_path = "/small.trips.xml"
	route_path = f"/small{SCENARIO_AGENT[0]}.rou.xml"
	sumocfg = f"/osm.small{SCENARIO_AGENT[0]}.sumocfg"

if NETWORK == 'network1':
	#test network1

	sumoBinary = "/opt/homebrew/Cellar/sumo/1.16.0/share/sumo/bin/sumo"
	sumoBinaryGui = "/opt/homebrew/Cellar/sumo/1.16.0/share/sumo/bin/sumo-gui"
	path = "networks/1_lane/network1/2hour_test"
	net_path = "/randomSmall.net.xml"
	trips_path = "/small.trips.xml"
	route_path = f"/small{SCENARIO_AGENT[0]}.rou.xml"
	sumocfg = f"/osm.small{SCENARIO_AGENT[0]}.sumocfg"

if NETWORK == '200_network':
	#test 200_network

	sumoBinary = "/opt/homebrew/Cellar/sumo/1.16.0/share/sumo/bin/sumo"
	sumoBinaryGui = "/opt/homebrew/Cellar/sumo/1.16.0/share/sumo/bin/sumo-gui"
	path = "networks/1_lane/200_network/2hour_test"
	net_path = "/200.net.xml"
	trips_path = "/small.trips.xml"
	route_path = f"/small{SCENARIO_AGENT[0]}.rou.xml"
	sumocfg = f"/osm.small{SCENARIO_AGENT[0]}.sumocfg"


if NETWORK == '100_network':
	#test network 100_network
	sumoBinary = "/opt/homebrew/Cellar/sumo/1.16.0/share/sumo/bin/sumo"
	sumoBinaryGui = "/opt/homebrew/Cellar/sumo/1.16.0/share/sumo/bin/sumo-gui"
	path = "networks/1_lane/100_network/2hour_test"
	net_path = "/100.net.xml"
	trips_path = "/small.trips.xml"
	route_path = f"/small{SCENARIO_AGENT[0]}.rou.xml"
	sumocfg = f"/osm.small{SCENARIO_AGENT[0]}.sumocfg"

if NETWORK == 'train3':

	sumoBinary = "/opt/homebrew/Cellar/sumo/1.16.0/share/sumo/bin/sumo"
	sumoBinaryGui = "/opt/homebrew/Cellar/sumo/1.16.0/share/sumo/bin/sumo-gui"
	path = "networks/1_lane/components/3_component/random"
	net_path = "/3.net.xml"
	trips_path = "/3.trips.xml"
	route_path = "/3.rou.xml"
	sumocfg = "/osm.small.sumocfg"

if NETWORK == 'train4':

	sumoBinary = "/opt/homebrew/Cellar/sumo/1.16.0/share/sumo/bin/sumo"
	sumoBinaryGui = "/opt/homebrew/Cellar/sumo/1.16.0/share/sumo/bin/sumo-gui"
	path = "networks/1_lane/components/4_component/random"
	net_path = "/4.net.xml"
	trips_path = "/4.trips.xml"
	route_path = "/4.rou.xml"
	sumocfg = "/osm.small.sumocfg"

save_name = "component_3_random_101"
