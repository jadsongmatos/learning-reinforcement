import gym
from queue import deque
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class Snake():

    """
    The Snake class holds all pertinent information regarding the Snake's movement and boday.
    The position of the snake is tracked using a queue that stores the positions of the body.

    Note:
    A potentially more space efficient implementation could track directional changes rather
    than tracking each location of the snake's body.
    """

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __init__(self, head_coord_start, length=3):
        """
        head_coord_start - tuple, list, or ndarray denoting the starting coordinates for the snake's head
        length - starting number of units in snake's body
        """

        self.direction = self.DOWN
        self.head = np.asarray(head_coord_start).astype(np.int64)
        self.head_color = np.array([255,0,0], np.uint8)
        self.body = deque()
        for i in range(length-1, 0, -1):
            self.body.append(self.head-np.asarray([0,i]).astype(np.int64))

    def step(self, coord, direction):
        """
        Takes a step in the specified direction from the specified coordinate.

        coord - list, tuple, or numpy array
        direction - integer from 1-4 inclusive.
            0: up
            1: right
            2: down
            3: left
        """

        assert direction < 4 and direction >= 0

        if direction == self.UP:
            return np.asarray([coord[0], coord[1]-1]).astype(np.int64)
        elif direction == self.RIGHT:
            return np.asarray([coord[0]+1, coord[1]]).astype(np.int64)
        elif direction == self.DOWN:
            return np.asarray([coord[0], coord[1]+1]).astype(np.int64)
        else:
            return np.asarray([coord[0]-1, coord[1]]).astype(np.int64)

    def action(self, direction):
        """
        This method sets a new head coordinate and appends the old head
        into the body queue. The Controller class handles popping the
        last piece of the body if no food is eaten on this step.

        The direction can be any integer value, but will be collapsed
        to 0, 1, 2, or 3 corresponding to up, right, down, left respectively.

        direction - integer from 0-3 inclusive.
            0: up
            1: right
            2: down
            3: left
        """

        # Ensure direction is either 0, 1, 2, or 3
        direction = (int(direction) % 4)

        if np.abs(self.direction-direction) != 2:
            self.direction = direction

        self.body.append(self.head)
        self.head = self.step(self.head, self.direction)

        return self.head

class Grid():

    """
    This class contains all data related to the grid in which the game is contained.
    The information is stored as a numpy array of pixels.
    The grid is treated as a cartesian [x,y] plane in which [0,0] is located at
    the upper left most pixel and [max_x, max_y] is located at the lower right most pixel.

    Note that it is assumed spaces that can kill a snake have a non-zero value as their 0 channel.
    It is also assumed that HEAD_COLOR has a 255 value as its 0 channel.
    """

    BODY_COLOR = np.array([1,0,0], dtype=np.uint8)
    HEAD_COLOR = np.array([255, 0, 0], dtype=np.uint8)
    FOOD_COLOR = np.array([0,0,255], dtype=np.uint8)
    SPACE_COLOR = np.array([0,255,0], dtype=np.uint8)

    def __init__(self, grid_size=[30,30], unit_size=10, unit_gap=1):
        """
        grid_size - tuple, list, or ndarray specifying number of atomic units in
                    both the x and y direction
        unit_size - integer denoting the atomic size of grid units in pixels
        """

        self.unit_size = int(unit_size)
        self.unit_gap = unit_gap
        self.grid_size = np.asarray(grid_size, dtype=np.int64) # size in terms of units
        height = self.grid_size[1]*self.unit_size
        width = self.grid_size[0]*self.unit_size
        channels = 3
        self.grid = np.zeros((height, width, channels), dtype=np.uint8)
        self.grid[:,:,:] = self.SPACE_COLOR
        self.open_space = grid_size[0]*grid_size[1]

    def check_death(self, head_coord):
        """
        Checks the grid to see if argued head_coord has collided with a death space (i.e. snake or wall)

        head_coord - x,y integer coordinates as a tuple, list, or ndarray
        """
        return self.off_grid(head_coord) or self.snake_space(head_coord)

    def color_of(self, coord):
        """
        Returns the color of the specified coordinate

        coord - x,y integer coordinates as a tuple, list, or ndarray
        """

        return self.grid[int(coord[1]*self.unit_size), int(coord[0]*self.unit_size), :]

    def connect(self, coord1, coord2, color=BODY_COLOR):
        """
        Draws connection between two adjacent pieces using the specified color.
        Created to indicate the relative ordering of the snake's body.
        coord1 and coord2 must be adjacent.

        coord1 - x,y integer coordinates as a tuple, list, or ndarray
        coord2 - x,y integer coordinates as a tuple, list, or ndarray
        color - [R,G,B] values as a tuple, list, or ndarray
        """

        # Check for adjacency
        # Next to one another:
        adjacency1 = (np.abs(coord1[0]-coord2[0]) == 1 and np.abs(coord1[1]-coord2[1]) == 0)
        # Stacked on one another:
        adjacency2  = (np.abs(coord1[0]-coord2[0]) == 0 and np.abs(coord1[1]-coord2[1]) == 1)
        assert adjacency1 or adjacency2

        if adjacency1: # x values differ
            min_x, max_x = sorted([coord1[0], coord2[0]])
            min_x = min_x*self.unit_size+self.unit_size-self.unit_gap
            max_x = max_x*self.unit_size
            self.grid[coord1[1]*self.unit_size, min_x:max_x, :] = color
            self.grid[coord1[1]*self.unit_size+self.unit_size-self.unit_gap-1, min_x:max_x, :] = color
        else: # y values differ
            min_y, max_y = sorted([coord1[1], coord2[1]])
            min_y = min_y*self.unit_size+self.unit_size-self.unit_gap
            max_y = max_y*self.unit_size
            self.grid[min_y:max_y, coord1[0]*self.unit_size, :] = color
            self.grid[min_y:max_y, coord1[0]*self.unit_size+self.unit_size-self.unit_gap-1, :] = color

    def cover(self, coord, color):
        """
        Colors a single space on the grid. Use erase if creating an empty space on the grid.
        This function is used like draw but without affecting the open_space count.

        coord - x,y integer coordinates as a tuple, list, or ndarray
        color - [R,G,B] values as a tuple, list, or ndarray
        """

        if self.off_grid(coord):
            return False
        x = int(coord[0]*self.unit_size)
        end_x = x+self.unit_size-self.unit_gap
        y = int(coord[1]*self.unit_size)
        end_y = y+self.unit_size-self.unit_gap
        self.grid[y:end_y, x:end_x, :] = np.asarray(color, dtype=np.uint8)
        return True

    def draw(self, coord, color):
        """
        Colors a single space on the grid. Use erase if creating an empty space on the grid.
        Affects the open_space count.

        coord - x,y integer coordinates as a tuple, list, or ndarray
        color - [R,G,B] values as a tuple, list, or ndarray
        """

        if self.cover(coord, color):
            self.open_space -= 1
            return True
        else:
            return False


    def draw_snake(self, snake, head_color=HEAD_COLOR):
        """
        Draws a snake with the given head color.

        snake - Snake object
        head_color - [R,G,B] values as a tuple, list, or ndarray
        """

        self.draw(snake.head, head_color)
        prev_coord = None
        for i in range(len(snake.body)):
            coord = snake.body.popleft()
            self.draw(coord, self.BODY_COLOR)
            if prev_coord is not None:
                self.connect(prev_coord, coord, self.BODY_COLOR)
            snake.body.append(coord)
            prev_coord = coord
        self.connect(prev_coord, snake.head, self.BODY_COLOR)

    def erase(self, coord):
        """
        Colors the entire coordinate with SPACE_COLOR to erase potential
        connection lines.

        coord - (x,y) as tuple, list, or ndarray
        """
        if self.off_grid(coord):
            return False
        self.open_space += 1
        x = int(coord[0]*self.unit_size)
        end_x = x+self.unit_size
        y = int(coord[1]*self.unit_size)
        end_y = y+self.unit_size
        self.grid[y:end_y, x:end_x, :] = self.SPACE_COLOR
        return True

    def erase_connections(self, coord):
        """
        Colors the dead space of the given coordinate with SPACE_COLOR to erase potential
        connection lines

        coord - (x,y) as tuple, list, or ndarray
        """

        if self.off_grid(coord):
            return False
        # Erase Horizontal Row Below Coord
        x = int(coord[0]*self.unit_size)
        end_x = x+self.unit_size
        y = int(coord[1]*self.unit_size)+self.unit_size-self.unit_gap
        end_y = y+self.unit_gap
        self.grid[y:end_y, x:end_x, :] = self.SPACE_COLOR

        # Erase the Vertical Column to Right of Coord
        x = int(coord[0]*self.unit_size)+self.unit_size-self.unit_gap
        end_x = x+self.unit_gap
        y = int(coord[1]*self.unit_size)
        end_y = y+self.unit_size
        self.grid[y:end_y, x:end_x, :] = self.SPACE_COLOR

        return True

    def erase_snake_body(self, snake):
        """
        Removes the argued snake's body and head from the grid.

        snake - Snake object
        """

        for i in range(len(snake.body)):
            self.erase(snake.body.popleft())

    def food_space(self, coord):
        """
        Checks if argued coord is snake food

        coord - x,y integer coordinates as a tuple, list, or ndarray
        """

        return np.array_equal(self.color_of(coord), self.FOOD_COLOR)

    def place_food(self, coord):
        """
        Draws a food at the coord. Ensures the same placement for
        each food at the beginning of a new episode. This is useful for
        experimentation with curiosity driven behaviors.

        num - the integer denoting the 
        """
        if self.open_space < 1 or not np.array_equal(self.color_of(coord), self.SPACE_COLOR):
            return False
        self.draw(coord, self.FOOD_COLOR)
        return True

    def new_food(self):
        """
        Draws a food on a random, open unit of the grid.
        Returns true if space left. Otherwise returns false.
        """

        if self.open_space < 1:
            return False
        coord_not_found = True
        while(coord_not_found):
            coord = (np.random.randint(0,self.grid_size[0]), np.random.randint(0,self.grid_size[1]))
            if np.array_equal(self.color_of(coord), self.SPACE_COLOR):
                coord_not_found = False
        self.draw(coord, self.FOOD_COLOR)
        return True

    def off_grid(self, coord):
        """
        Checks if argued coord is off of the grid

        coord - x,y integer coordinates as a tuple, list, or ndarray
        """

        return coord[0]<0 or coord[0]>=self.grid_size[0] or coord[1]<0 or coord[1]>=self.grid_size[1]

    def snake_space(self, coord):
        """
        Checks if argued coord is occupied by a snake

        coord - x,y integer coordinates as a tuple, list, or ndarray
        """

        color = self.color_of(coord)
        return np.array_equal(color, self.BODY_COLOR) or color[0] == self.HEAD_COLOR[0]

class Controller():
    """
    This class combines the Snake, Food, and Grid classes to handle the game logic.
    """

    def __init__(self, grid_size=[30,30], unit_size=10, unit_gap=1, snake_size=3, n_snakes=1, n_foods=1, random_init=True):

        assert n_snakes < grid_size[0]//3
        assert n_snakes < 25
        assert snake_size < grid_size[1]//2
        assert unit_gap >= 0 and unit_gap < unit_size

        self.snakes_remaining = n_snakes
        self.grid = Grid(grid_size, unit_size, unit_gap)

        self.snakes = []
        self.dead_snakes = []
        for i in range(1,n_snakes+1):
            start_coord = [i*grid_size[0]//(n_snakes+1), snake_size+1]
            self.snakes.append(Snake(start_coord, snake_size))
            color = [self.grid.HEAD_COLOR[0], i*10, 0]
            self.snakes[-1].head_color = color
            self.grid.draw_snake(self.snakes[-1], color)
            self.dead_snakes.append(None)

        if not random_init:
            for i in range(2,n_foods+2):
                start_coord = [i*grid_size[0]//(n_foods+3), grid_size[1]-5]
                self.grid.place_food(start_coord)
        else:
            for i in range(n_foods):
                self.grid.new_food()

    def move_snake(self, direction, snake_idx):
        """
        Moves the specified snake according to the game's rules dependent on the direction.
        Does not draw head and does not check for reward scenarios. See move_result for these
        functionalities.
        """

        snake = self.snakes[snake_idx]
        if type(snake) == type(None):
            return

        # Cover old head position with body
        self.grid.cover(snake.head, self.grid.BODY_COLOR)
        # Erase tail without popping so as to redraw if food eaten
        self.grid.erase(snake.body[0])
        # Find and set next head position conditioned on direction
        snake.action(direction)

    def move_result(self, direction, snake_idx=0):
        """
        Checks for food and death collisions after moving snake. Draws head of snake if
        no death scenarios.
        """

        snake = self.snakes[snake_idx]
        if type(snake) == type(None):
            return 0

        # Check for death of snake
        if self.grid.check_death(snake.head):
            self.dead_snakes[snake_idx] = self.snakes[snake_idx]
            self.snakes[snake_idx] = None
            self.grid.cover(snake.head, snake.head_color) # Avoid miscount of grid.open_space
            self.grid.connect(snake.body.popleft(), snake.body[0], self.grid.SPACE_COLOR)
            reward = -1
        # Check for reward
        elif self.grid.food_space(snake.head):
            self.grid.draw(snake.body[0], self.grid.BODY_COLOR) # Redraw tail
            self.grid.connect(snake.body[0], snake.body[1], self.grid.BODY_COLOR)
            self.grid.cover(snake.head, snake.head_color) # Avoid miscount of grid.open_space
            reward = 1
            self.grid.new_food()
        else:
            reward = 0
            empty_coord = snake.body.popleft()
            self.grid.connect(empty_coord, snake.body[0], self.grid.SPACE_COLOR)
            self.grid.draw(snake.head, snake.head_color)

        self.grid.connect(snake.body[-1], snake.head, self.grid.BODY_COLOR)

        return reward

    def kill_snake(self, snake_idx):
        """
        Deletes snake from game and subtracts from the snake_count 
        """
        
        assert self.dead_snakes[snake_idx] is not None
        self.grid.erase(self.dead_snakes[snake_idx].head)
        self.grid.erase_snake_body(self.dead_snakes[snake_idx])
        self.dead_snakes[snake_idx] = None
        self.snakes_remaining -= 1

    def step(self, directions):
        """
        Takes an action for each snake in the specified direction and collects their rewards
        and dones.

        directions - tuple, list, or ndarray of directions corresponding to each snake.
        """

        # Ensure no more play until reset
        if self.snakes_remaining < 1 or self.grid.open_space < 1:
            if type(directions) == type(int()) or len(directions) is 1:
                return self.grid.grid.copy(), 0, True, {"snakes_remaining":self.snakes_remaining}
            else:
                return self.grid.grid.copy(), [0]*len(directions), True, {"snakes_remaining":self.snakes_remaining}

        rewards = []

        if type(directions) == type(int()):
            directions = [directions]

        for i, direction in enumerate(directions):
            if self.snakes[i] is None and self.dead_snakes[i] is not None:
                self.kill_snake(i)
            self.move_snake(direction,i)
            rewards.append(self.move_result(direction, i))

        done = self.snakes_remaining < 1 or self.grid.open_space < 1
        if len(rewards) is 1:
            return self.grid.grid.copy(), rewards[0], done, {"snakes_remaining":self.snakes_remaining}
        else:
            return self.grid.grid.copy(), rewards, done, {"snakes_remaining":self.snakes_remaining}

class Discrete():
    def __init__(self, n_actions):
        self.dtype = np.int32
        self.n = n_actions
        self.actions = np.arange(self.n, dtype=self.dtype)
        self.shape = self.actions.shape

    def contains(self, argument):
        for action in self.actions:
            if action == argument:
                return True
        return False

    def sample(self):
        return np.random.choice(self.n)

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=[15,15], unit_size=10, unit_gap=1, snake_size=3, n_snakes=1, n_foods=1, random_init=True):
        self.grid_size = grid_size
        self.unit_size = unit_size
        self.unit_gap = unit_gap
        self.snake_size = snake_size
        self.n_snakes = n_snakes
        self.n_foods = n_foods
        self.viewer = None
        self.action_space = Discrete(4)
        self.random_init = random_init

    def step(self, action):
        self.last_obs, rewards, done, info = self.controller.step(action)
        return self.last_obs, rewards, done, info

    def reset(self):
        self.controller = Controller(self.grid_size, self.unit_size, self.unit_gap, self.snake_size, self.n_snakes, self.n_foods, random_init=self.random_init)
        self.last_obs = self.controller.grid.grid.copy()
        return self.last_obs

    def render(self, mode='human', close=False, frame_speed=.1):
        if self.viewer is None:
            self.fig = plt.figure()
            self.viewer = self.fig.add_subplot(111)
            plt.ion()
            self.fig.show()
        else:
            self.viewer.clear()
            self.viewer.imshow(self.last_obs)
            plt.pause(frame_speed)
        self.fig.canvas.draw()

    def seed(self, x):
        pass


# Initialize environment
env = SnakeEnv(grid_size=[15,15], unit_size=10, unit_gap=1, snake_size=3, n_snakes=1, n_foods=1, random_init=True)

# Reset the environment before starting
obs = env.reset()
"""
# Number of iterations to run the environment for
num_iterations = 100

for i in range(num_iterations):
    # Choose an action. Here, we're choosing a random action
    # For an actual reinforcement learning model, the action would be chosen by the model
    action = env.action_space.sample()

    # Take a step in the environment with the chosen action
    obs, reward, done, info = env.step(action)

    # Render the environment
    env.render()

    # If the episode is finished, reset the environment
    if done:
        obs = env.reset()

# Close the environment at the end
env.close()
"""


# Parameters
GAMMA = 0.95  # discount factor for target Q
LEARNING_RATE = 0.005
MEMORY_SIZE = 1000000  # size of replay memory
BATCH_SIZE = 20
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
EPISODES = 1000

# QNetwork
def build_model(state_size, action_size):
    model = Sequential()
    model.add(Dense(24, input_dim=state_size, activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(action_size, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(learning_rate=LEARNING_RATE))
    return model

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

model = build_model(state_size, action_size)
memory = deque(maxlen=MEMORY_SIZE)  # replay memory
exploration_rate = EXPLORATION_MAX

# Q-Learning procedure
for episode in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    while not done:
        if np.random.rand() < exploration_rate:  # Exploration
            action = env.action_space.sample()
        else:  # Exploitation
            Q_values = model.predict(state)
            action = np.argmax(Q_values[0])

        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        # Store experience in memory
        memory.append((state, action, reward, next_state, done))

        state = next_state

        if done:
            print("Episode: " + str(episode) + ", exploration: " + str(exploration_rate))
            break

        # Experience replay
        if len(memory) > BATCH_SIZE:
            minibatch = random.sample(memory, BATCH_SIZE)
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    target = reward + GAMMA * np.amax(model.predict(next_state)[0])
                target_f = model.predict(state)
                target_f[0][action] = target
                model.fit(state, target_f, epochs=1, verbose=0)

            # Exploration rate decay
            if exploration_rate > EXPLORATION_MIN:
                exploration_rate *= EXPLORATION_DECAY



