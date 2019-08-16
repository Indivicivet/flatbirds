import random
import time

import numpy as np
from skimage import transform

from vispy.scene import canvas
from vispy.scene import visuals

TIME_STEP = 0.1
PLAY_SPEED = 5

def vector_normalize(vec2):
    return np.array(vec2) / np.hypot(*vec2)

class Environment:
    def __init__(self, size=(600, 600)):
        self.size = size
        self.birds = []

        self.canvas = canvas.SceneCanvas(
            size=size,
            bgcolor=(0.9, 0.9, 0.8, 1),  # white-ish
            show=True
        )
        self.bird_visuals = []

    def rerender(self):
        for bird, vis in zip(self.birds, self.bird_visuals):
            vis.set_data(**bird.get_location())

    def _tick(self, time_step):
        for bird in self.birds:
            bird.tick(time_step=time_step, environment=self)
            bird.position = np.mod(bird.position, self.size)

    def tick(self, time_step=TIME_STEP):
        MAX_STEP = 0.5
        while time_step > MAX_STEP:
            self._tick(MAX_STEP)
            time_step -= MAX_STEP
        self._tick(time_step)
        self.rerender()

    def add_bird(self, bird):
        self.birds.append(bird)
        self.bird_visuals.append(bird.get_visual(parent=self.canvas.scene))

    def add_random_pos_bird(self, *tags):
        self.add_bird(
            Bird(
                *tags,
                position=[ (0.25 + 0.5 * random.random()) * dim for dim in self.size],
                facing=[(2 * random.random() - 1) for _ in range(2)],
            )
        )

class Bird:
    def __init__(self, *tags, position=(10, 10), facing=(1, 0), size=10):
        self.tags = tags
        self.position = np.array(position)
        self.size = size

        self.facing = vector_normalize(facing)
        self.velocity = 20
        self.max_turn_rate = 0.1

    def get_color(self):
        if "blah" in self.tags:
            return np.array((0.2, 0.2, 0.9, 1))
        return (0, 0, 0, 1)

    def get_location(self):
        return {
            "pos": np.stack((self.position, self.position + self.size * self.facing)),
            "arrows": np.hstack((self.position, self.position + self.size * self.facing)).reshape(1, 4),
        }

    def get_visual(self, parent=None):
        #return visuals.Text("bird", anchor_x="left", color="red", pos=np.array(self.position), parent=parent)
        return visuals.Arrow(
            # points
            **self.get_location(),
            color=np.stack([self.get_color()] * 2),
            # for arrowhead
            arrow_color = self.get_color(),
            arrow_type="angle_60",
            arrow_size=10,
            # add to scene
            parent=parent,
        )

    def tick(self, environment=None, time_step=TIME_STEP):
        # first turn
        if environment is not None:
            target_position = 0
            # nearby_birds = 0
            nearby_facing = np.zeros((2))
            for bird in environment.birds:
                if bird is not self:
                    target_position += bird.position
                    if np.hypot(*(bird.position - self.position)) < 20:
                        # nearby_birds += 1
                        nearby_facing += bird.facing
            target_position /= len(environment.birds) - 1
            
            # rotate to be near birds
            def vector_sin(a, b):
                return np.cross(a, b) / (np.hypot(*a) * np.hypot(*b))
            turning_sin = -vector_sin(target_position - self.position, self.facing)
            turn_by = np.clip(turning_sin, -self.max_turn_rate, self.max_turn_rate)
            def vector_rotate(v, theta):
                return np.dot([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], v)
            self.facing = vector_rotate(self.facing, turn_by)
            
            # rotate to be aligned with birds, a bit.
            self.facing = vector_normalize(self.facing + 0.2 * nearby_facing)
        # now move
        self.position += self.velocity * self.facing * time_step

if __name__ == "__main__":
    env = Environment()
    for i in range(30):
        env.add_random_pos_bird()
    
    time_stored = 0
    last_time = time.time()
    while True:
        time_stored += time.time() - last_time
        last_time = time.time()
        if time_stored > TIME_STEP:
            step_by = TIME_STEP * (time_stored // TIME_STEP)
            time_stored -= step_by
            env.tick(PLAY_SPEED * step_by)  # this does stupid big ticks if in background
        if "q" in input("press enter to update!"):n
            exit()
