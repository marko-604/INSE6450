import pyglet
from pyglet.window import key
import pyglet.gl as gl
import pyglet.graphics as graphics
import numpy as np
import time
import car
import math
import dynamics
import utils_driving as utils
import matplotlib.cm
import torch
import feature
import pickle
import sys
from car import Car


class Visualizer(object):
    def __init__(self, dt=0.5, fullscreen=False, name='unnamed', iters=1000, magnify=1.0):
        self.autoquit = False
        self.frame = None
        self.subframes = None
        self.visible_cars = []
        self.magnify = magnify
        self.camera_center = None
        self.name = name
        self.output = None
        self.iters = iters
        self.objects = []
        self.event_loop = pyglet.app.EventLoop()
        self.window = pyglet.window.Window(600, 600, fullscreen=fullscreen, caption=name)
        self.grass = pyglet.resource.texture('imgs/grass.png')
        self.window.on_draw = self.on_draw
        self.lanes = []
        self.cars = []
        self.dt = dt
        self.anim_x = {}
        self.prev_x = {}
        self.feed_u = None
        self.feed_x = None
        self.prev_t = None
        self.joystick = None
        self.keys = key.KeyStateHandler()
        self.window.push_handlers(self.keys)
        self.window.on_key_press = self.on_key_press
        self.main_car = None
        self.heat = None
        self.heatmap = None
        self.heatmap_valid = False
        self.heatmap_show = False
        self.cm = matplotlib.cm.jet
        self.paused = False

        self.label = pyglet.text.Label(
            'Speed: ',
            font_name='Times New Roman',
            font_size=24,
            x=30, y=self.window.height - 30,
            anchor_x='left', anchor_y='top'
        )

        def centered_image(filename):
            img = pyglet.resource.image(filename)
            img.anchor_x = img.width / 2.0
            img.anchor_y = img.height / 2.0
            return img

        def car_sprite(color, scale=0.15 / 600.0):
            sprite = pyglet.sprite.Sprite(centered_image('imgs/car-{}.png'.format(color)), subpixel=True)
            sprite.scale = scale
            return sprite

        def object_sprite(name, scale=0.15 / 600.0):
            sprite = pyglet.sprite.Sprite(centered_image('imgs/{}.png'.format(name)), subpixel=True)
            sprite.scale = scale
            return sprite

        self.sprites = {c: car_sprite(c) for c in ['white', 'orange']}

    def use_world(self, world):
        self.cars = [c for c in world.cars]
        self.lanes = [c for c in world.lanes]
        self.objects = [c for c in world.objects]

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            self.event_loop.exit()
        if symbol == key.SPACE:
            self.paused = not self.paused
        if symbol == key.D:
            self.reset()

    def control_loop(self, _=None):
        if self.paused:
            return

        if self.iters is not None and len(self.history_x[0]) >= self.iters:
            if self.autoquit:
                self.event_loop.exit()
            return

        if self.feed_u is not None and len(self.history_u[0]) >= len(self.feed_u[0]):
            if self.autoquit:
                self.event_loop.exit()
            return

        if self.pause_every is not None and self.pause_every > 0 and len(self.history_u[0]) % self.pause_every == 0:
            self.paused = True

        steer = 0.0
        gas = 0.0

        if self.keys[key.UP]:
            gas += 1.0
        if self.keys[key.DOWN]:
            gas -= 1.0
        if self.keys[key.LEFT]:
            steer += 1.5
        if self.keys[key.RIGHT]:
            steer -= 1.5

        if self.joystick:
            steer -= self.joystick.x * 3.0
            gas -= self.joystick.y

        self.heatmap_valid = False

        # store previous positions as numpy arrays for interpolation
        for c in self.cars:
            self.prev_x[c] = utils.extract(c.x)

        # choose control (keyboard/joystick or feed_u)
        if self.feed_u is None:
            for c in reversed(self.cars):
                c.control(steer, gas)
        else:
            for c, fu, hu in zip(self.cars, self.feed_u, self.history_u):
                c.u = fu[len(hu)]

        # log control history as numpy
        for c, hist in zip(self.cars, self.history_u):
            hist.append(utils.extract(c.u))

        # advance cars
        for c in self.cars:
            c.move()

        # log state history as numpy
        for c, hist in zip(self.cars, self.history_x):
            hist.append(utils.extract(c.x))

        self.prev_t = time.time()

    def center(self):
        if self.main_car is None:
            return np.asarray([0.0, 0.0])
        elif self.camera_center is not None:
            return np.asarray(self.camera_center[0:2])
        else:
            return self.anim_x[self.main_car][0:2]

    def camera(self):
        o = self.center()
        gl.glOrtho(
            o[0] - 1.0 / self.magnify, o[0] + 1.0 / self.magnify,
            o[1] - 1.0 / self.magnify, o[1] + 1.0 / self.magnify,
            -1.0, 1.0
        )

    def set_heat(self, f):
        """
        f is a Feature-like object: f(t, x, u) -> scalar (torch.Tensor or float).
        We wrap it into a function that takes p = [x, y] and returns a float.
        """
        def val(p):
            x = torch.tensor([p[0], p[1], 0.0, 0.0], dtype=torch.float32)
            u = torch.zeros(2, dtype=torch.float32)
            v = f(0.0, x, u)
            if isinstance(v, torch.Tensor):
                return float(v.item())
            return float(v)

        self.heat = val

    def draw_heatmap(self):
        if not self.heatmap_show:
            return

        SIZE = (256, 256)

        if not self.heatmap_valid:
            o = self.center()
            x0 = o - np.asarray([1.5, 1.5]) / self.magnify
            x0 = np.asarray([
                x0[0] - x0[0] % (1.0 / self.magnify),
                x0[1] - x0[1] % (1.0 / self.magnify)
            ])
            x1 = x0 + np.asarray([4.0, 4.0]) / self.magnify
            x0 = o - np.asarray([1.0, 1.0]) / self.magnify
            x1 = o + np.asarray([1.0, 1.0]) / self.magnify

            self.heatmap_x0 = x0
            self.heatmap_x1 = x1

            vals = np.zeros(SIZE)
            for i, xx in enumerate(np.linspace(x0[0], x1[0], SIZE[0])):
                for j, yy in enumerate(np.linspace(x0[1], x1[1], SIZE[1])):
                    vals[j, i] = self.heat(np.asarray([xx, yy]))

            vals = (vals - np.min(vals)) / (np.max(vals) - np.min(vals) + 1e-6)
            vals = self.cm(vals)
            vals[:, :, 3] = 0.7
            vals = (vals * 255.99).astype('uint8').flatten()
            vals = (gl.GLubyte * vals.size)(*vals)

            img = pyglet.image.ImageData(SIZE[0], SIZE[1], 'RGBA', vals, pitch=SIZE[1] * 4)
            self.heatmap = img.get_texture()
            self.heatmap_valid = True

        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        gl.glEnable(self.heatmap.target)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glBindTexture(self.heatmap.target, self.heatmap.id)
        gl.glEnable(gl.GL_BLEND)

        x0 = self.heatmap_x0
        x1 = self.heatmap_x1
        graphics.draw(
            4, gl.GL_QUADS,
            ('v2f', (x0[0], x0[1], x1[0], x0[1], x1[0], x1[1], x0[0], x1[1])),
            ('t2f', (0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0))
        )
        gl.glDisable(self.heatmap.target)

    def output_loop(self, _):
        if self.frame % self.subframes == 0:
            self.control_loop()

        alpha = float(self.frame % self.subframes) / float(self.subframes)
        for c in self.cars:
            current_x = utils.extract(c.x)
            self.anim_x[c] = (1 - alpha) * self.prev_x[c] + alpha * current_x

        self.frame += 1

    def animation_loop(self, _):
        t = time.time()
        alpha = min((t - self.prev_t) / self.dt, 1.0)
        for c in self.cars:
            current_x = utils.extract(c.x)
            self.anim_x[c] = (1 - alpha) * self.prev_x[c] + alpha * current_x

    def draw_lane_surface(self, lane):
        gl.glColor3f(0.4, 0.4, 0.4)
        W = 1000
        graphics.draw(
            4, gl.GL_QUAD_STRIP,
            ('v2f', np.hstack([
                lane.p - lane.m * W - 0.5 * lane.w * lane.n,
                lane.p - lane.m * W + 0.5 * lane.w * lane.n,
                lane.q + lane.m * W - 0.5 * lane.w * lane.n,
                lane.q + lane.m * W + 0.5 * lane.w * lane.n
            ]))
        )

    def draw_lane_lines(self, lane):
        gl.glColor3f(1.0, 1.0, 1.0)
        W = 1000
        graphics.draw(
            4, gl.GL_LINES,
            ('v2f', np.hstack([
                lane.p - lane.m * W - 0.5 * lane.w * lane.n,
                lane.p + lane.m * W - 0.5 * lane.w * lane.n,
                lane.p - lane.m * W + 0.5 * lane.w * lane.n,
                lane.p + lane.m * W + 0.5 * lane.w * lane.n
            ]))
        )

    def draw_car(self, x, color='orange', opacity=255):
        # x is stored as numpy array
        sprite = self.sprites[color]
        sprite.x, sprite.y = float(x[0]), float(x[1])
        sprite.rotation = -float(x[2]) * 180.0 / math.pi
        sprite.opacity = opacity
        sprite.draw()

    def on_draw(self):
        self.window.clear()
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        self.camera()

        gl.glEnable(self.grass.target)
        gl.glEnable(gl.GL_BLEND)
        gl.glBindTexture(self.grass.target, self.grass.id)
        W = 10000.0
        graphics.draw(
            4, gl.GL_QUADS,
            ('v2f', (-W, -W, W, -W, W, W, -W, W)),
            ('t2f', (0.0, 0.0, W * 5.0, 0.0, W * 5.0, W * 5.0, 0.0, W * 5.0))
        )
        gl.glDisable(self.grass.target)

        for lane in self.lanes:
            self.draw_lane_surface(lane)
        for lane in self.lanes:
            self.draw_lane_lines(lane)
        for obj in self.objects:
            self.draw_object(obj)
        for c in self.cars:
            if c != self.main_car and c not in self.visible_cars:
                self.draw_car(self.anim_x[c], c.color)
        if self.heat is not None:
            self.draw_heatmap()
        for c in self.cars:
            if c == self.main_car or c in self.visible_cars:
                self.draw_car(self.anim_x[c], c.color)

        gl.glPopMatrix()

        if isinstance(self.main_car, Car):
            self.label.text = 'Speed: %.2f' % float(self.anim_x[self.main_car][3])
            self.label.draw()

        if self.output is not None:
            pyglet.image.get_buffer_manager().get_color_buffer().save(self.output.format(self.frame))

    def draw_object(self, obj):
        # If you have object sprites, implement them here.
        pass

    def reset(self):
        for c in self.cars:
            c.reset()
        self.prev_t = time.time()
        for c in self.cars:
            x_np = utils.extract(c.x)
            self.prev_x[c] = x_np
            self.anim_x[c] = x_np
        self.paused = True
        self.history_x = [[] for _ in self.cars]
        self.history_u = [[] for _ in self.cars]

    def run(self, filename=None, pause_every=None):
        self.pause_every = pause_every
        self.reset()

        if filename is not None:
            with open(filename, 'rb') as f:
                self.feed_u, self.feed_x = pickle.load(f)

        if self.output is None:
            pyglet.clock.schedule_interval(self.animation_loop, 0.02)
            pyglet.clock.schedule_interval(self.control_loop, self.dt)
        else:
            self.paused = False
            self.subframes = 6
            self.frame = 0
            self.autoquit = True
            pyglet.clock.schedule(self.output_loop)

        self.event_loop.run()

    def run_modified(self, history_x, history_u):
        self.pause_every = None
        self.reset()
        self.feed_x = history_x
        self.feed_u = history_u
        pyglet.clock.schedule_interval(self.animation_loop, 0.02)
        pyglet.clock.schedule_interval(self.control_loop, self.dt)
        self.event_loop.run()


if __name__ == '__main__' and False:
    import lane
    dyn = dynamics.CarDynamics(0.1)
    vis = Visualizer(dyn.dt)
    vis.lanes.append(lane.StraightLane([0.0, -1.0], [0.0, 1.0], 0.13))
    vis.lanes.append(vis.lanes[0].shifted(1))
    vis.lanes.append(vis.lanes[0].shifted(-1))
    vis.cars.append(car.UserControlledCar(dyn, [0.0, 0.0, math.pi / 2.0, 0.1]))
    vis.cars.append(car.SimpleOptimizerCar(dyn, [0.0, 0.5, math.pi / 2.0, 0.0], color='red'))

    r = -60.0 * vis.cars[0].linear.gaussian()
    r = r + vis.lanes[0].gaussian()
    r = r + vis.lanes[1].gaussian()
    r = r + vis.lanes[2].gaussian()
    r = r - 30.0 * vis.lanes[1].shifted(1).gaussian()
    r = r - 30.0 * vis.lanes[2].shifted(-1).gaussian()
    r = r + 30.0 * feature.speed(0.5)
    r = r + 10.0 * vis.lanes[0].gaussian(10.0)
    r = r + 0.1 * feature.control()

    vis.cars[1].reward = r
    vis.main_car = vis.cars[0]
    vis.paused = True
    vis.set_heat(r)
    vis.run()

if __name__ == '__main__' and len(sys.argv) == 1:
    import world as wrld
    import car
    world = wrld.world2()
    vis = Visualizer(0.1, name='replay')
    vis.use_world(world)
    vis.main_car = world.cars[0]
    vis.run()

if __name__ == '__main__' and len(sys.argv) > 1:
    filename = sys.argv[1]
    import world
    world_name = (filename.split('/')[-1]).split('-')[0]
    magnify = 1.0
    if len(sys.argv) > 3:
        magnify = float(sys.argv[3])
    vis = Visualizer(0.2, name=world_name, magnify=magnify)
    the_world = getattr(world, world_name)()
    vis.use_world(the_world)
    vis.main_car = the_world.cars[0]
    if len(sys.argv) > 4:
        vis.camera_center = list(eval(sys.argv[4]))
    if len(sys.argv) > 2:
        pause_every = int(sys.argv[2])
        vis.run(filename, pause_every=pause_every)
    else:
        vis.run(filename)
