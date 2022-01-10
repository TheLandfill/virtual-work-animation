#!/usr/bin/env python3

from manim import *
import numpy as np
import math as m
from os.path import isfile

def field_func(pos):
    temp_pos = pos + 2.0 * RIGHT
    return ((5.0 * temp_pos[1] ** 2 * LEFT + temp_pos[0] * LEFT) - temp_pos) / 3

def path_func(t):
    phi = 2.0 * PI * t
    rho = 2.0 + np.cos(3.0 * phi)
    return np.array([ rho * np.cos(phi), rho * np.sin(phi), 0]) + 2.0 * LEFT

def dot(a, b):
    return sum(a[k] * b[k] for k in range(len(a)))

def sqr_magnitude(a):
    return dot(a, a)

def magnitude(a):
    return np.sqrt(sqr_magnitude(a))

def normalize(a):
    return a / magnitude(a)

def proj(a, b):
    return dot(a, normalize(b)) * b

def tan_vector(path, t):
    delta_t = 1e-8
    delta_delta_t = delta_t
    ds = path.point_from_proportion(t + delta_t) - path.point_from_proportion(t)
    while sqr_magnitude(ds) < 1e-12:
        delta_t += delta_delta_t
        ds = path.point_from_proportion(t + delta_t) - path.point_from_proportion(t)
    delta_s = normalize(ds)
    return delta_s

class Non_Manim_Rate_Func:
    def __init__(
        self,
        path : np.ndarray,
        field,
        total_time,
        filename
    ):
        self.total_time = total_time
        self.steps = []
        self.path = path
        if isfile(filename):
            with open(filename, 'r') as reader:
                self.steps = [ float(k) for k in reader ]
            self.total_time = 1.0
        else:
            self.steps = Non_Manim_Rate_Func.speed(path, field, total_time)
            with open(filename, 'w') as writer:
                for k in range(6000):
                    writer.write(str(self.point_from_proportion(k / 6000.0)) + "\n")

    def point_from_proportion(self, t):
        cur_index = m.modf(t * (len(self.steps) - 2))
        frac = cur_index[0]
        index = int(cur_index[1])
        points = self.steps[index:index + 2]
        alpha = m.modf((points[0] * (1.0 - frac) + points[1] * frac) / self.total_time)[0]
        return alpha

    def vel_from_proportion(self, t):
        cur_index = m.modf(t * (len(self.steps) - 1))
        index = int(cur_index[1])
        points = self.steps[index:index + 2]
        return (dot(tan_vector(self.path, sum(points) / 2.0), self.path.point_from_proportion(points[1]) - self.path.point_from_proportion(points[0]))) / 1e-3

    @staticmethod
    def speed(path, field, total_time):
        scaled_dist = [ 0.0 ]
        delta_t = 1e-3
        cur_speed = 0.05
        t = 0.0
        while t < total_time:
            cur_speed += Non_Manim_Rate_Func.acc(path, field, scaled_dist[-1]) * delta_t
            scaled_dist.append(scaled_dist[-1] + cur_speed * delta_t)
            t += delta_t
            print(str(t) + "\r\x1b[A")
        print()
        return scaled_dist

    @staticmethod
    def acc(path, field, alpha):
        alpha = m.modf(alpha)[0]
        pos = path.point_from_proportion(alpha)
        return 4.0 * dot(field_func(pos), tan_vector(path, alpha))

MELLOR_SCALAR_FIELD_COLORS: list = [
    "#80FFC8",
    "#89C7FF",
    "#8DA5FF",
    "#BD93FF",
    "#FF99FF",
    "#FC72E0",
    "#F74DB3",
    "#DC143C"
]
min_color_value = 0.0
max_color_value = 17.0

def vec_color(vec):
    mag = np.linalg.norm(vec)
    color_value = np.clip(
        mag,
        min_color_value,
        max_color_value
    )
    alpha = inverse_interpolate(
        min_color_value,
        max_color_value,
        color_value
    )
    colors = np.array(list(map(color_to_rgb, MELLOR_SCALAR_FIELD_COLORS)))
    alpha *= len(colors) - 1
    c1 = colors[int(alpha)]
    c2 = colors[min(int(alpha + 1), len(colors) - 1)]
    alpha %= 1.0
    return rgb_to_color(interpolate(c1, c2, alpha))

class Virtual_Work2D(Scene):
    def construct(self):
        path = ParametricFunction(path_func, color="#909090", sheen_factor=0.3)
        circle = Circle(0.2, stroke_width=10., color="#8F00FF", sheen_factor=0.5)
        circle.move_to(path.point_from_proportion(0))
        percent_animated = ValueTracker(0)
        circle.add_updater(lambda m : m.move_to(path.point_from_proportion(percent_animated.get_value())))
        vec_field = ArrowVectorField(field_func, colors=MELLOR_SCALAR_FIELD_COLORS, min_color_scheme_value=min_color_value, max_color_scheme_value=max_color_value)
        stream_lines = StreamLines(field_func, colors=MELLOR_SCALAR_FIELD_COLORS, min_color_scheme_value=min_color_value, max_color_scheme_value=max_color_value)
        self.add(stream_lines, vec_field)
        stream_lines.start_animation(warm_up=False, flow_speed=1.5)
        self.add(path, circle)
        #tan_vecs = [ Arrow(start = path.point_from_proportion(k / 15.0), end = path.point_from_proportion(k / 15.0) + tan_vector(path, k / 15.0)) for k in range(15) ]
        #self.add(*tan_vecs)
        distances = Non_Manim_Rate_Func(path, field_func, 2.0, 'temp_path.txt')
        rf = lambda t : distances.point_from_proportion(t)
        black_rectangle = Rectangle(width = 4.0, height = 10.0, fill_color="#000000", fill_opacity=1.0)
        black_rectangle.move_to([5.5, 0, 0])
        force_label = MathTex(r'\vec a_\text{free}')
        force_colon = MathTex(':')
        force_label.move_to([4.5, 3.5, 0])
        force_colon.next_to(force_label)
        force_vec = always_redraw(lambda : vec_field.get_vector(path.point_from_proportion(percent_animated.get_value())).scale(2.0).next_to(force_colon))
        force_label.add_updater(lambda m : m.set_color(force_vec.get_color()))
        s_hat_colon = MathTex(':')
        s_hat_colon.next_to(force_colon, DOWN)
        s_hat = MathTex(r'\hat s', color=path.get_color())
        s_hat.next_to(s_hat_colon, LEFT)
        s_hat_vec = always_redraw(lambda : Arrow(start = ORIGIN, end = tan_vector(path, percent_animated.get_value()), color = path.get_color()).next_to(s_hat_colon))

        cur_field_acc = lambda : vec_field.get_vector(
            path.point_from_proportion(
                percent_animated.get_value()
            )
        ).get_vector()

        cur_tan_vec = lambda : tan_vector(
            path,
            percent_animated.get_value()
        )

        net_vec_func = lambda : proj(
            cur_field_acc(),
            cur_tan_vec()
        )

        constraint_label = MathTex(r'\vec a_\text{con}')
        constraint_colon = MathTex(':')
        constraint_colon.next_to(s_hat_colon, DOWN)
        constraint_label.next_to(constraint_colon, LEFT)
        constraint_label.add_updater(lambda m : m.set_color(vec_color(net_vec_func() - cur_field_acc())))

        constraint_vec = always_redraw(
            lambda : Arrow(
                start = ORIGIN, end = net_vec_func() - cur_field_acc(),
                color = vec_color(net_vec_func() - cur_field_acc())
            ).scale(2.0).next_to(constraint_colon)
        )

        net_colon = MathTex(':')
        net_colon.next_to(constraint_colon, DOWN)
        net_label = MathTex(r'\vec a_\text{net}')
        net_label.add_updater(lambda m : m.set_color(vec_color(net_vec_func())))
        net_label.next_to(net_colon, LEFT)
        net_vec = always_redraw(lambda : Arrow(start = ORIGIN, end = net_vec_func(), color = vec_color(net_vec_func())).scale(2.0).next_to(net_colon))
        net_vec.next_to(net_colon)

        acc_label = MarkupText(f'Acceleration (m/s<sup>2</sup>)', font = "Noto Sans", color="#DC143C")
        acc_label.scale(0.5)
        acc_label.next_to(net_colon, DOWN)
        acc_label.shift(DOWN * 0.5)
        acc_line = Line(acc_label.get_left(), acc_label.get_right(), color="#DC143C")
        acc_line.next_to(acc_label, DOWN)
        acc_val = DecimalNumber()
        acc_val.next_to(acc_line, DOWN)
        acc_val.add_updater(lambda m : m.set_value(dot(net_vec_func(), s_hat_vec.get_vector())))
        acc_val.add_updater(lambda m : m.set_color(vec_color(np.array([acc_val.get_value(), 0, 0]))))

        vel_label = MarkupText(f'Velocity (m/s)', font = "Noto Sans", color="#80FFC8")
        vel_label.scale(0.5)
        vel_label.next_to(acc_val, DOWN)
        vel_line = Line(vel_label.get_left(), vel_label.get_right(), color="#80FFC8")
        vel_line.next_to(vel_label, DOWN)
        vel_val = DecimalNumber()
        vel_val.next_to(vel_line, DOWN)
        vel_val.add_updater(lambda m : m.set_value(distances.vel_from_proportion(percent_animated.get_value())))
        vel_val.add_updater(lambda m : m.set_color(vec_color(np.array([vel_val.get_value(), 0, 0]))))
        self.add(black_rectangle)
        self.add(
            force_label,
            force_colon,
            force_vec,
            s_hat,
            s_hat_colon,
            s_hat_vec,
            constraint_label,
            constraint_colon,
            constraint_vec,
            net_label,
            net_colon,
            net_vec,
            acc_label,
            acc_line,
            acc_val,
            vel_label,
            vel_line,
            vel_val
        )
        self.play(percent_animated.animate.set_value(1), rate_func=rf, run_time = 20.0)
