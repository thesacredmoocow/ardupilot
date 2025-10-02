from time import time_ns


class Pose:
    def __init__(self):
        self.vel_x = 0
        self.vel_y = 0
        self.vel_z = 0
        self.pos_x = 0
        self.pos_y = 0
        self.pos_z = 0
        self.roll = 0
        self.pitch = 0
        self.yaw = 0

        # filtered position state (initialized on first update)
        self.fpos_x = 0
        self.fpos_y = 0
        self.fpos_z = 0

        # LPF coefficient for position filtering in (0, 1]; higher favors new data
        self.pos_lpf_alpha = 0.2

        self.last_update_time_us = 0

        self.pos_valid = False
        self.vel_valid = False

    def set_position_lpf_alpha(self, alpha):
        # clamp alpha to sensible bounds
        if alpha <= 0:
            alpha = 1e-6
        if alpha > 1:
            alpha = 1
        self.pos_lpf_alpha = alpha

    def update_pos(self, pos_x, pos_y, pos_z, timestamp_us=None):
        # mark position as valid and store raw
        self.pos_valid = True
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z

        now_us = timestamp_us if timestamp_us is not None else time_ns() / 1000

        # first update initializes filtered position and time base
        if self.last_update_time_us == 0:
            self.fpos_x = pos_x
            self.fpos_y = pos_y
            self.fpos_z = pos_z
            self.vel_x = 0
            self.vel_y = 0
            self.vel_z = 0
            self.vel_valid = False
            self.last_update_time_us = now_us
            return

        dt = (now_us - self.last_update_time_us) * 1e-6
        if dt <= 0:
            # time did not advance; skip velocity update
            return

        alpha = self.pos_lpf_alpha

        prev_fpos_x = self.fpos_x
        prev_fpos_y = self.fpos_y
        prev_fpos_z = self.fpos_z

        # exponential moving average for position
        self.fpos_x = alpha * pos_x + (1 - alpha) * prev_fpos_x
        self.fpos_y = alpha * pos_y + (1 - alpha) * prev_fpos_y
        self.fpos_z = alpha * pos_z + (1 - alpha) * prev_fpos_z

        # velocity from filtered position derivative
        self.vel_x = (self.fpos_x - prev_fpos_x) / dt
        self.vel_y = (self.fpos_y - prev_fpos_y) / dt
        self.vel_z = (self.fpos_z - prev_fpos_z) / dt
        self.vel_valid = True

        self.last_update_time_us = now_us
