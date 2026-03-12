import enum
import math
import time
import numpy as np


DOCKED_POSITION = np.array([-0.04, -0.383, -0.458])
ALIGNMENT_ALT_OFFSET = 0.3
ALIGNMENT_POSITION = np.array([DOCKED_POSITION[0], DOCKED_POSITION[1] + ALIGNMENT_ALT_OFFSET, DOCKED_POSITION[2]])
ENTER_INSERTION_SPHERE_RADIUS = 0.07
INSERTION_HOLD_TIME_SEC = 3.0

ALIGNMENT_CONE_THRESHOLD_TOP = 0.05
ALIGNMENT_CONE_THRESHOLD_BOTTOM = 0.12

ALIGNMENT_LEAD_DISTANCE = 0.3

class DockingManager:
    class DockingState(enum.Enum):
        DOCKING_STATE_INITIAL = 0
        DOCKING_STATE_ALIGNMENT = 1
        DOCKING_STATE_INSERTION = 2
        DOCKING_STATE_DOCKED = 3
        DOCKING_STATE_ERROR = 4
        DOCKING_STATE_NUM = 5


    def __init__(self):
        self.state = self.DockingState.DOCKING_STATE_INITIAL
        self._enter_sphere_start_time_s = None

    def should_insert(self):
        if self._enter_sphere_start_time_s is None:
            return False
        return (time.monotonic() - self._enter_sphere_start_time_s) >= INSERTION_HOLD_TIME_SEC

    def get_target_positition(self, Tbody_in_tag: np.ndarray) -> np.ndarray:
        self._update_state(Tbody_in_tag)
        if self.state == self.DockingState.DOCKING_STATE_INSERTION:
            return DOCKED_POSITION
        elif self.state == self.DockingState.DOCKING_STATE_ALIGNMENT or self.state == self.DockingState.DOCKING_STATE_INITIAL:
            # return ALIGNMENT_POSITION
            return ALIGNMENT_POSITION
            direction = ALIGNMENT_POSITION - Tbody_in_tag
            distance = np.linalg.norm(direction)
            if distance < 0.5:
                return ALIGNMENT_POSITION
            else:
                direction_normalized = direction / distance if distance != 0 else np.zeros_like(direction)
                step = direction_normalized * 0.2
                return Tbody_in_tag + step

            error = Tbody_in_tag - ALIGNMENT_POSITION
            error_scaled = error * np.linalg.norm(error) * 0.3
            return Tbody_in_tag - error_scaled
        elif self.state == self.DockingState.DOCKING_STATE_DOCKED:
            return DOCKED_POSITION
        elif self.state == self.DockingState.DOCKING_STATE_ERROR:
            return None
        else:
            return 
            
    def _update_state(self, Tbody_in_tag: np.ndarray) -> None:
        self._update_enter_insertion_timer(Tbody_in_tag)
        if self.state == self.DockingState.DOCKING_STATE_INITIAL:
            self.state = self.DockingState.DOCKING_STATE_ALIGNMENT
        elif self.state == self.DockingState.DOCKING_STATE_ALIGNMENT:
            if self._enter_insertion_conditions_met(Tbody_in_tag):
                pass
                # self.state = self.DockingState.DOCKING_STATE_INSERTION
        elif self.state == self.DockingState.DOCKING_STATE_INSERTION:
            if self._exit_insertion_conditions_met(Tbody_in_tag):
                self.state = self.DockingState.DOCKING_STATE_ALIGNMENT
        elif self.state == self.DockingState.DOCKING_STATE_DOCKED:
            pass
        elif self.state == self.DockingState.DOCKING_STATE_ERROR:
            pass
        else:
            return

    def _update_enter_insertion_timer(self, Tbody_in_tag: np.ndarray) -> None:
        in_sphere = np.linalg.norm(Tbody_in_tag - ALIGNMENT_POSITION) < ENTER_INSERTION_SPHERE_RADIUS
        if in_sphere:
            if self._enter_sphere_start_time_s is None:
                self._enter_sphere_start_time_s = time.monotonic()
        else:
            self._enter_sphere_start_time_s = None

    def _enter_insertion_conditions_met(self, Tbody_in_tag: np.ndarray) -> bool:
        return np.linalg.norm(Tbody_in_tag - ALIGNMENT_POSITION) < ENTER_INSERTION_SPHERE_RADIUS

    def _exit_insertion_conditions_met(self, Tbody_in_tag: np.ndarray) -> bool:
        vertical_error = DOCKED_POSITION[2] - Tbody_in_tag[2]

        if vertical_error < -(ALIGNMENT_ALT_OFFSET + ENTER_INSERTION_SPHERE_RADIUS):
            return True
        
        max_xy_deviation = (ALIGNMENT_CONE_THRESHOLD_BOTTOM - ALIGNMENT_CONE_THRESHOLD_TOP) * (-vertical_error / ALIGNMENT_ALT_OFFSET) + ALIGNMENT_CONE_THRESHOLD_TOP
        
        print(f"vertical_error: {vertical_error}, max_xy_deviation: {max_xy_deviation}")
        if math.fabs(Tbody_in_tag[0] - ALIGNMENT_POSITION[0]) > max_xy_deviation or math.fabs(Tbody_in_tag[1] - ALIGNMENT_POSITION[1]) > max_xy_deviation:
            return True
        
        return False




# class DockingManager:
#     WP2_Z = 0.4 # 40cm
#     WP2_ENTER_THRESHOLD = 0.02

#     ALIGNMENT_CONE_THRESHOLD_TOP = 0.01
#     ALIGNMENT_CONE_THRESHOLD_BOTTOM = 0.04
#     
#     def __init__(self):
#         self.pos_x = 0
#         self.pos_y = 0
#         self.pos_z = 0
#         self.tracking_valid = False
#         self.state = self.DockingState.DOCKING_STATE_INITIAL
#         pass

#     def get_state(self):
#         return self.state

#     def get_position_error(self):
#         if self.state == self.DockingState.DOCKING_STATE_ALIGNMENT:
#             return [self.pos_x, self.pos_y, self.pos_z + self.WP2_Z]
#         elif self.state == self.DockingState.DOCKING_STATE_INSERTION:
#             z_output = min(-0.02, self.pos_z + 0.12)
#             return [self.pos_x, self.pos_y, z_output]
#         else:
#             return [0, 0, 0]

#     def update_position_error(self, pos_x, pos_y, pos_z, tracking_valid) -> None:
#         self.pos_x = pos_x
#         self.pos_y = pos_y
#         self.pos_z = pos_z
#         self.tracking_valid = tracking_valid

#         if not self.tracking_valid:
#             self.state = self.DockingState.DOCKING_STATE_ERROR

#         if self.state == self.DockingState.DOCKING_STATE_INITIAL:
#             if self.tracking_valid:
#                 self.state = self.DockingState.DOCKING_STATE_ALIGNMENT
#         elif self.state == self.DockingState.DOCKING_STATE_ALIGNMENT:
#             pass
#             # if self._enter_insertion_conditions_met():
#             #     self.state = self.DockingState.DOCKING_STATE_INSERTION
#             # elif not self.tracking_valid:
#             #     self.state = self.DockingState.DOCKING_STATE_ERROR
#         elif self.state == self.DockingState.DOCKING_STATE_INSERTION:
#             if self._exit_insertion_conditions_met():
#                 self.state = self.DockingState.DOCKING_STATE_ALIGNMENT
#             elif not self.tracking_valid:
#                 self.state = self.DockingState.DOCKING_STATE_ERROR
#         elif self.state == self.DockingState.DOCKING_STATE_DOCKED:
#             pass

#     def _enter_insertion_conditions_met(self) -> bool:
#         z_error = -self.pos_z - self.WP2_Z
#         return all(
#             [
#                 math.fabs(self.pos_x) < self.WP2_ENTER_THRESHOLD,
#                 math.fabs(self.pos_y) < self.WP2_ENTER_THRESHOLD,
#                 math.fabs(z_error) < self.WP2_ENTER_THRESHOLD,
#             ]
#         )

#     def _exit_insertion_conditions_met(self) -> bool:
#         if self.pos_z < -(self.WP2_Z + self.WP2_ENTER_THRESHOLD):
#             return True
        
#         max_xy_deviation = (self.ALIGNMENT_CONE_THRESHOLD_BOTTOM - self.ALIGNMENT_CONE_THRESHOLD_TOP) * (-self.pos_z / self.WP2_Z) + self.ALIGNMENT_CONE_THRESHOLD_TOP
#         if math.fabs(self.pos_x) > max_xy_deviation or math.fabs(self.pos_y) > max_xy_deviation:
#             return True
        
#         return False
