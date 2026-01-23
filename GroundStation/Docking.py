import enum
import math

class DockingManager:
    WP2_Z = 0.4 # 40cm
    WP2_ENTER_THRESHOLD = 0.02

    ALIGNMENT_CONE_THRESHOLD_TOP = 0.01
    ALIGNMENT_CONE_THRESHOLD_BOTTOM = 0.04
    class DockingState(enum.Enum):
        DOCKING_STATE_INITIAL = 0
        DOCKING_STATE_ALIGNMENT = 1
        DOCKING_STATE_INSERTION = 2
        DOCKING_STATE_DOCKED = 3
        DOCKING_STATE_ERROR = 4
        DOCKING_STATE_NUM = 5

    def __init__(self):
        self.pos_x = 0
        self.pos_y = 0
        self.pos_z = 0
        self.tracking_valid = False
        self.state = self.DockingState.DOCKING_STATE_INITIAL
        pass

    def get_state(self):
        return self.state

    def get_position_error(self):
        if self.state == self.DockingState.DOCKING_STATE_ALIGNMENT:
            return [self.pos_x, self.pos_y, self.pos_z + self.WP2_Z]
        elif self.state == self.DockingState.DOCKING_STATE_INSERTION:
            z_output = min(-0.02, self.pos_z + 0.12)
            return [self.pos_x, self.pos_y, z_output]
        else:
            return [0, 0, 0]

    def update_position_error(self, pos_x, pos_y, pos_z, tracking_valid) -> None:
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.tracking_valid = tracking_valid

        if not self.tracking_valid:
            self.state = self.DockingState.DOCKING_STATE_ERROR

        if self.state == self.DockingState.DOCKING_STATE_INITIAL:
            if self.tracking_valid:
                self.state = self.DockingState.DOCKING_STATE_ALIGNMENT
        elif self.state == self.DockingState.DOCKING_STATE_ALIGNMENT:
            pass
            # if self._enter_insertion_conditions_met():
            #     self.state = self.DockingState.DOCKING_STATE_INSERTION
            # elif not self.tracking_valid:
            #     self.state = self.DockingState.DOCKING_STATE_ERROR
        elif self.state == self.DockingState.DOCKING_STATE_INSERTION:
            if self._exit_insertion_conditions_met():
                self.state = self.DockingState.DOCKING_STATE_ALIGNMENT
            elif not self.tracking_valid:
                self.state = self.DockingState.DOCKING_STATE_ERROR
        elif self.state == self.DockingState.DOCKING_STATE_DOCKED:
            pass

    def _enter_insertion_conditions_met(self) -> bool:
        z_error = -self.pos_z - self.WP2_Z
        return all(
            [
                math.fabs(self.pos_x) < self.WP2_ENTER_THRESHOLD,
                math.fabs(self.pos_y) < self.WP2_ENTER_THRESHOLD,
                math.fabs(z_error) < self.WP2_ENTER_THRESHOLD,
            ]
        )

    def _exit_insertion_conditions_met(self) -> bool:
        if self.pos_z < -(self.WP2_Z + self.WP2_ENTER_THRESHOLD):
            return True
        
        max_xy_deviation = (self.ALIGNMENT_CONE_THRESHOLD_BOTTOM - self.ALIGNMENT_CONE_THRESHOLD_TOP) * (-self.pos_z / self.WP2_Z) + self.ALIGNMENT_CONE_THRESHOLD_TOP
        if math.fabs(self.pos_x) > max_xy_deviation or math.fabs(self.pos_y) > max_xy_deviation:
            return True
        
        return False
