"""
MAVLink connection manager and DEBUG_VECT publishing for tvec/rvec.
"""

import time
from typing import Optional, Union, Tuple
import os
import numpy as np

# DEBUG_VECT name field is 10 characters max
DEBUG_VECT_NAME_LEN = 10

# Ensure we use MAVLink2 in pymavlink (must be set before importing pymavlink).
os.environ["MAVLINK20"] = "1"

from pymavlink import mavutil

class MavlinkPublisher:
    """
    Manages connection to the flight controller and publishes DEBUG_VECT messages.
    Default connection: TCP 127.0.0.1:5760.
    """

    def __init__(
        self,
        address: str = "127.0.0.1",
        port: int = 5760,
        source_system: int = 1,
        source_component: int = 192,
        target_system: Optional[int] = None,
        target_component: Optional[int] = None,
        rc7_high_threshold_pwm: int = 1700,
    ):
        self._address = address
        self._port = port
        self._source_system = source_system
        self._source_component = source_component
        # Target is the flight controller we want to command (usually sysid=1, compid=1).
        # If not provided, we learn it from the first heartbeat on connect().
        self._target_system = target_system
        self._target_component = target_component
        self._connection: Optional[mavutil.mavlink_connection] = None

        # RC input tracking (updated by poll_incoming()).
        self._rc7_pwm: Optional[int] = None
        self._rc7_high: bool = False
        self._rc7_high_threshold_pwm: int = int(rc7_high_threshold_pwm)
        self._last_rc_update_ms: Optional[int] = None

    def connect(self) -> bool:
        """Open TCP connection to the flight controller. Returns True if successful."""
        try:
            self._connection = mavutil.mavlink_connection(
                f"tcp:{self._address}:{self._port}",
                source_system=self._source_system,
                source_component=self._source_component,
            )
            # Learn target IDs from heartbeat if not explicitly configured.
            if self._target_system is None or self._target_component is None:
                try:
                    self._connection.wait_heartbeat(timeout=2)
                    if getattr(self._connection, "target_system", 0):
                        self._target_system = int(self._connection.target_system)
                        self._target_component = int(self._connection.target_component)
                except Exception:
                    # Heartbeat might not be available immediately; fall back to defaults.
                    pass
            if self._target_system is None:
                self._target_system = 1
            if self._target_component is None:
                self._target_component = 1
            print(
                f"MavlinkPublisher: connected src={self._source_system}.{self._source_component} "
                f"-> tgt={self._target_system}.{self._target_component}"
            )
            return True
        except Exception as e:
            print(f"MavlinkPublisher: connection failed: {e}")
            self._connection = None
            return False

    def close(self) -> None:
        """Close the connection if open."""
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None
        self._rc7_pwm = None
        self._rc7_high = False
        self._last_rc_update_ms = None

    @property
    def is_connected(self) -> bool:
        return self._connection is not None

    def _prepare_name(self, base: str, suffix: str) -> str:
        """Return base + suffix truncated to DEBUG_VECT_NAME_LEN."""
        name = base + suffix
        if len(name) > DEBUG_VECT_NAME_LEN:
            name = name[:DEBUG_VECT_NAME_LEN]
        return str(name)

    def _to_xyz(self, v: Union[np.ndarray, tuple, list]) -> tuple:
        """Ensure vector is (x, y, z) floats."""
        a = np.asarray(v, dtype=np.float64).reshape(3)
        return (float(a[0]), float(a[1]), float(a[2]))

    def _get_target_ids(self) -> Tuple[int, int]:
        """
        Return (target_system, target_component) for addressed MAVLink messages.
        Prefer IDs learned from heartbeat; fall back to configured/default 1/1.
        """
        if self._connection is not None:
            ts = int(getattr(self._connection, "target_system", 0) or 0)
            tc = int(getattr(self._connection, "target_component", 0) or 0)
            if ts:
                return ts, (tc or 1)
        return int(self._target_system or 1), int(self._target_component or 1)

    def set_rc7_high_threshold(self, pwm: int) -> None:
        """Set the PWM threshold above which RC channel 7 is considered 'high'."""
        self._rc7_high_threshold_pwm = int(pwm)

    def _handle_rc_message(self, msg) -> None:
        """
        Update cached RC7 state from an incoming RC message.
        Supports RC_CHANNELS and RC_CHANNELS_RAW.
        """
        try:
            mtype = msg.get_type()
        except Exception:
            return

        pwm: Optional[int] = None
        if mtype == "RC_CHANNELS":
            # MAVLink RC_CHANNELS has chan1_raw..chan18_raw
            pwm = int(getattr(msg, "chan7_raw", 0) or 0)
        elif mtype == "RC_CHANNELS_RAW":
            # MAVLink RC_CHANNELS_RAW has chan1_raw..chan8_raw
            pwm = int(getattr(msg, "chan7_raw", 0) or 0)
        else:
            return

        if pwm <= 0:
            # Don't update state on "no data" values.
            return

        self._rc7_pwm = pwm
        self._rc7_high = pwm >= self._rc7_high_threshold_pwm

        # Prefer a message timestamp if present; else wall clock.
        t_ms = None
        if hasattr(msg, "time_boot_ms"):
            t_ms = int(getattr(msg, "time_boot_ms") or 0) or None
        if t_ms is None:
            t_ms = int(time.time() * 1000)
        self._last_rc_update_ms = t_ms

    def poll_incoming(self, max_messages: int = 50) -> int:
        """
        Non-blocking poll of incoming MAVLink messages.
        Updates internal RC7 state when RC_CHANNELS/RC_CHANNELS_RAW are received.

        Returns number of messages processed.
        """
        if self._connection is None:
            return 0
        processed = 0
        try:
            while processed < max_messages:
                msg = self._connection.recv_match(
                    type=["RC_CHANNELS", "RC_CHANNELS_RAW"],
                    blocking=False,
                )
                if msg is None:
                    break
                processed += 1
                self._handle_rc_message(msg)
        except Exception:
            # Swallow polling exceptions; caller can keep running.
            return processed
        return processed

    def is_rc7_high(self) -> bool:
        """
        Return True if RC channel 7 is currently 'high' (>= threshold).
        This performs a non-blocking poll first so the value stays fresh.
        """
        self.poll_incoming()
        return bool(self._rc7_high)

    def get_rc7_pwm(self) -> Optional[int]:
        """Return last seen PWM value for RC channel 7 (or None if never seen)."""
        self.poll_incoming()
        return self._rc7_pwm

    def get_rc7_last_update_ms(self) -> Optional[int]:
        """Return last update time (ms) for RC input cache, or None if never updated."""
        return self._last_rc_update_ms

    # -------------------------------------------------------------------------
    # Message interval (request stream rate from FC)
    # -------------------------------------------------------------------------

    def request_local_position_ned_at_50hz(self) -> bool:
        """
        Send MAV_CMD_SET_MESSAGE_INTERVAL to request LOCAL_POSITION_NED at 50 Hz from the FC.
        Returns True if the command was sent, False if not connected or send failed.
        """
        return self._set_message_interval_hz(
            mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED,
            20.0,
        )

    def _set_message_interval_hz(self, message_id: int, rate_hz: float) -> bool:
        """
        Send MAV_CMD_SET_MESSAGE_INTERVAL for the given message ID at rate_hz.
        rate_hz: 0 = default, -1 = disable. Returns True if sent.
        """
        if self._connection is None:
            return False
        try:
            target_system, target_component = self._get_target_ids()
            print(f"Setting message interval for {message_id} to {rate_hz}  target={target_system}.{target_component}")
            if rate_hz == 0 or rate_hz == -1:
                interval_us = int(rate_hz)
            else:
                interval_us = int(1_000_000.0 / float(rate_hz))
            self._connection.mav.command_long_send(
                target_system,
                target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,  # confirmation
                float(message_id),  # param1: message ID
                float(interval_us),  # param2: interval_us
                0, 0, 0, 0, 0,
            )
            return True
        except Exception as e:
            print(f"MavlinkPublisher: _set_message_interval_hz failed: {e}")
            return False

    # -------------------------------------------------------------------------
    # RC override output
    # -------------------------------------------------------------------------

    def send_rc_override(
        self,
        chan3: Optional[int] = None,
        chan4: Optional[int] = None,
    ) -> bool:
        """
        Send an RC_CHANNELS_OVERRIDE for channels 3 and 4.

        - chan3_pwm: PWM value for channel 3 (throttle on many radios), or None to leave unchanged.
        - chan4_pwm: PWM value for channel 4 (yaw on many radios), or None to leave unchanged.

        All other channels are set to 65535 (no override).
        """

        chan3_pwm = None if chan3 is None else int(1500 + max(-1.0, min(1.0, chan3)) * 500)
        chan4_pwm = None if chan4 is None else int(1500 + max(-1.0, min(1.0, chan4)) * 500)
        
        if self._connection is None:
            return False
        try:
            target_system, target_component = self._get_target_ids()

            def _val(pwm: Optional[int]) -> int:
                return int(pwm) if pwm is not None else 0

            self._connection.mav.rc_channels_override_send(
                1,
                1,
                65535,                  # chan1_raw
                65535,                  # chan2_raw
                _val(chan3_pwm),        # chan3_raw
                _val(chan4_pwm),        # chan4_raw
                65535,                  # chan5_raw
                65535,                  # chan6_raw
                65535,                  # chan7_raw
                65535,                  # chan8_raw
            )

            print(f"{chan3_pwm}, {chan4_pwm}")
            return True
        except Exception as e:
            # print(f"MavlinkPublisher: send_rc_override failed: {e}\")
            return False

    def publish_tvec_rvec(
        self,
        name: str,
        tvec: Union[np.ndarray, tuple, list],
        rvec: Union[np.ndarray, tuple, list],
    ) -> bool:
        """
        Publish two DEBUG_VECT messages: one for rvec (name + "_R") and one for tvec (name + "_T").
        name is truncated so that name + "_R" / name + "_T" fit in 10 characters.

        Returns True if both messages were sent, False if not connected or send failed.
        """
        if self._connection is None:
            return False
        try:
            usec = int(time.time() * 1e6)
            name_r = self._prepare_name(name, "_R")
            name_t = self._prepare_name(name, "_T")
            # pymavlink debug_vect expects name as bytes (splits on b'\\x00')
            name_r_bytes = name_r.encode("ascii", errors="replace")
            name_t_bytes = name_t.encode("ascii", errors="replace")
            xr, yr, zr = self._to_xyz(rvec)
            xt, yt, zt = self._to_xyz(tvec)
            self._connection.mav.debug_vect_send(name_r_bytes, usec, xr, yr, zr)
            self._connection.mav.debug_vect_send(name_t_bytes, usec, xt, yt, zt)
            return True
        except Exception as e:
            print(f"MavlinkPublisher: publish_tvec_rvec failed: {e}")
            return False

    def publish_target_point(self, target_point: Union[np.ndarray, tuple, list]) -> bool:
        """
        Publish a DEBUG_VECT message for a target point.

        Returns True if the message was sent, False if not connected or send failed.
        """
        if self._connection is None:
            return False
        try:
            usec = int(time.time() * 1e6)
            name_t = self._prepare_name("TARGET", "_T")
            name_t_bytes = name_t.encode("ascii", errors="replace")
            xt, yt, zt = self._to_xyz(target_point)
            self._connection.mav.debug_vect_send(name_t_bytes, usec, xt, yt, zt)
            return True
        except Exception as e:
            print(f"MavlinkPublisher: publish_target_point failed: {e}")
            return False

    def publish_position_target(
        self,
        tvec: Union[np.ndarray, tuple, list],
        yaw,
        timestamp: int
    ) -> bool:

        if self._connection is None:
                return False
        try:
            usec = int(time.time() * 1e6)
            x, y, z = self._to_xyz(tvec)
            distance = float(np.sqrt(x * x + y * y + z * z))
            target_system, target_component = self._get_target_ids()
            # Use position + yaw, ignore velocity/accel/yaw_rate.
            type_mask = (
                # mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
                # mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
                # mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE |
                0xF000  # "last byte" set; accepted by ArduPilot either way
            )
            self._connection.mav.set_position_target_local_ned_send(
                time_boot_ms=timestamp,
                target_system=target_system,
                target_component=target_component,
                coordinate_frame=mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
                type_mask=type_mask,
                x=z,
                y=x,
                z=y,
                vx = 0,
                vy = 0,
                vz = 0,
                afx = 0,
                afy = 0,
                afz = 0,
                yaw = yaw,
                yaw_rate = 0,
            )
            print(
                f"Published position target to {target_system}.{target_component}: "
                f"forward={z}, right={x}, down={y}, yaw={yaw}, {type(yaw)}, timestamp={timestamp}"
            )
            # print(yaw)
            
            return True
        except Exception as e:
            print(f"MavlinkPublisher: publish_position_target failed: {e}")
            return False


    def publish_landing_target(
        self,
        tvec: Union[np.ndarray, tuple, list],
        use_angles: bool = False,
    ) -> bool:
        """
        Publish a LANDING_TARGET message with the target position in body FRD frame.

        tvec: (x, y, z) target position — forward, right, down in body frame (meters).

        use_angles: If True, set position_valid=0 and send angle_x, angle_y, distance
            derived from tvec (angle/distance form). If False, set position_valid=1 and
            send position as x, y, z (body FRD).

        Returns True if the message was sent, False if not connected or send failed.
        """
        if self._connection is None:
            return False
        try:
            usec = int(time.time() * 1e6)
            x, y, z = self._to_xyz(tvec)
            distance = float(np.sqrt(x * x + y * y + z * z))
            if use_angles:
                if distance < 1e-6:
                    angle_x = 0.0
                    angle_y = 0.0
                else:
                    angle_x = float(np.arctan2(x, y))
                    angle_y = float(np.arctan2(z, y))
                self._connection.mav.landing_target_send(
                    time_usec=usec,
                    target_num=0,
                    frame=mavutil.mavlink.MAV_FRAME_BODY_FRD,
                    angle_x=angle_x,
                    angle_y=angle_y,
                    distance=distance,
                    size_x=0.0,
                    size_y=0.0,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    q=[1.0, 0.0, 0.0, 0.0],
                    type=mavutil.mavlink.LANDING_TARGET_TYPE_VISION_FIDUCIAL,
                    position_valid=0,
                )
            else:
                self._connection.mav.landing_target_send(
                    time_usec=usec,
                    target_num=0,
                    frame=mavutil.mavlink.MAV_FRAME_BODY_FRD,
                    angle_x=0.0,
                    angle_y=0.0,
                    distance=distance,
                    size_x=0.0,
                    size_y=0.0,
                    x=z,
                    y=x,
                    z=y,
                    q=[1.0, 0.0, 0.0, 0.0],
                    type=mavutil.mavlink.LANDING_TARGET_TYPE_VISION_FIDUCIAL,
                    position_valid=1,
                )
            return True
        except Exception as e:
            print(f"MavlinkPublisher: publish_landing_target failed: {e}")
            return False

    def __enter__(self) -> "MavlinkPublisher":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
