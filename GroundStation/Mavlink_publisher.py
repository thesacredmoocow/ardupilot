"""
MAVLink connection manager and DEBUG_VECT publishing for tvec/rvec.
"""

import os
import threading
import time
import Utils
from typing import List, Optional, Union, Tuple

# Path for saving IMU calibration data when save_calib_data is True
CALIB_IMU_CSV_PATH = "/home/raspi/ardupilot/GroundStation/SLAM/calibration_data/imu_data.csv"

import numpy as np

# DEBUG_VECT name field is 10 characters max
DEBUG_VECT_NAME_LEN = 10

# Ensure we use MAVLink2 in pymavlink (must be set before importing pymavlink).
os.environ["MAVLINK20"] = "1"

from pymavlink import mavutil

OFFSET_IMU_TIME = 0.0164402 - 0.0256

FLOW_SCALE_X = 0.4
FLOW_SCALE_Y = 0.45

class MavlinkPublisher:
    """
    Manages connection to the flight controller and publishes DEBUG_VECT messages.
    Default connection: TCP 127.0.0.1:5760.
    """

    def __init__(
        self,
        format: str = "tcp",
        address: str = "127.0.0.1",
        port: int = 5760,
        source_system: int = 1,
        source_component: int = 192,
        target_system: Optional[int] = None,
        target_component: Optional[int] = None,
        save_calib_data: bool = False,
    ):
        self._format = format
        self._address = address
        self._port = port
        self._source_system = source_system
        self._source_component = source_component
        self._save_calib_data = save_calib_data
        # Target is the flight controller we want to command (usually sysid=1, compid=1).
        # If not provided, we learn it from the first heartbeat on connect().
        self._target_system = target_system
        self._target_component = target_component
        self._connection: Optional[mavutil.mavlink_connection] = None

        # Last RAW_IMU message (updated by poll_incoming() or poll thread via _handle_raw_imu).
        self._raw_imu: Optional[dict] = None

        # SYSTEM_TIME from target: last time_unix_usec, time_boot_ms, and offset for boot_ms -> Unix time.
        # unix_time_sec = time_boot_ms/1000 + _boot_to_unix_offset_sec
        self._system_time_unix_usec: Optional[int] = None
        self._system_time_boot_ms: Optional[int] = None
        self._boot_to_unix_offset_sec: Optional[float] = None

        # Background message polling thread.
        self._poll_stop = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None
        self._poll_lock = threading.Lock()
        self._poll_start_time: Optional[float] = None  # time.time() when start_polling_thread was called
        # IMU packets accumulated since last get_raw_imu_packets_since_last_call(); each dict has imu fields + "time_delta_sec"
        self._raw_imu_packets: List[dict] = []
        # Rolling history of the last 200 IMU packets (same dict format as above)
        self._raw_imu_history: List[dict] = []

        # Heartbeat/timesync thread: runs at 1 Hz while connected.
        self._heartbeat_timesync_stop = threading.Event()
        self._heartbeat_timesync_thread: Optional[threading.Thread] = None
    def _raw_imu_dict_from_msg(self, msg) -> dict:
        """Build raw IMU dict from a RAW_IMU message."""
        # return {
        #     "time_usec": getattr(msg, "time_usec", 0) or 0,
        #     "xacc": -getattr(msg, "yacc", 0) or 0,
        #     "yacc": -getattr(msg, "zacc", 0) or 0,
        #     "zacc": -getattr(msg, "xacc", 0) or 0,
        #     "xgyro": -getattr(msg, "ygyro", 0) or 0,
        #     "ygyro": -getattr(msg, "zgyro", 0) or 0,
        #     "zgyro": -getattr(msg, "xgyro", 0) or 0,
        #     "xmag": getattr(msg, "xmag", 0) or 0,
        #     "ymag": getattr(msg, "ymag", 0) or 0,
        #     "zmag": getattr(msg, "zmag", 0) or 0,
        #     "id": getattr(msg, "id", 0) or 0,
        # }
        return {
            "time_usec": msg.time_usec,
            "xacc": msg.xacc,
            "yacc": msg.yacc,
            "zacc": msg.zacc,
            "xgyro": msg.xgyro,
            "ygyro": msg.ygyro,
            "zgyro": msg.zgyro,
            "id": msg.id,
        }

    def _append_imu_to_calib_csv(self, packet: dict) -> None:
        """Append one IMU packet to the calibration file when save_calib_data is True.
        Format: EuRoC-style header, then rows with timestamp [ns], omega [rad/s], acc [m/s^2].
        RAW_IMU: time_usec (us), xacc/yacc/zacc in 0.001 m/s^2, xgyro/ygyro/zgyro in 0.001 rad/s.
        """
        if not self._save_calib_data:
            return
        try:
            dirname = os.path.dirname(CALIB_IMU_CSV_PATH)
            os.makedirs(dirname, exist_ok=True)
            write_header = not os.path.isfile(CALIB_IMU_CSV_PATH) or os.path.getsize(CALIB_IMU_CSV_PATH) == 0
            # timestamp [ns]; gyro x,y,z [rad/s]; acc x,y,z [m/s^2]
            # timestamp_ns = int(packet["time_usec"]) * 1000
            timestamp_ns = int((time.time() - self._poll_start_time) * 1e9)
            w_x = float(packet.get("xgyro") or 0) / 1000.0
            w_y = float(packet.get("ygyro") or 0) / 1000.0
            w_z = float(packet.get("zgyro") or 0) / 1000.0
            # ArduPilot RAW_IMU acc is in 0.001*g; convert to m/s^2
            a_x = float(packet.get("xacc") or 0) / 1000.0 * 9.81
            a_y = float(packet.get("yacc") or 0) / 1000.0 * 9.81
            a_z = float(packet.get("zacc") or 0) / 1000.0 * 9.81
            with open(CALIB_IMU_CSV_PATH, "a", newline="") as imu_file:
                if write_header:
                    imu_file.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")
                imu_file.write(f"{timestamp_ns},{w_x},{w_y},{w_z},{a_x},{a_y},{a_z}\n")
        except Exception as e:
            print(f"MavlinkPublisher: save calib IMU failed: {e}")

    def _poll_loop(self) -> None:
        """Background loop: poll for RAW_IMU and SYSTEM_TIME, accumulate IMU packets with time_delta_sec."""
        while not self._poll_stop.wait(timeout=0.001):
            conn = self._connection
            if conn is None:
                continue
            try:
                msg = conn.recv_match(
                    type=["RAW_IMU", "SYSTEM_TIME"],
                    blocking=False,
                )
            except Exception:
                continue
            if msg is None:
                continue
            try:
                mtype = msg.get_type() if hasattr(msg, "get_type") else None
            except Exception:
                continue
            if mtype == "RAW_IMU":
                start_time = self._poll_start_time
                time_delta_sec = (time.time() - start_time) if start_time is not None else 0.0
                packet = {**self._raw_imu_dict_from_msg(msg), "time_delta_sec": time_delta_sec}
                with self._poll_lock:
                    self._raw_imu_packets.append(packet)
                    self._raw_imu = packet
                    self._raw_imu_history.append(packet)
                    if len(self._raw_imu_history) > 200:
                        self._raw_imu_history = self._raw_imu_history[-200:]
                self._append_imu_to_calib_csv(packet)
            elif mtype == "SYSTEM_TIME":
                try:
                    if hasattr(msg, "get_srcSystem") and hasattr(msg, "get_srcComponent"):
                        target_sys, target_comp = self._get_target_ids()
                        if msg.get_srcSystem() != target_sys or msg.get_srcComponent() != target_comp:
                            continue
                    time_unix_usec = int(getattr(msg, "time_unix_usec", 0) or 0)
                    time_boot_ms = int(getattr(msg, "time_boot_ms", 0) or 0)
                    boot_to_unix_offset_sec = (time_unix_usec / 1e6) - (time_boot_ms / 1000.0)
                    with self._poll_lock:
                        self._system_time_unix_usec = time_unix_usec
                        self._system_time_boot_ms = time_boot_ms
                        self._boot_to_unix_offset_sec = boot_to_unix_offset_sec
                except Exception:
                    pass

    def start_polling_thread(self, start_time: float) -> None:
        """
        Start the background thread that polls for RAW_IMU and SYSTEM_TIME.
        start_time: reference time (seconds since epoch, e.g. time.time()) used to compute
            time_delta_sec for each IMU packet (time received - start_time).
        Idempotent: if the thread is already running, it is stopped and restarted with the new start_time.
        """
        self.stop_polling_thread()
        self._poll_start_time = float(start_time) + OFFSET_IMU_TIME
        self._poll_stop.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="MavlinkPoll",
            daemon=True,
        )
        self._poll_thread.start()

    def stop_polling_thread(self) -> None:
        """Stop the background polling thread if it is running."""
        self._poll_stop.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2.0)
            self._poll_thread = None

    def _heartbeat_timesync_loop(self) -> None:
        """Send HEARTBEAT and TIMESYNC at 1 Hz until stop is set."""
        interval = 1.0
        start_time = time.time()
        while not self._heartbeat_timesync_stop.wait(timeout=interval):
            conn = self._connection
            if conn is None:
                continue
            try:
                conn.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0,
                    mavutil.mavlink.MAV_STATE_ACTIVE,
                    0,
                )
                # TIMESYNC: tc1=0 means we are sending our time (ts1) for sync.
                # Keep ts1 in 32-bit range to avoid 'I' format error in dialects that pack as uint32.
                # ts1_ns = int(time.time() * 1e9) & 0xFFFFFFFF
                # conn.mav.timesync_send(0, ts1_ns)
                # SYSTEM_TIME: time since loop start (boot) and Unix time in microseconds.
                time_boot_ms = int((time.time() - start_time) * 1000)
                time_unix_usec = int(time.time() * 1e6)
                conn.mav.system_time_send(time_unix_usec, time_boot_ms)
            except Exception as e:
                print(f"MavlinkPublisher: _heartbeat_timesync_loop failed: {e}")

    def connect(self) -> bool:
        """Open TCP connection to the flight controller. Returns True if successful."""
        try:
            self._connection = mavutil.mavlink_connection(
                f"{self._format}:{self._address}:{self._port}",
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
            self._heartbeat_timesync_stop.clear()
            self._heartbeat_timesync_thread = threading.Thread(
                target=self._heartbeat_timesync_loop,
                name="MavlinkHeartbeatTimesync",
                daemon=True,
            )
            self._heartbeat_timesync_thread.start()
            return True
        except Exception as e:
            print(f"MavlinkPublisher: connection failed: {e}")
            self._connection = None
            return False

    def close(self) -> None:
        """Close the connection if open and stop the heartbeat/timesync and polling threads."""
        self.stop_polling_thread()
        self._heartbeat_timesync_stop.set()
        if self._heartbeat_timesync_thread is not None:
            self._heartbeat_timesync_thread.join(timeout=2.5)
            self._heartbeat_timesync_thread = None
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None
        self._raw_imu = None
        self._raw_imu_packets = []
        self._raw_imu_history = []
        self._system_time_unix_usec = None
        self._system_time_boot_ms = None
        self._boot_to_unix_offset_sec = None

    @property
    def is_connected(self) -> bool:
        return self._connection is not None

    @property
    def save_calib_data(self) -> bool:
        """When True, IMU data is appended to CALIB_IMU_CSV_PATH while the polling thread runs."""
        return self._save_calib_data

    @save_calib_data.setter
    def save_calib_data(self, value: bool) -> None:
        self._save_calib_data = bool(value)

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

    def _handle_raw_imu(self, msg) -> None:
        """
        Store the latest RAW_IMU message for later retrieval.
        Fields: time_usec, xacc, yacc, zacc, xgyro, ygyro, zgyro, xmag, ymag, zmag, id.
        """
        try:
            packet = self._raw_imu_dict_from_msg(msg)
            # When not using the high-rate polling thread, we don't have a precise reference start time;
            # still provide a time_delta_sec field for API consistency.
            packet["time_delta_sec"] = 0.0
            self._raw_imu = packet
            with self._poll_lock:
                self._raw_imu_history.append(packet)
                if len(self._raw_imu_history) > 200:
                    self._raw_imu_history = self._raw_imu_history[-200:]
        except Exception:
            pass

    def _handle_system_time(self, msg) -> None:
        """
        Update SYSTEM_TIME from target component: store time_unix_usec, time_boot_ms,
        and the offset so that unix_time_sec = time_boot_ms/1000 + _boot_to_unix_offset_sec.
        """
        try:
            if hasattr(msg, "get_srcSystem") and hasattr(msg, "get_srcComponent"):
                target_sys, target_comp = self._get_target_ids()
                if msg.get_srcSystem() != target_sys or msg.get_srcComponent() != target_comp:
                    return
            time_unix_usec = int(getattr(msg, "time_unix_usec", 0) or 0)
            time_boot_ms = int(getattr(msg, "time_boot_ms", 0) or 0)
            self._system_time_unix_usec = time_unix_usec
            self._system_time_boot_ms = time_boot_ms
            # offset such that unix_sec = time_boot_ms/1000 + offset  =>  offset = unix_sec - time_boot_ms/1000
            self._boot_to_unix_offset_sec = (time_unix_usec / 1e6) - (time_boot_ms / 1000.0)
        except Exception:
            pass

    def poll_incoming(self, max_messages: int = 50) -> int:
        """
        Non-blocking poll of incoming MAVLink messages.
        Updates last RAW_IMU and SYSTEM_TIME (from target) when received.

        Returns number of messages processed.
        """
        if self._connection is None:
            return 0
        processed = 0
        try:
            while processed < max_messages:
                msg = self._connection.recv_match(
                    type=["RAW_IMU", "SYSTEM_TIME"],
                    blocking=False,
                )
                if msg is None:
                    break
                processed += 1
                mtype = msg.get_type() if hasattr(msg, "get_type") else None
                if mtype == "RAW_IMU":
                    self._handle_raw_imu(msg)
                elif mtype == "SYSTEM_TIME":
                    self._handle_system_time(msg)
        except Exception:
            # Swallow polling exceptions; caller can keep running.
            return processed
        return processed

    def get_raw_imu_packets_since_last_call(self) -> List[dict]:
        """
        Return all RAW_IMU packets received since the last call to this method.
        Each packet is a dict with keys: time_usec, xacc, yacc, zacc, xgyro, ygyro, zgyro,
        xmag, ymag, zmag, id, and time_delta_sec (seconds from start_time to receive time).
        Only populated when the polling thread is running (start_polling_thread).
        """
        with self._poll_lock:
            packets = self._raw_imu_packets.copy()
            self._raw_imu_packets.clear()
        return packets

    def get_last_raw_imu_packets(self, max_packets: int = 200) -> List[dict]:
        """
        Return up to the last `max_packets` RAW_IMU packets seen (most recent last).
        Packets are dicts with the same keys and format as get_raw_imu_packets_since_last_call().
        """
        if max_packets <= 0:
            return []
        with self._poll_lock:
            return self._raw_imu_history[-max_packets:].copy()

    def get_raw_imu(self) -> Optional[dict]:
        """
        Return the last stored RAW_IMU message as a dict, or None if none received yet.
        When the polling thread is running, returns the latest packet (with time_delta_sec if set).
        Otherwise call poll_incoming() first to refresh.
        """
        if self._poll_thread is not None:
            with self._poll_lock:
                return self._raw_imu.copy() if self._raw_imu is not None else None
        self.poll_incoming()
        return self._raw_imu

    def get_boot_to_unix_offset_sec(self) -> Optional[float]:
        """
        Return the offset (seconds) such that unix_time_sec = time_boot_ms/1000 + offset,
        or None if no SYSTEM_TIME from target yet.
        """
        if self._poll_thread is not None:
            with self._poll_lock:
                return self._boot_to_unix_offset_sec
        self.poll_incoming()
        return self._boot_to_unix_offset_sec

    def boot_ms_to_unix_time_sec(self, time_boot_ms: int) -> Optional[float]:
        """
        Convert vehicle time_boot_ms to Unix time (seconds since epoch).
        Returns None if no SYSTEM_TIME from target has been received yet.
        """
        offset = self.get_boot_to_unix_offset_sec()
        if offset is None:
            return None
        return (time_boot_ms / 1000.0) + offset

    def get_last_system_time(self) -> Optional[Tuple[int, int]]:
        """
        Return (time_unix_usec, time_boot_ms) from the last SYSTEM_TIME message from the target,
        or None if none received yet.
        """
        if self._poll_thread is not None:
            with self._poll_lock:
                if self._system_time_unix_usec is None or self._system_time_boot_ms is None:
                    return None
                return (self._system_time_unix_usec, self._system_time_boot_ms)
        self.poll_incoming()
        if self._system_time_unix_usec is None or self._system_time_boot_ms is None:
            return None
        return (self._system_time_unix_usec, self._system_time_boot_ms)

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

    def publish_position_target(
        self,
        tvec: Union[np.ndarray, tuple, list],
        yaw,
        vertical_speed: float,
        timestamp: int,
    ) -> bool:

        if self._connection is None:
                return False
        try:
            x, y, z = self._to_xyz(tvec)
            # y = 0.0
            y = vertical_speed
            y = max(-20, min(20, y))
            # print(y)
            # y = 0.0
            target_system, target_component = self._get_target_ids()
            # Use position + yaw, ignore velocity/accel/yaw_rate.
            type_mask = (
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
                target_system=1,
                target_component=1,
                coordinate_frame=mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
                type_mask=type_mask,
                x=0,
                y=0,
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
            # print(
            #     f"Published position target to {target_system}.{target_component}: "
            #     f"forward={z}, right={x}, down={y}, yaw={yaw}, {type(yaw)}, timestamp={timestamp}"
            # )
            # print(y)
            
            return True
        except Exception as e:
            print(f"MavlinkPublisher: publish_position_target failed: {e}")
            return False


    def publish_landing_target(
        self,
        tvec: Union[np.ndarray, tuple, list],
        use_angles: bool = False,
        timestamp: int = None,
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
                    time_usec=timestamp*1000,
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
                    time_usec=timestamp*1000,
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

    def publish_odometry_tag_frame(
        self,
        time_usec: int,
        x: float,
        y: float,
        z: float,
        q: Union[tuple, list],
        reset_counter: int = 0,
        estimator_type: Optional[int] = None,
        quality: int = 0,
    ) -> bool:
        """
        Publish ODOMETRY (MAVLink common) with tag frame as origin.

        Sends only position and rotation (attitude); linear and angular velocities
        are sent as zero. Compatible with ArduPilot Non-GPS Position Estimation.
        https://ardupilot.org/dev/docs/mavlink-nongps-position-estimation.html

        Pose (x, y, z, q) is the body pose in the tag frame (tag = origin).
        frame_id = MAV_FRAME_LOCAL_FRD (20), child_frame_id = MAV_FRAME_BODY_FRD (12).

        Args:
            time_usec: Timestamp since system boot (microseconds); need not match autopilot.
            x, y, z: Position in tag frame (m), FRD: x forward, y right, z down (positive down).
            q: Quaternion (w, x, y, z) from tag frame to body frame; (1,0,0,0) = null rotation.
            reset_counter: Increment when the estimate resets (position/velocity/attitude).
            estimator_type: MAV_ESTIMATOR_TYPE; None = MAV_ESTIMATOR_TYPE_VISION.
            quality: 0 = unknown, -1 = failed, 1–100 = best. Below VISO_QUAL_MIN is ignored.

        Returns:
            True if sent, False if not connected or send failed.
        """
        if self._connection is None:
            return False
        try:
            q_list = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
            pose_cov = [float("nan")] * 21
            vel_cov = [float("nan")] * 21
            est_type = (
                int(estimator_type)
                if estimator_type is not None
                else mavutil.mavlink.MAV_ESTIMATOR_TYPE_VISION
            )
            self._connection.mav.odometry_send(
                time_usec=int(time_usec),
                frame_id=mavutil.mavlink.MAV_FRAME_LOCAL_FRD,
                child_frame_id=mavutil.mavlink.MAV_FRAME_BODY_FRD,
                x=float(x),
                y=float(y),
                z=float(z),
                q=q_list,
                vx=0.0,
                vy=0.0,
                vz=0.0,
                rollspeed=0.0,
                pitchspeed=0.0,
                yawspeed=0.0,
                pose_covariance=pose_cov,
                velocity_covariance=vel_cov,
                reset_counter=int(reset_counter) & 0xFF,
                estimator_type=est_type,
                quality=int(quality),
            )
            return True
        except Exception as e:
            print(f"MavlinkPublisher: publish_odometry_tag_frame failed: {e}")
            return False

    def publish_vision_position_estimate_tag_frame(
        self,
        time_usec: int,
        x: float,
        y: float,
        z: float,
        q: Union[tuple, list],
        reset_counter: int = 0,
        estimator_type: Optional[int] = None,
        quality: int = 0,
    ) -> bool:
        """
        Publish VISION_POSITION_ESTIMATE with tag frame as origin.

        Signature intentionally matches publish_odometry_tag_frame for drop-in use.
        VISION_POSITION_ESTIMATE does not carry estimator_type/quality and many stacks
        may ignore reset_counter; these are accepted for API compatibility.
        """
        if self._connection is None:
            return False
        try:
            q_list = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
            roll, pitch, yaw = Utils.rpy_from_quaternion(
                (q_list[0], q_list[1], q_list[2], q_list[3])
            )

            roll, pitch, yaw = -yaw, -roll, -pitch

            # print(f"roll: {roll}, pitch: {pitch}, yaw: {yaw}")

            covariance = [float("nan")] * 21
            self._connection.mav.vision_position_estimate_send(
                usec=int(time_usec),
                x=float(x),
                y=float(y),
                z=float(z),
                roll=float(roll),
                pitch=float(pitch),
                yaw=float(yaw),
                covariance=covariance,
                reset_counter=int(reset_counter) & 0xFF,
            )
            return True
        except Exception as e:
            print(f"MavlinkPublisher: publish_vision_position_estimate_tag_frame failed: {e}")
            return False

    def publish_optical_flow(
        self,
        time_usec: int,
        flow_x: float,
        flow_y: float,
        quality: int = 0,
        ground_distance: float = -1.0,
        flow_comp_m_x: float = 0.0,
        flow_comp_m_y: float = 0.0,
        sensor_id: int = 0,
    ) -> bool:
        """
        Publish OPTICAL_FLOW (MAVLink common).

        Args:
            time_usec: Timestamp (microseconds).
            flow_x, flow_y: Optical flow in pixels (frame-to-frame). Will be rounded and clamped to int16.
            quality: 0..255 quality/confidence.
            ground_distance: Distance to ground (meters), if known (0 if unknown).
            flow_comp_m_x, flow_comp_m_y: Flow in meters (integrated in the X/Y plane), if available.
            sensor_id: ID of the flow sensor (0..255).
        """
        if self._connection is None:
            return False
        try:
            fx_i16 = int(np.clip(np.rint(float(flow_x * FLOW_SCALE_X)), -32768, 32767))
            fy_i16 = int(np.clip(np.rint(float(flow_y * FLOW_SCALE_Y)), -32768, 32767))
            q_u8 = int(np.clip(int(quality), 0, 255))
            sid_u8 = int(np.clip(int(sensor_id), 0, 255))

            self._connection.mav.optical_flow_send(
                time_usec=int(time_usec),
                sensor_id=sid_u8,
                flow_x=fx_i16,
                flow_y=fy_i16,
                flow_comp_m_x=float(flow_comp_m_x),
                flow_comp_m_y=float(flow_comp_m_y),
                quality=q_u8,
                ground_distance=float(ground_distance),
            )
            return True
        except Exception as e:
            print(f"MavlinkPublisher: publish_optical_flow failed: {e}")
            return False

    def __enter__(self) -> "MavlinkPublisher":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
