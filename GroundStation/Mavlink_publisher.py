"""
MAVLink connection manager and DEBUG_VECT publishing for tvec/rvec.
"""

import time
from typing import Optional, Union
import os
import numpy as np
from pymavlink import mavutil

# DEBUG_VECT name field is 10 characters max
DEBUG_VECT_NAME_LEN = 10

os.environ['MAVLINK20'] = '1'

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
    ):
        self._address = address
        self._port = port
        self._source_system = source_system
        self._source_component = source_component
        self._connection: Optional[mavutil.mavlink_connection] = None

    def connect(self) -> bool:
        """Open TCP connection to the flight controller. Returns True if successful."""
        try:
            self._connection = mavutil.mavlink_connection(
                f"tcp:{self._address}:{self._port}",
                source_system=self._source_system,
                source_component=self._source_component,
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
    
    def __enter__(self) -> "MavlinkPublisher":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
