# udp_tf_client.py
from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict, Any

import msgpack
import numpy as np
import cv2


def _now_ns() -> int:
    return time.monotonic_ns()


def _rvec_to_quat_xyzw(rvec: np.ndarray) -> np.ndarray:
    """
    Convert OpenCV Rodrigues rotation vector to quaternion (x,y,z,w).
    """
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)  # 3x3

    # Convert rotation matrix to quaternion
    # Returns xyzw
    tr = float(np.trace(R))
    if tr > 0.0:
        S = (tr + 1.0) ** 0.5 * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        # Find the major diagonal element
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = (1.0 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5 * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = (1.0 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5 * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = (1.0 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5 * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    # Normalize for safety
    n = np.linalg.norm(q)
    if n > 0:
        q /= n
    return q.astype(np.float32)


@dataclass
class UDPclient:
    """
    Sends frame transforms over UDP for a host-side visualizer (Rerun) to consume.

    Message schema:
      {
        "seq": int,
        "t_ns": int,
        "parent": str,
        "child": str,
        "p": [x,y,z],
        "q": [x,y,z,w],
        "source": str,
        "static": bool
      }
    """
    host_ip: str
    host_port: int = 5005
    source: str = "vision"
    mtu: int = 1400

    def __post_init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._seq = 0

    def close(self) -> None:
        try:
            self._sock.close()
        except Exception:
            pass

    def send_transform(
        self,
        parent: str,
        child: str,
        p_xyz: Iterable[float],
        q_xyzw: Iterable[float],
        *,
        t_ns: Optional[int] = None,
        static: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        msg = {
            "seq": int(self._seq),
            "t_ns": int(_now_ns() if t_ns is None else t_ns),
            "parent": str(parent),
            "child": str(child),
            "p": [float(v) for v in p_xyz],
            "q": [float(v) for v in q_xyzw],
            "source": str(self.source),
            "static": bool(static),
        }
        if extra:
            msg["extra"] = extra

        payload = msgpack.packb(msg, use_bin_type=True)
        # keep it simple: if it doesn't fit MTU, drop (don’t fragment UDP)
        if len(payload) <= self.mtu:
            self._sock.sendto(payload, (self.host_ip, self.host_port))
            self._seq += 1

    def send_rvec_tvec(
        self,
        parent: str,
        child: str,
        rvec: np.ndarray,
        tvec: np.ndarray,
        *,
        t_ns: Optional[int] = None,
        static: bool = False,
        tvec_scale: float = 1.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Send an OpenCV pose (rvec,tvec) as a transform.

        tvec_scale: multiply tvec by this (e.g., 0.01 if your tvec is in cm and you want meters)
        """
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        p = (tvec * float(tvec_scale)).astype(np.float32)
        q = _rvec_to_quat_xyzw(rvec)
        self.send_transform(parent, child, p, q, t_ns=t_ns, static=static, extra=extra)
