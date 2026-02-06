# host_viz_rerun.py
import socket
import msgpack
import rerun as rr

UDP_PORT = 5005

rr.init("drone_tf_debug", spawn=True)

# Pick a view convention (example: right-handed, Z-up)
rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # :contentReference[oaicite:4]{index=4}

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", UDP_PORT))
sock.settimeout(0.1)

print(f"Listening on UDP :{UDP_PORT}")

while True:
    try:
        data, addr = sock.recvfrom(65535)
    except TimeoutError:
        continue

    msg = msgpack.unpackb(data, raw=False)

    parent = msg["parent"]
    child = msg["child"]
    p = msg["p"]      # [x,y,z]
    q = msg["q"]      # [x,y,z,w]
    t_ns = msg["t_ns"]
    source = msg.get("source", "unknown")

    # Put each source in its own namespace so you can compare them visually
    # e.g. /tf/vio/world/drone/est
    entity = f"/tf/{source}{child}"

    rr.set_time_nanos("monotonic_ns", int(t_ns))

    # Log a transform at the child entity. (Rerun expects transforms along the entity hierarchy.)
    rr.log(entity, rr.Transform3D(translation=p, rotation=rr.Quaternion(xyzw=q), from_parent=True))

    # Optional: show axes explicitly at that entity
    rr.log(entity, rr.Arrows3D(origins=[[0, 0, 0]], vectors=[[0.3, 0, 0]]))
    rr.log(entity, rr.Arrows3D(origins=[[0, 0, 0]], vectors=[[0, 0.3, 0]]))
    rr.log(entity, rr.Arrows3D(origins=[[0, 0, 0]], vectors=[[0, 0, 0.3]]))
