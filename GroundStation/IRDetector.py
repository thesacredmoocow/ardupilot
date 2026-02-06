import cv2
import numpy as np
import json
import os
import itertools
import time
from picamera2 import Picamera2

THRESHOLD_VALUE = 240  # adjust based on lighting


class PiCameraFrameSource:
    def __init__(self, size=(2304, 1296), fps=30):
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": size, "format": "BGR888"},
            controls={"FrameRate": fps},
        )
        self.picam2.configure(config)
        self.picam2.start()
        self.picam2.set_controls({"ExposureTime": 300})
        time.sleep(0.5)

    def read(self):
        rgb = self.picam2.capture_array()
        rgb = cv2.resize(rgb, (640, 480))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return True, bgr

    def release(self):
        self.picam2.stop()

class UprightTPoseEstimator:
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = np.array(camera_matrix, dtype=np.float32)
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float32)
        self.last_debug = None

        # 3D Model Points (in cm)
        # Upside-down T: stem point is above the horizontal bar in image coords.
        self.model_points = np.array([
            (0.0,   0.0, 0.0),   # Bar-Middle (BM) - Origin
            (-5.0,  0.0, 0.0),   # Bar-Left (BL)
            (10.0,   0.0, 0.0),   # Bar-Right (BR)
            (0.0, -10.0, 0.0)    # Stem-Top (ST)
        ], dtype=np.float32)

        self.axis_points = np.array([
            (5.0, 0, 0), (0, 5.0, 0), (0, 0, 5.0)
        ], dtype=np.float32)

    def process_frame(self, frame, thresh_val):
        # 1. Pre-processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Thresholding (Intermediary 1)
        _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        # Remove small noise and fill tiny gaps before contouring.
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 3. Blob detection
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centers = []
        areas = []
        circularities = []
        min_area = 5.0
        min_circularity = 0.45
        min_solidity = 0.8
        max_aspect_ratio = 1.6
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            circularity = 4.0 * np.pi * area / (peri * peri) if peri > 0 else 0.0
            if circularity < min_circularity:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = max(w, h) / max(1.0, min(w, h))
            if aspect_ratio > max_aspect_ratio:
                continue
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0.0
            if solidity < min_solidity:
                continue
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = M["m10"] / M["m00"]
                cY = M["m01"] / M["m00"]
                centers.append([cX, cY])
                areas.append(area)
                circularities.append(circularity)
        # return frame, mask, debug_blob_frame
        # Need exactly 4 blobs
        if len(centers) < 4:
            print(f"Detected {len(centers)} blobs, need 4.")
            self.last_debug = {"mask": mask, "centers": centers, "sorted_pts": None}
            return None

        # 4. Select best 4 candidates in upside-down T formation
        all_centers = np.array(centers, dtype=np.float32)
        areas = np.array(areas, dtype=np.float32)
        circularities = np.array(circularities, dtype=np.float32)
        selected = self.select_t_points(all_centers, areas, circularities)
        if selected is None:
            self.last_debug = {"mask": mask, "centers": all_centers, "sorted_pts": None}
            return None
        centers = selected

        # Refine
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        refined_pts = cv2.cornerSubPix(gray, centers, (5, 5), (-1, -1), criteria)

        # Sort
        sorted_pts = self.sort_t_points(refined_pts)

        success, rvec, tvec = cv2.solvePnP(
            self.model_points, sorted_pts,
            self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP
        )
        cv2_results = (success, rvec, tvec)
        self.last_debug = {"mask": mask, "centers": all_centers, "sorted_pts": sorted_pts}
        return cv2_results

    def sort_t_points(self, points):
        pts = np.array(points)
        stem_idx = np.argmin(pts[:, 1])
        stem = pts[stem_idx]
        bar_row = np.delete(pts, stem_idx, axis=0)
        bar_row = bar_row[bar_row[:, 0].argsort()]
        return np.array([bar_row[1], bar_row[0], bar_row[2], stem], dtype=np.float32)

    def select_t_points(self, centers, areas, circularities, max_candidates=12, min_circularity=0.45):
        if centers.shape[0] < 4:
            return None

        # Prefer roughly circular blobs and keep only the strongest candidates.
        if circularities is not None and circularities.shape[0] == centers.shape[0]:
            valid = circularities >= min_circularity
            if np.count_nonzero(valid) >= 4:
                centers = centers[valid]
                areas = areas[valid]
                circularities = circularities[valid]

        if centers.shape[0] > max_candidates:
            idx = np.argsort(areas)[-max_candidates:]
            centers = centers[idx]

        best_score = None
        best_combo = None

        for combo in itertools.combinations(range(centers.shape[0]), 4):
            pts = centers[list(combo)]

            # Upside-down T: stem is highest point (min Y).
            stem_idx = np.argmin(pts[:, 1])
            stem = pts[stem_idx]
            bar = np.delete(pts, stem_idx, axis=0)

            # Bar should be roughly horizontal.
            bar_y = bar[:, 1]
            bar_y_std = np.std(bar_y)

            # Stem should be centered over the bar and above it.
            bar_x_center = np.mean(bar[:, 0])
            stem_x_offset = abs(stem[0] - bar_x_center)
            stem_above = np.mean(bar[:, 1]) - stem[1]

            # Bar should have some width.
            bar_width = np.ptp(bar[:, 0])

            if stem_above <= 2.0 or bar_width <= 6.0:
                continue

            # Score: lower is better.
            score = (
                bar_y_std * 2.0 +
                stem_x_offset * 0.5 +
                1.0 / (stem_above + 1e-3)
            )

            if best_score is None or score < best_score:
                best_score = score
                best_combo = pts

        if best_combo is None:
            return None

        return best_combo.astype(np.float32)

def nothing(x):
    pass

if __name__ == "__main__":
    # Load your specific calibration
    with open("camera_params.json", "r") as f:
        data = json.load(f)
        K, D = np.array(data["camera_matrix"]), np.array(data["dist_coeff"])

    estimator = UprightTPoseEstimator(K, D)

    cam = PiCameraFrameSource(size=(2304, 1296), fps=30) # updated to keep size consistent with calibration intrinsics

    # Create a tuning window
    cv2.namedWindow("Tuning")
    cv2.createTrackbar("Threshold", "Tuning", 240, 255, nothing)

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # t_val = cv2.getTrackbarPos("Threshold", "Tuning")
        cv2_results = estimator.process_frame(frame, THRESHOLD_VALUE)
        debug = estimator.last_debug

        if debug is not None:
            mask_out = debug["mask"]
            debug_blob_frame = frame.copy()
            for cX, cY in debug["centers"]:
                cv2.circle(debug_blob_frame, (int(cX), int(cY)), 5, (0, 255, 255), 1)

            if debug["sorted_pts"] is not None:
                labels = ["1: BM (origin)", "2: BL", "3: BR", "4: ST"]
                colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
                for i, pt in enumerate(debug["sorted_pts"]):
                    x, y = int(pt[0]), int(pt[1])
                    cv2.circle(debug_blob_frame, (x, y), 6, colors[i], -1)
                    cv2.putText(
                        debug_blob_frame,
                        labels[i],
                        (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        colors[i],
                        2
                    )

            # cv2.imshow("1. Binary Mask (Adjust Threshold)", mask_out)
            # cv2.imshow("2. Detected Blobs", debug_blob_frame)

        if cv2_results is not None:
            success, rvec, tvec = cv2_results
            if success:
                axis_length = 5.0  # same units as model points (cm)
                cv2.drawFrameAxes(
                    frame,
                    K,
                    D,
                    rvec,
                    tvec,
                    axis_length,
                    3
                )
                for i, pt in enumerate(debug["sorted_pts"]):
                    cv2.putText(frame, str(i), tuple(pt.astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imwrite("final_pose.jpg", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
