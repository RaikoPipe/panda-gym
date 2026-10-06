"""Minimal distance queries between robot links and obstacles (replaces the pyb_utils fork)."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pybullet as p


@dataclass
class NamedCollisionObject:
    """A body (key in the `bodies` dict) and optionally one of its links (None = base link)."""

    body_name: str
    link_name: Optional[str] = None


def _link_ids(client: int, uid: int) -> dict:
    return {p.getJointInfo(uid, i, client)[12].decode(): i for i in range(p.getNumJoints(uid, client))}


class CollisionDetector:
    def __init__(self, col_id: int, bodies: dict, named_collision_pairs: list):
        self.col_id = col_id
        self.bodies = bodies
        self.robot_id = bodies["robot"]
        self.robot_link_ids = _link_ids(col_id, self.robot_id)
        link_maps = {name: _link_ids(col_id, uid) for name, uid in bodies.items()}

        def index(obj):
            link = -1 if obj.link_name is None else link_maps[obj.body_name][obj.link_name]
            return bodies[obj.body_name], link

        self.indexed_collision_pairs = [(index(a), index(b)) for a, b in named_collision_pairs]

    def _closest(self, a, b, max_distance):
        """Closest point (distance, point on a, point on b) or None if farther than max_distance."""
        pts = p.getClosestPoints(a[0], b[0], max_distance, linkIndexA=a[1], linkIndexB=b[1], physicsClientId=self.col_id)
        if not pts:
            return None
        best = min(pts, key=lambda pt: pt[8])
        return best[8], best[5], best[6]

    def get_distances_per_link(self, max_distance=1.0) -> list:
        """Closest distance for every (link, obstacle) pair."""
        out = []
        for a, b in self.indexed_collision_pairs:
            c = self._closest(a, b, max_distance)
            out.append(max_distance if c is None else c[0])
        return out

    def compute_distances_per_link(self, max_distance=1.0):
        """Distances grouped by robot link, plus the closest point pairs per link in info['closest_points']."""
        distances, points = {}, {}
        for a, b in self.indexed_collision_pairs:
            distances.setdefault(a[1], [])
            points.setdefault(a[1], [])
            c = self._closest(a, b, max_distance)
            if c is None:
                distances[a[1]].append(max_distance)
            else:
                distances[a[1]].append(c[0])
                points[a[1]].append((c[1], c[2]))
        return distances, {"closest_points": points}

    def get_link_object_distance(self, object_name, object_link=-1, max_distance=1.0, ignore_link=()):
        """Closest distance of every robot link (except ignored ones) to one object."""
        out = []
        for name, link in self.robot_link_ids.items():
            if name in ignore_link:
                continue
            c = self._closest((self.robot_id, link), (self.bodies[object_name], object_link), max_distance)
            out.append(max_distance if c is None else c[0])
        return out
