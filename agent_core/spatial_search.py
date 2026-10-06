"""Complete bounded field coverage; original spatial lookup when disabled."""
import bisect
import math
import os

from .required_guard import GuardedSurveyState


EPSILON_DEG = .001  # Covers four-decimal pointing rounding plus numerical edges.


class SpatialSurveyState(GuardedSurveyState):
    def __init__(self, payload):
        self.neighbour_search_enabled = os.environ.get('MERGE510_NEIGHBOUR_SEARCH', '1') != '0'
        self.spatial_queries = 0
        self.spatial_returned = 0
        self.spatial_anchor_queries = 0
        self.spatial_vectors_built = 0
        super().__init__(payload)

    def _build_index(self):
        super()._build_index()
        if self.neighbour_search_enabled:
            self._ra_keys = {key: [ra for ra, _ in band] for key, band in self._cells.items()}

    def neighbours(self, ra, dec, radius):
        if not self.neighbour_search_enabled:
            return super().neighbours(ra, dec, radius)
        if not all(math.isfinite(v) for v in (ra, dec, radius)) or radius < 0:
            return []
        ra %= 360.
        radius = min(180., radius)
        # The enclosing spherical cap crosses a pole -> every RA is possible.
        if radius >= 180. or abs(dec) + radius >= 90.:
            spans = [(0., 360.)]
        else:
            ratio = math.sin(math.radians(radius)) / math.cos(math.radians(dec))
            width = math.degrees(math.asin(min(1., max(0., ratio)))) + 1e-10
            lo, hi = ra-width, ra+width
            spans = ([(0., hi), (lo+360., 360.)] if lo < 0 else
                     [(lo, 360.), (0., hi-360.)] if hi >= 360 else [(lo, hi)])
        found = []
        low_band = math.floor(max(-90., dec-radius-1e-10))
        high_band = math.floor(min(90., dec+radius+1e-10))
        for key in range(low_band, high_band+1):
            band = self._cells.get(key)
            if not band:
                continue
            keys = self._ra_keys[key]
            for low, high in spans:
                start = bisect.bisect_left(keys, low)
                end = bisect.bisect_right(keys, high)
                found.extend(band[k][1] for k in range(start, end))
        self.spatial_queries += 1
        self.spatial_returned += len(found)
        return found

    def spatial_snapshot(self):
        return dict(enabled=self.neighbour_search_enabled, queries=self.spatial_queries,
                    returned_candidates=self.spatial_returned,
                    anchor_queries=self.spatial_anchor_queries,
                    vectors_built=self.spatial_vectors_built)


class FieldSearch:
    """One anchor query, cached target vectors, then exact field projection upstream.

    near_rows returns the existing joint-loop row layout with lazy gain rows.
    Caches are local to one search at one observation time, never reused at a
    different LST. Geometry and target tie ordering remain the caller's policy.
    """
    def __init__(self, state, grid, altaz, visible, fibers, legacy_radius):
        self.state, self.altaz, self.visible = state, altaz, visible
        self.vectors = {}
        angular = lambda length: math.degrees(math.atan(math.radians(length)))
        self.field_radius = angular(grid.fov/math.sqrt(2.)) + EPSILON_DEG
        offset = max((angular(math.hypot(*grid.fiber_center(f))) for f in fibers), default=0.)
        self.anchor_radius = max(legacy_radius, offset+self.field_radius+EPSILON_DEG)
        self.cos_radius = math.cos(math.radians(self.field_radius))

    def near_rows(self, anchor, edge_margin):
        state = self.state
        near = state.neighbours(state.ra[anchor], state.dec[anchor], self.anchor_radius)
        state.spatial_anchor_queries += 1
        rows = []
        for j in near:
            if j not in self.visible:
                continue
            vector = self.vectors.get(j)
            if vector is None:
                alt, az = map(math.radians, self.altaz(j))
                ca = math.cos(alt)
                vector = (ca*math.cos(az), ca*math.sin(az), math.sin(alt))
                self.vectors[j] = vector
                state.spatial_vectors_built += 1
            rows.append((j, None, *vector, edge_margin*(1+1.5*state.misses[j])))
        return rows

    def targets_in_cap(self, rows, c_alt, c_az):
        alt, az = map(math.radians, (c_alt, c_az))
        ca = math.cos(alt)
        cx, cy, cz = ca*math.cos(az), ca*math.sin(az), math.sin(alt)
        for j, _row, tx, ty, tz, _clear in rows:
            if tx*cx+ty*cy+tz*cz >= self.cos_radius:
                yield j
