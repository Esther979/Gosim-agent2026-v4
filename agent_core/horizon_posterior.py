"""Persistent azimuth-dependent terrain height from public scored fibre hits.

No pointing-offset estimator or calibration exposures. A returned zero-score
hit is different from an absent hit. Positive scores constrain ANY clear part
of an exposure, not its minimum altitude. Heights below the public observing
limit remain unidentifiable. Grid posterior intervals are conditional on this
piecewise-constant model and the robust binary likelihood, not hard boundaries.
"""
from __future__ import annotations

import math
from collections import defaultdict
from .geometry import radec_to_altaz, SIDEREAL_DEG_PER_SECOND

WARMUP_NIGHTS = 5
SECTOR_DEG = 5.0
N_SECTORS = 72
HEIGHTS = tuple(float(h) for h in range(91))
LABEL_ERROR = 0.03


def trajectory(ra, dec, lst, lat, seconds):
    """Equal-duration midpoint quadrature, at most 120 seconds per piece."""
    n = max(1, math.ceil(seconds / 120.0))
    step = seconds / n
    return [radec_to_altaz(ra, dec, lst + SIDEREAL_DEG_PER_SECOND * (k + .5) * step, lat)
            for k in range(n)]


def sector(az):
    return int((az % 360.0) // SECTOR_DEG) % N_SECTORS


class HorizonPosterior:
    def __init__(self, minimum_altitude):
        self.minimum_altitude = minimum_altitude
        self.announced = set()
        self.evidence = []  # (action_index, night_index, sector, [(max_alt, clear)])
        self.soft_evidence = []  # LLM-adjudicated returned zeros, never absent hits
        self._soft_seen = set()
        self._seen = set()
        self._weights = {}
        self._logs = {}
        self._cdf = {}
        self._nights = {}
        self.public_terrain = {}

    def set_public_terrain(self, facts):
        """Validated duty-log measurements are revisable soft priors, not hits."""
        wanted = dict(facts)
        if wanted != self.public_terrain:
            self.public_terrain = wanted
            self._rebuild()

    def public_fact(self, az):
        if not self.public_terrain:
            return None
        centers = {'N':0., 'NE':45., 'E':90., 'SE':135., 'S':180.,
                   'SW':225., 'W':270., 'NW':315.}
        # A compass-only statement supplies no exact obstruction boundary. Use
        # only its central 45-degree compass bin; retain fallback outside it.
        return next((r for d,r in self.public_terrain.items()
                     if abs((az-centers[d]+180)%360-180) <= 22.5), None)

    def announce(self, directions):
        changed = set(directions) - self.announced
        if changed:
            self.announced.update(changed)
            self._rebuild()

    def _prior(self, b):
        # Public coarse directions are priors, never exact azimuth boundaries.
        centers = {'N': 0, 'NE': 45, 'E': 90, 'SE': 135, 'S': 180, 'SW': 225, 'W': 270, 'NW': 315}
        az = (b + .5) * SECTOR_DEG
        near = any(abs((az - centers[d] + 180) % 360 - 180) <= 67.5
                   for d in self.announced if d in centers)
        raised = .6 if near else .15
        prior = [1.0 - raised] + [raised / 90.0] * 90
        fact = self.public_fact(az)
        if fact:
            lo = (0. if fact.get('relation') == 'upper_bound'
                  else max(0., fact['height_deg']-fact['uncertainty_deg']))
            hi = min(90., fact['height_deg']+fact['uncertainty_deg'])
            support = [h for h in HEIGHTS if lo <= h <= hi]
            confidence = min(.98, fact['confidence'])
            prior = [(1-confidence)*p + (confidence/len(support) if h in support else 0.)
                     for h,p in zip(HEIGHTS,prior)]
        return prior

    def add(self, action_index, night_index, samples):
        """samples: (trajectory [(alt, az)], scored_positive).

        Only single-sector paths identify one height without assumptions about
        neighbouring sectors. Correlated fibres contribute ONE average log
        likelihood per sector/exposure. Multiple nights are required for use.
        """
        if not 0 <= night_index < WARMUP_NIGHTS or action_index is None or action_index in self._seen:
            return 0
        self._seen.add(action_index)
        groups = defaultdict(list)
        for path, clear in samples:
            bins = {sector(az) for alt, az in path}
            if len(bins) != 1 or not path:
                continue
            groups[next(iter(bins))].append((max(alt for alt, az in path), bool(clear)))
        for b, labels in groups.items():
            self.evidence.append((action_index, night_index, b, labels))
            self._nights.setdefault(b, set()).add(night_index)
            self._apply_labels(b, labels)
            self._refresh(b)
        return len(groups)

    def invalidate(self, window):
        lo, hi = window.get('action_index_start'), window.get('action_index_end_exclusive')
        if lo is None or hi is None:
            self.evidence.clear()
            self._seen.clear()
            self.soft_evidence.clear()
            self._soft_seen.clear()
        else:
            self.evidence[:] = [r for r in self.evidence if not lo <= r[0] < hi]
            self._seen = {i for i in self._seen if not lo <= i < hi}
            self.soft_evidence[:] = [r for r in self.soft_evidence if not lo <= r[0] < hi]
            self._soft_seen = {i for i in self._soft_seen if not lo <= i < hi}
        self._rebuild()

    def add_soft(self, action_index, night_index, samples):
        """samples: (path, LLM horizon probability). Caller verifies returned zero.

        At most 0.25 of an ordinary exposure likelihood, averaged across correlated
        fibres. This confidence is a model judgment, not a calibrated probability.
        """
        if not 0<=night_index<WARMUP_NIGHTS or action_index is None or action_index in self._soft_seen:
            return 0
        self._soft_seen.add(action_index)
        groups=defaultdict(list)
        for path,confidence in samples:
            bins={sector(az) for alt,az in path}
            if len(bins)!=1 or not path or not math.isfinite(confidence) or not .8<=confidence<=1:continue
            groups[next(iter(bins))].append((max(alt for alt,az in path),confidence))
        for b,labels in groups.items():
            self.soft_evidence.append((action_index,night_index,b,labels))
            self._nights.setdefault(b,set()).add(night_index)
            self._apply_soft(b,labels)
            self._refresh(b)
        return len(groups)

    def _rebuild(self):
        groups = defaultdict(list)
        self._nights = defaultdict(set)
        for _, night, b, labels in self.evidence:
            groups[b].append(labels)
            self._nights[b].add(night)
        soft=defaultdict(list)
        for _,night,b,labels in self.soft_evidence:
            soft[b].append(labels)
            self._nights[b].add(night)
        self._weights.clear()
        self._cdf.clear()
        self._logs.clear()
        public_bins = {b for b in range(N_SECTORS) if self.public_fact((b+.5)*SECTOR_DEG)}
        for b in groups.keys() | soft.keys() | public_bins:
            self._logs[b] = [math.log(p) for p in self._prior(b)]
            for labels in groups[b]:
                self._apply_labels(b, labels)
            for labels in soft[b]:
                self._apply_soft(b,labels)
            self._refresh(b)

    def _apply_soft(self,b,labels):
        logs=self._logs.setdefault(b,[math.log(p) for p in self._prior(b)])
        right,wrong=math.log(1-LABEL_ERROR),math.log(LABEL_ERROR)
        for j,height in enumerate(HEIGHTS):
            logs[j]+=.25*sum(confidence*(right if height>=alt else wrong)
                            for alt,confidence in labels)/len(labels)

    def _apply_labels(self, b, labels):
        logs = self._logs.setdefault(b, [math.log(p) for p in self._prior(b)])
        right, wrong = math.log(1.0 - LABEL_ERROR), math.log(LABEL_ERROR)
        for j, height in enumerate(HEIGHTS):
            match = sum((alt > height) == clear for alt, clear in labels) / len(labels)
            logs[j] += match * right + (1.0 - match) * wrong

    def _refresh(self, b):
        logs = self._logs[b]
        scale = max(logs)
        weights = [math.exp(v - scale) for v in logs]
        total = sum(weights)
        weights = [p / total for p in weights]
        self._weights[b] = weights
        cumulative, cdf = 0.0, []
        for p in weights:
            cumulative += p
            cdf.append(min(1.0, cumulative))
        cdf[-1] = 1.0
        self._cdf[b] = cdf

    def informed(self, az):
        return (bool(self.public_terrain) and self.public_fact(az) is not None
                or len(self._nights.get(sector(az), ())) >= 2)

    def clear_probability(self, alt, az):
        cdf = self._cdf.get(sector(az))
        if cdf is None:
            return None
        # Terrain covers altitude <= H, so equality must count as blocked.
        j = min(90, math.ceil(alt) - 1)
        return cdf[j] if j >= 0 else 0.0

    def visibility_distribution(self, path, public_only=False):
        """Posterior distribution of clear exposure fraction.

        One latent H per 5-degree sector; sector heights are independent in this
        model, while all time pieces within a sector share the same H. Convolve
        clear-piece counts instead of multiplying independent time-point risks.
        Uninformed sectors contribute no new restriction (planner fallback).
        """
        groups = defaultdict(list)
        for alt, az in path:
            b = sector(az)
            if not self.informed(az) or public_only and self.public_fact(az) is None:
                groups[None].append(alt)
            else:
                groups[b].append(alt)
        counts = {0: 1.0}
        for b, alts in groups.items():
            distribution = defaultdict(float)
            if b is None:
                distribution[len(alts)] = 1.0
            else:
                # CDF differences give the exact clear-piece distribution in O(n),
                # avoiding a height-grid scan for every target / exposure candidate.
                previous = 0.0
                cdf = self._cdf[b]
                for k, alt in enumerate(sorted(alts)):
                    j = min(90, math.ceil(alt) - 1)
                    below = cdf[j] if j >= 0 else 0.0
                    if below > previous:
                        distribution[len(alts) - k] += below - previous
                    previous = below
                if previous < 1.0:
                    distribution[0] += max(0.0, 1.0 - previous)
            updated = defaultdict(float)
            for before, p in counts.items():
                for extra, q in distribution.items():
                    updated[before + extra] += p * q
            counts = updated
        n = len(path)
        return [(count / n, p) for count, p in counts.items() if p > 1e-12] if n else [(1.0, 1.0)]

    def snapshot(self):
        def quantile(cdf, q):
            return next((h for h, p in zip(HEIGHTS, cdf) if p >= q), 90.0)
        rows = []
        for b in sorted(self._weights):
            cdf = self._cdf[b]
            records = [r for r in self.evidence if r[2] == b]
            soft_records=[r for r in self.soft_evidence if r[2]==b]
            rows.append(dict(azimuth_range_deg=[b * SECTOR_DEG, (b + 1) * SECTOR_DEG],
                nights=len(self._nights.get(b, ())), used=self.informed((b+.5)*SECTOR_DEG),
                public_log_prior=self.public_fact((b+.5)*SECTOR_DEG) is not None,
                evidence_exposures=len(records),
                positive_hits=sum(clear for r in records for alt,clear in r[3]),
                controlled_zero_hits=sum(not clear for r in records for alt,clear in r[3]),
                llm_soft_zero_hits=sum(len(r[3]) for r in soft_records),
                height_p05_deg=quantile(cdf, .05), height_p50_deg=quantile(cdf, .5),
                height_p95_deg=quantile(cdf, .95),
                probability_above_observing_limit=1.0 - cdf[min(90, int(self.minimum_altitude))]))
        return dict(warmup_nights=WARMUP_NIGHTS, sector_deg=SECTOR_DEG,
                    evidence_exposures=len({r[0] for r in self.evidence}),
                    llm_soft_exposures=len({r[0] for r in self.soft_evidence}),
                    sectors=rows, unobserved_sectors=N_SECTORS - len(rows),
                    below_observing_limit_identifiable=False)
