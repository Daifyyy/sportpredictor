import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from data.models import Fixture


class FeatureEngineer:
    """
    Feature pipeline for match prediction models.

    Call precompute(history) once before a training batch.
    Caches team index, H2H index, season index, Elo timeline, and avg_goals.
    build_features() computes home/away matches once per call and passes to sub-functions.
    """

    def __init__(self, form_window: int = 5):
        self.form_window = form_window
        self._cache_key: Optional[int] = None
        self._team_index:   Dict[int, List[Fixture]] = {}
        self._home_index:   Dict[int, List[Fixture]] = {}
        self._away_index:   Dict[int, List[Fixture]] = {}
        self._h2h_index:    Dict[frozenset, List[Fixture]] = {}
        self._season_index: Dict[int, List[Fixture]] = {}
        self._elo_by_fixture: Dict[int, Dict[int, float]] = {}
        self._elo_final:    Dict[int, float] = {}
        self._avg_goals:    float = 1.4
        # Sorted fixture IDs for binary past lookup
        self._sorted_fixtures: List[Fixture] = []
        self._fixture_pos:     Dict[int, int] = {}  # fixture_id → position in sorted list
        self._referee_index:   Dict[str, List[int]] = {}  # referee → list of total goals per game

    # ── Precompute ────────────────────────────────────────────────────────────

    def precompute(self, history: List[Fixture]) -> None:
        """Build all indexes and precompute Elo. O(n log n) — call once per training batch."""
        cache_key = id(history)
        if cache_key == self._cache_key:
            return
        self._cache_key = cache_key

        sorted_hist = sorted(
            [f for f in history if f.result is not None], key=lambda x: x.date
        )
        self._sorted_fixtures = sorted_hist
        self._fixture_pos = {f.id: i for i, f in enumerate(sorted_hist)}

        # Team / home / away / H2H / season indexes
        self._team_index = {}
        self._home_index = {}
        self._away_index = {}
        self._h2h_index = {}
        self._season_index = {}

        for f in sorted_hist:
            h, a = f.home_team.id, f.away_team.id
            pair = frozenset((h, a))

            self._team_index.setdefault(h, []).append(f)
            self._team_index.setdefault(a, []).append(f)
            self._home_index.setdefault(h, []).append(f)
            self._away_index.setdefault(a, []).append(f)
            self._h2h_index.setdefault(pair, []).append(f)
            self._season_index.setdefault(f.season, []).append(f)

        # Elo timeline — O(n), snapshot before each match
        elo: Dict[int, float] = {}
        K = 32
        self._elo_by_fixture = {}

        for f in sorted_hist:
            h, a = f.home_team.id, f.away_team.id
            elo.setdefault(h, 1500.0)
            elo.setdefault(a, 1500.0)
            self._elo_by_fixture[f.id] = {h: elo[h], a: elo[a]}  # only store needed teams
            exp_h = 1 / (1 + 10 ** ((elo[a] - elo[h]) / 400))
            score_h = {'H': 1.0, 'D': 0.5, 'A': 0.0}[f.result.outcome]
            elo[h] += K * (score_h - exp_h)
            elo[a] += K * ((1 - score_h) - (1 - exp_h))

        self._elo_final = dict(elo)

        # Rolling avg_goals snapshot per fixture — no leakage
        self._avg_goals_by_fixture: Dict[int, float] = {}
        cumulative_goals = 0.0
        for i, f in enumerate(sorted_hist):
            if i > 0:
                self._avg_goals_by_fixture[f.id] = (cumulative_goals / i) / 2
            else:
                self._avg_goals_by_fixture[f.id] = 1.4  # prior for first match
            cumulative_goals += f.result.home_goals + f.result.away_goals

        # Fallback: overall average (used for upcoming fixtures)
        if sorted_hist:
            self._avg_goals = float(np.mean(
                [f.result.home_goals + f.result.away_goals for f in sorted_hist]
            )) / 2

        # Referee index — total goals per game keyed by referee name
        self._referee_index = {}
        for f in sorted_hist:
            if f.referee:
                self._referee_index.setdefault(f.referee, []).append(
                    f.result.home_goals + f.result.away_goals
                )

    # ── Build features ────────────────────────────────────────────────────────

    def build_features(self, fixture: Fixture, history: List[Fixture]) -> Dict[str, float]:
        home_id = fixture.home_team.id
        away_id = fixture.away_team.id
        pair = frozenset((home_id, away_id))

        # Get past cutoff position once — O(1) with index
        pos = self._fixture_pos.get(fixture.id)
        if pos is not None and self._sorted_fixtures:
            # Use sorted index for O(log n) past lookup
            past_set = {f.id for f in self._sorted_fixtures[:pos]}
        else:
            # Fallback: linear filter
            past_set = {f.id for f in history if f.date < fixture.date and f.result is not None}

        # Compute home/away matches ONCE — O(team_matches) with index
        home_matches = [f for f in self._team_index.get(home_id, []) if f.id in past_set]
        away_matches = [f for f in self._team_index.get(away_id, []) if f.id in past_set]
        home_home_m  = [f for f in self._home_index.get(home_id, []) if f.id in past_set]
        away_away_m  = [f for f in self._away_index.get(away_id, []) if f.id in past_set]
        h2h_matches  = [f for f in self._h2h_index.get(pair, []) if f.id in past_set][-10:]
        season_home  = [f for f in self._season_index.get(fixture.season, [])
                        if f.id in past_set and (f.home_team.id == home_id or f.away_team.id == home_id)]
        season_away  = [f for f in self._season_index.get(fixture.season, [])
                        if f.id in past_set and (f.home_team.id == away_id or f.away_team.id == away_id)]

        # Elo — O(1) dict lookup
        elo_snap = self._elo_by_fixture.get(fixture.id, {})
        home_elo = elo_snap.get(home_id, self._elo_final.get(home_id, 1500.0))
        away_elo = elo_snap.get(away_id, self._elo_final.get(away_id, 1500.0))

        features: Dict[str, float] = {}
        features.update(self._form(home_matches, home_id, prefix="home"))
        features.update(self._form(away_matches, away_id, prefix="away"))
        features.update(self._venue_form(home_home_m, home_id, prefix="home", at_home=True))
        features.update(self._venue_form(away_away_m, away_id, prefix="away", at_home=False))
        features.update(self._h2h(h2h_matches, home_id))
        features.update({
            "elo_home": home_elo,
            "elo_away": away_elo,
            "elo_diff": home_elo - away_elo,
        })
        features.update(self._attack_defense(home_matches, away_matches, home_id, away_id, fixture.id))
        features.update(self._streak(home_matches, home_id, prefix="home"))
        features.update(self._streak(away_matches, away_id, prefix="away"))
        features.update(self._rest_days(home_matches, away_matches, fixture.date))
        features.update(self._trend(home_matches, home_id, prefix="home"))
        features.update(self._trend(away_matches, away_id, prefix="away"))
        features.update(self._consistency(home_matches, home_id, prefix="home"))
        features.update(self._consistency(away_matches, away_id, prefix="away"))
        features.update(self._season_ppg(season_home, season_away, home_id, away_id))
        features.update(self._stats_features(home_matches, away_matches, home_id, away_id))
        features.update(self._referee_features(fixture.referee))
        return features

    # ── Feature functions (accept pre-filtered matches) ───────────────────────

    def _form(self, matches: List[Fixture], team_id: int, prefix: str) -> Dict:
        recent = matches[-self.form_window:]
        if not recent:
            return {f"{prefix}_form": 0.0, f"{prefix}_gf": 0.0, f"{prefix}_ga": 0.0}
        pts, gf, ga = [], [], []
        for f in recent:
            is_home = f.home_team.id == team_id
            r = f.result
            scored   = r.home_goals if is_home else r.away_goals
            conceded = r.away_goals if is_home else r.home_goals
            o = r.outcome
            p = 3 if (o == 'H' and is_home) or (o == 'A' and not is_home) else 1 if o == 'D' else 0
            pts.append(p); gf.append(scored); ga.append(conceded)
        return {
            f"{prefix}_form": float(np.mean(pts)),
            f"{prefix}_gf":   float(np.mean(gf)),
            f"{prefix}_ga":   float(np.mean(ga)),
        }

    def _venue_form(self, matches: List[Fixture], team_id: int, prefix: str, at_home: bool) -> Dict:
        recent = matches[-self.form_window:]
        if not recent:
            return {f"{prefix}_venue_form": 0.0, f"{prefix}_venue_gf": 0.0, f"{prefix}_venue_ga": 0.0}
        pts, gf, ga = [], [], []
        for f in recent:
            r = f.result
            scored   = r.home_goals if at_home else r.away_goals
            conceded = r.away_goals if at_home else r.home_goals
            o = r.outcome
            p = 3 if (o == 'H' and at_home) or (o == 'A' and not at_home) else 1 if o == 'D' else 0
            pts.append(p); gf.append(scored); ga.append(conceded)
        return {
            f"{prefix}_venue_form": float(np.mean(pts)),
            f"{prefix}_venue_gf":   float(np.mean(gf)),
            f"{prefix}_venue_ga":   float(np.mean(ga)),
        }

    def _h2h(self, h2h: List[Fixture], home_id: int) -> Dict:
        if not h2h:
            return {"h2h_home_wins": 0.5, "h2h_draws": 0.33, "h2h_away_wins": 0.17,
                    "h2h_home_gf": 1.5, "h2h_away_gf": 1.2}
        total = len(h2h)
        home_wins = draws = 0
        home_gf_list, away_gf_list = [], []
        for f in h2h:
            r = f.result
            is_home = f.home_team.id == home_id
            o = r.outcome
            if (o == 'H' and is_home) or (o == 'A' and not is_home):
                home_wins += 1
            elif o == 'D':
                draws += 1
            home_gf_list.append(r.home_goals if is_home else r.away_goals)
            away_gf_list.append(r.away_goals if is_home else r.home_goals)
        return {
            "h2h_home_wins": home_wins / total,
            "h2h_draws":     draws / total,
            "h2h_away_wins": (total - home_wins - draws) / total,
            "h2h_home_gf":   float(np.mean(home_gf_list)),
            "h2h_away_gf":   float(np.mean(away_gf_list)),
        }

    def _attack_defense(self, home_m: List[Fixture], away_m: List[Fixture],
                        home_id: int, away_id: int, fixture_id: int = 0) -> Dict:
        avg = self._avg_goals_by_fixture.get(fixture_id, self._avg_goals)

        def strength(matches: List[Fixture], team_id: int) -> Tuple[float, float]:
            recent = matches[-10:]
            if not recent or avg == 0:
                return 1.0, 1.0
            gf, ga = [], []
            for f in recent:
                is_home = f.home_team.id == team_id
                r = f.result
                gf.append(r.home_goals if is_home else r.away_goals)
                ga.append(r.away_goals if is_home else r.home_goals)
            return float(np.mean(gf)) / avg, float(np.mean(ga)) / avg

        h_att, h_def = strength(home_m, home_id)
        a_att, a_def = strength(away_m, away_id)
        return {
            "home_attack_str":  round(h_att, 4),
            "home_defense_str": round(h_def, 4),
            "away_attack_str":  round(a_att, 4),
            "away_defense_str": round(a_def, 4),
            "attack_str_diff":  round(h_att - a_att, 4),
            "defense_str_diff": round(a_def - h_def, 4),
        }

    def _streak(self, matches: List[Fixture], team_id: int, prefix: str) -> Dict:
        if not matches:
            return {f"{prefix}_streak": 0.0}
        streak = 0
        last = None
        for f in reversed(matches):
            is_home = f.home_team.id == team_id
            o = f.result.outcome
            won  = (o == 'H' and is_home) or (o == 'A' and not is_home)
            lost = (o == 'A' and is_home) or (o == 'H' and not is_home)
            cur = 'W' if won else ('L' if lost else 'D')
            if last is None:
                last = cur
                streak = 1 if won else (-1 if lost else 0)
            elif cur == last and last != 'D':
                streak += 1 if won else -1
            else:
                break
        return {f"{prefix}_streak": float(streak)}

    def _rest_days(self, home_m: List[Fixture], away_m: List[Fixture],
                   fixture_date: datetime) -> Dict:
        def days(matches: List[Fixture]) -> float:
            if not matches:
                return 7.0
            return min(float((fixture_date - matches[-1].date).days), 30.0)
        h = days(home_m)
        a = days(away_m)
        return {"home_rest_days": h, "away_rest_days": a, "rest_days_diff": h - a}

    def _trend(self, matches: List[Fixture], team_id: int, prefix: str) -> Dict:
        def avg_pts(last_n: int) -> float:
            recent = matches[-last_n:] if len(matches) >= last_n else matches
            if not recent:
                return 1.0
            pts = []
            for f in recent:
                is_home = f.home_team.id == team_id
                o = f.result.outcome
                pts.append(3 if (o == 'H' and is_home) or (o == 'A' and not is_home) else 1 if o == 'D' else 0)
            return float(np.mean(pts))
        short = avg_pts(3)
        long_ = avg_pts(8)
        return {f"{prefix}_trend": round(short - long_, 4), f"{prefix}_form_short": round(short, 4)}

    def _consistency(self, matches: List[Fixture], team_id: int, prefix: str) -> Dict:
        recent = matches[-10:]
        if len(recent) < 3:
            return {f"{prefix}_gf_std": 1.0, f"{prefix}_ga_std": 1.0}
        gf, ga = [], []
        for f in recent:
            is_home = f.home_team.id == team_id
            gf.append(f.result.home_goals if is_home else f.result.away_goals)
            ga.append(f.result.away_goals if is_home else f.result.home_goals)
        return {
            f"{prefix}_gf_std": round(float(np.std(gf)), 4),
            f"{prefix}_ga_std": round(float(np.std(ga)), 4),
        }

    def _stats_features(self, home_matches: List[Fixture], away_matches: List[Fixture],
                        home_id: int, away_id: int) -> Dict:
        """Rolling average match statistics (shots, corners, xG) from last 5 matches per team."""

        def avg_stats(matches: List[Fixture], team_id: int) -> dict:
            recent = matches[-5:]
            shots_on, total_shots, corners, xg_vals = [], [], [], []
            for f in recent:
                s = f.home_stats if f.home_team.id == team_id else f.away_stats
                if s is None:
                    continue
                if s.shots_on_target is not None:
                    shots_on.append(s.shots_on_target)
                if s.total_shots is not None:
                    total_shots.append(s.total_shots)
                if s.corners is not None:
                    corners.append(s.corners)
                if s.xg is not None:
                    xg_vals.append(s.xg)
            return {
                "shots_on":    float(np.mean(shots_on))    if shots_on    else None,
                "total_shots": float(np.mean(total_shots)) if total_shots else None,
                "corners":     float(np.mean(corners))     if corners     else None,
                "xg":          float(np.mean(xg_vals))     if xg_vals     else None,
            }

        h = avg_stats(home_matches, home_id)
        a = avg_stats(away_matches, away_id)

        features: Dict[str, float] = {}

        if h["shots_on"] is not None and a["shots_on"] is not None:
            features["home_avg_shots_on_target"] = round(h["shots_on"], 3)
            features["away_avg_shots_on_target"] = round(a["shots_on"], 3)
            features["shots_on_target_diff"] = round(h["shots_on"] - a["shots_on"], 3)

        if h["total_shots"] is not None and a["total_shots"] is not None:
            features["home_avg_total_shots"] = round(h["total_shots"], 3)
            features["away_avg_total_shots"] = round(a["total_shots"], 3)

        if h["corners"] is not None and a["corners"] is not None:
            features["home_avg_corners"] = round(h["corners"], 3)
            features["away_avg_corners"] = round(a["corners"], 3)

        if h["xg"] is not None:
            features["home_avg_xg"] = round(h["xg"], 3)
        if a["xg"] is not None:
            features["away_avg_xg"] = round(a["xg"], 3)
        if h["xg"] is not None and a["xg"] is not None:
            features["xg_diff"] = round(h["xg"] - a["xg"], 3)

        return features

    def _referee_features(self, referee: Optional[str]) -> Dict:
        """Referee historical goal rate vs league average (display-only, does not affect model)."""
        if not referee:
            return {}
        games = self._referee_index.get(referee, [])
        n = len(games)
        league_avg_total = self._avg_goals * 2  # total goals/game
        result: Dict = {"referee_name": referee, "referee_n_games": n}
        if n >= 5 and league_avg_total > 0:
            ref_avg = sum(games) / n
            # Bayesian shrinkage toward league avg (prior weight = 20 games)
            blended = (n * ref_avg + 20 * league_avg_total) / (n + 20)
            factor = blended / league_avg_total
            result["referee_avg_goals"] = round(ref_avg, 3)
            result["referee_goal_factor"] = round(max(0.80, min(1.20, factor)), 4)
        return result

    def _season_ppg(self, home_season: List[Fixture], away_season: List[Fixture],
                    home_id: int, away_id: int) -> Dict:
        def ppg(matches: List[Fixture], team_id: int) -> float:
            if not matches:
                return 1.2
            pts = 0
            for f in matches:
                is_home = f.home_team.id == team_id
                o = f.result.outcome
                pts += 3 if (o == 'H' and is_home) or (o == 'A' and not is_home) else 1 if o == 'D' else 0
            return pts / len(matches)
        h = ppg(home_season, home_id)
        a = ppg(away_season, away_id)
        return {"home_season_ppg": round(h, 4), "away_season_ppg": round(a, 4), "season_ppg_diff": round(h - a, 4)}
