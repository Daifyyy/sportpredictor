import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from data.models import Fixture


class FeatureEngineer:
    """
    Feature pipeline for XGBoost.
    Each method returns a dict of features — build_features merges them all.
    Adding new features: add a new method and call it in build_features.
    """

    def __init__(self, form_window: int = 5):
        self.form_window = form_window

    def build_features(self, fixture: Fixture, history: List[Fixture]) -> Dict[str, float]:
        home_id = fixture.home_team.id
        away_id = fixture.away_team.id

        # Only use history before this fixture to avoid leakage
        past = [f for f in history if f.date < fixture.date and f.result is not None]

        features: Dict[str, float] = {}
        features.update(self._form_features(home_id, past, prefix="home"))
        features.update(self._form_features(away_id, past, prefix="away"))
        features.update(self._venue_form(home_id, past, prefix="home", at_home=True))
        features.update(self._venue_form(away_id, past, prefix="away", at_home=False))
        features.update(self._h2h_features(home_id, away_id, past))
        features.update(self._elo_features(home_id, away_id, past))
        features.update(self._attack_defense_strength(home_id, away_id, past))
        features.update(self._streak(home_id, past, prefix="home"))
        features.update(self._streak(away_id, past, prefix="away"))
        features.update(self._rest_days(home_id, away_id, past, fixture.date))
        features.update(self._trend(home_id, past, prefix="home"))
        features.update(self._trend(away_id, past, prefix="away"))
        features.update(self._consistency(home_id, past, prefix="home"))
        features.update(self._consistency(away_id, past, prefix="away"))
        features.update(self._season_points(home_id, away_id, past, fixture.season))
        return features

    # ── Existing features (improved) ─────────────────────────────────────────

    def _form_features(self, team_id: int, history: List[Fixture], prefix: str) -> Dict:
        matches = self._team_matches(team_id, history)[-self.form_window:]
        if not matches:
            return {f"{prefix}_form": 0.0, f"{prefix}_gf": 0.0, f"{prefix}_ga": 0.0}

        points, gf, ga = [], [], []
        for f in matches:
            is_home = f.home_team.id == team_id
            r = f.result
            scored = r.home_goals if is_home else r.away_goals
            conceded = r.away_goals if is_home else r.home_goals
            outcome = r.outcome
            pts = 3 if (outcome == 'H' and is_home) or (outcome == 'A' and not is_home) else \
                  1 if outcome == 'D' else 0
            points.append(pts)
            gf.append(scored)
            ga.append(conceded)

        return {
            f"{prefix}_form": np.mean(points) if points else 0.0,
            f"{prefix}_gf":   np.mean(gf)     if gf     else 0.0,
            f"{prefix}_ga":   np.mean(ga)      if ga     else 0.0,
        }

    def _h2h_features(self, home_id: int, away_id: int, history: List[Fixture]) -> Dict:
        h2h = [
            f for f in history
            if {f.home_team.id, f.away_team.id} == {home_id, away_id}
        ][-10:]

        if not h2h:
            return {"h2h_home_wins": 0.5, "h2h_draws": 0.33, "h2h_away_wins": 0.17,
                    "h2h_home_gf": 1.5, "h2h_away_gf": 1.2}

        total = len(h2h)
        home_wins = sum(
            1 for f in h2h
            if (f.home_team.id == home_id and f.result.outcome == 'H') or
               (f.away_team.id == home_id and f.result.outcome == 'A')
        )
        draws = sum(1 for f in h2h if f.result.outcome == 'D')

        home_gf = np.mean([
            r.home_goals if f.home_team.id == home_id else r.away_goals
            for f in h2h if (r := f.result)
        ])
        away_gf = np.mean([
            r.away_goals if f.home_team.id == home_id else r.home_goals
            for f in h2h if (r := f.result)
        ])
        return {
            "h2h_home_wins": home_wins / total,
            "h2h_draws":     draws / total,
            "h2h_away_wins": (total - home_wins - draws) / total,
            "h2h_home_gf":   float(home_gf),
            "h2h_away_gf":   float(away_gf),
        }

    def _elo_features(self, home_id: int, away_id: int, history: List[Fixture]) -> Dict:
        elo: Dict[int, float] = {}
        K = 32

        for f in sorted(history, key=lambda x: x.date):
            h, a = f.home_team.id, f.away_team.id
            elo.setdefault(h, 1500.0)
            elo.setdefault(a, 1500.0)
            exp_h = 1 / (1 + 10 ** ((elo[a] - elo[h]) / 400))
            score_h = {'H': 1.0, 'D': 0.5, 'A': 0.0}[f.result.outcome]
            elo[h] += K * (score_h - exp_h)
            elo[a] += K * ((1 - score_h) - (1 - exp_h))

        home_elo = elo.get(home_id, 1500.0)
        away_elo = elo.get(away_id, 1500.0)
        return {
            "elo_home": home_elo,
            "elo_away": away_elo,
            "elo_diff": home_elo - away_elo,
        }

    # ── New features ─────────────────────────────────────────────────────────

    def _venue_form(self, team_id: int, history: List[Fixture], prefix: str, at_home: bool) -> Dict:
        """Form split by home/away venue — teams often perform very differently."""
        if at_home:
            matches = [f for f in history if f.home_team.id == team_id][-self.form_window:]
        else:
            matches = [f for f in history if f.away_team.id == team_id][-self.form_window:]

        if not matches:
            return {f"{prefix}_venue_form": 0.0, f"{prefix}_venue_gf": 0.0, f"{prefix}_venue_ga": 0.0}

        points, gf, ga = [], [], []
        for f in matches:
            r = f.result
            scored = r.home_goals if at_home else r.away_goals
            conceded = r.away_goals if at_home else r.home_goals
            outcome = r.outcome
            pts = 3 if (outcome == 'H' and at_home) or (outcome == 'A' and not at_home) else \
                  1 if outcome == 'D' else 0
            points.append(pts)
            gf.append(scored)
            ga.append(conceded)

        return {
            f"{prefix}_venue_form": np.mean(points) if points else 0.0,
            f"{prefix}_venue_gf":   np.mean(gf)     if gf     else 0.0,
            f"{prefix}_venue_ga":   np.mean(ga)      if ga     else 0.0,
        }

    def _attack_defense_strength(self, home_id: int, away_id: int, history: List[Fixture]) -> Dict:
        """Goals scored/conceded relative to league average."""
        completed = [f for f in history if f.result is not None]
        if not completed:
            return {"home_attack_str": 1.0, "home_defense_str": 1.0,
                    "away_attack_str": 1.0, "away_defense_str": 1.0}

        avg_goals = np.mean([f.result.home_goals + f.result.away_goals for f in completed]) / 2

        def strength(team_id: int):
            matches = self._team_matches(team_id, history)[-10:]
            if not matches or avg_goals == 0:
                return 1.0, 1.0
            gf_list, ga_list = [], []
            for f in matches:
                r = f.result
                is_home = f.home_team.id == team_id
                gf_list.append(r.home_goals if is_home else r.away_goals)
                ga_list.append(r.away_goals if is_home else r.home_goals)
            return np.mean(gf_list) / avg_goals, np.mean(ga_list) / avg_goals

        h_att, h_def = strength(home_id)
        a_att, a_def = strength(away_id)
        return {
            "home_attack_str":  round(float(h_att), 4),
            "home_defense_str": round(float(h_def), 4),
            "away_attack_str":  round(float(a_att), 4),
            "away_defense_str": round(float(a_def), 4),
            "attack_str_diff":  round(float(h_att - a_att), 4),
            "defense_str_diff": round(float(a_def - h_def), 4),  # high = home defense stronger
        }

    def _streak(self, team_id: int, history: List[Fixture], prefix: str) -> Dict:
        """Current win/loss/draw streak. Positive = wins, negative = losses."""
        matches = self._team_matches(team_id, history)
        if not matches:
            return {f"{prefix}_streak": 0}

        streak = 0
        last_result = None
        for f in reversed(matches):
            is_home = f.home_team.id == team_id
            outcome = f.result.outcome
            won = (outcome == 'H' and is_home) or (outcome == 'A' and not is_home)
            lost = (outcome == 'A' and is_home) or (outcome == 'H' and not is_home)
            drew = outcome == 'D'

            if last_result is None:
                last_result = 'W' if won else ('L' if lost else 'D')
                streak = 1 if won else (-1 if lost else 0)
            else:
                current = 'W' if won else ('L' if lost else 'D')
                if current == last_result and last_result != 'D':
                    streak += 1 if won else -1
                else:
                    break

        return {f"{prefix}_streak": float(streak)}

    def _rest_days(self, home_id: int, away_id: int, history: List[Fixture],
                   fixture_date: datetime) -> Dict:
        """Days since last match — fatigue proxy."""
        def last_match_days(team_id: int) -> float:
            matches = self._team_matches(team_id, history)
            if not matches:
                return 7.0  # neutral default
            last_date = matches[-1].date
            delta = fixture_date - last_date
            return min(float(delta.days), 30.0)  # cap at 30

        return {
            "home_rest_days": last_match_days(home_id),
            "away_rest_days": last_match_days(away_id),
            "rest_days_diff": last_match_days(home_id) - last_match_days(away_id),
        }

    def _trend(self, team_id: int, history: List[Fixture], prefix: str) -> Dict:
        """Recent trend: last 3 form vs last 8 form. Positive = improving."""
        matches = self._team_matches(team_id, history)

        def avg_pts(last_n: int) -> float:
            recent = matches[-last_n:] if len(matches) >= last_n else matches
            if not recent:
                return 1.0
            pts_list = []
            for f in recent:
                is_home = f.home_team.id == team_id
                outcome = f.result.outcome
                pts = 3 if (outcome == 'H' and is_home) or (outcome == 'A' and not is_home) else \
                      1 if outcome == 'D' else 0
                pts_list.append(pts)
            return np.mean(pts_list)

        short = avg_pts(3)
        long_ = avg_pts(8)
        return {
            f"{prefix}_trend": round(float(short - long_), 4),  # positive = improving
            f"{prefix}_form_short": round(float(short), 4),
        }

    def _consistency(self, team_id: int, history: List[Fixture], prefix: str) -> Dict:
        """Goals scored variance — low = consistent, high = unpredictable."""
        matches = self._team_matches(team_id, history)[-10:]
        if len(matches) < 3:
            return {f"{prefix}_gf_std": 1.0, f"{prefix}_ga_std": 1.0}

        gf_list, ga_list = [], []
        for f in matches:
            is_home = f.home_team.id == team_id
            gf_list.append(f.result.home_goals if is_home else f.result.away_goals)
            ga_list.append(f.result.away_goals if is_home else f.result.home_goals)

        return {
            f"{prefix}_gf_std": round(float(np.std(gf_list)), 4),
            f"{prefix}_ga_std": round(float(np.std(ga_list)), 4),
        }

    def _season_points(self, home_id: int, away_id: int, history: List[Fixture],
                       season: int) -> Dict:
        """Cumulative points in current season — proxy for table position."""
        season_matches = [f for f in history if f.season == season]

        def pts_in_season(team_id: int) -> float:
            matches = self._team_matches(team_id, season_matches)
            total = 0
            for f in matches:
                is_home = f.home_team.id == team_id
                outcome = f.result.outcome
                total += 3 if (outcome == 'H' and is_home) or (outcome == 'A' and not is_home) else \
                         1 if outcome == 'D' else 0
            return float(total)

        def ppg(team_id: int) -> float:
            matches = self._team_matches(team_id, season_matches)
            if not matches:
                return 1.2  # league average ~1.2 ppg
            return pts_in_season(team_id) / len(matches)

        home_ppg = ppg(home_id)
        away_ppg = ppg(away_id)
        return {
            "home_season_ppg": round(home_ppg, 4),
            "away_season_ppg": round(away_ppg, 4),
            "season_ppg_diff": round(home_ppg - away_ppg, 4),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _team_matches(self, team_id: int, history: List[Fixture]) -> List[Fixture]:
        return sorted(
            [f for f in history
             if (f.home_team.id == team_id or f.away_team.id == team_id)
             and f.result is not None],
            key=lambda x: x.date
        )
