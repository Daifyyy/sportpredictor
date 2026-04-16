from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from data.models import PlayerInjury


class InjuryAdjuster:
    """Adjusts Dixon-Coles λ/μ based on injured/suspended players.

    Attack impact (→ reduces team's λ):
      Attacker:   per-90 goal contribution relative to team avg goals/game
      Midfielder: same, weighted by dynamic G+A tier (0.15–0.70)

    Defense impact (→ increases opponent's λ):
      Defender:   (minutes / 3420) * 0.15 * defense_quality_factor
      Goalkeeper: (minutes / 3420) * 0.20 * defense_quality_factor

    defense_quality_factor scales by team's goals_against: elite defenses concede
    fewer goals so losing a key defender matters proportionally more.
    3420 = 38 games × 90 min (season maximum).
    Minimum 450 min threshold for per-90 calc to filter super-sub noise.
    Questionable players apply 35% of their full impact.
    Total λ adjustment is capped at ±25%/±20%.
    """

    QUESTIONABLE_FACTOR = 0.35
    MAX_ATTACK_REDUCTION = 0.25   # max λ decrease
    MAX_DEFENSE_INCREASE = 0.20   # max opp_λ increase
    MIN_MEANINGFUL_MINUTES = 450  # below this, fall back to raw ratio

    @staticmethod
    def _midfielder_weight(goals: int, assists: int) -> float:
        """Dynamic weight by G+A total — attacking midfielders penalized harder."""
        ga = goals + assists
        if ga >= 12:
            return 0.70
        if ga >= 6:
            return 0.50
        if ga >= 2:
            return 0.30
        return 0.15

    def player_impact(
        self,
        inj: PlayerInjury,
        team_goals: int,
        team_goals_against: int = 50,
    ) -> Tuple[float, float]:
        """Returns (attack_delta, defense_delta) — fractional adjustments, uncapped.

        attack_delta: how much team's λ should decrease (positive value = reduction)
        defense_delta: how much opponent's λ should increase (positive value = increase)
        team_goals_against: used to scale defender/GK impact by defensive quality.
        """
        factor = self.QUESTIONABLE_FACTOR if inj.status == "Questionable" else 1.0
        pos = inj.position

        if pos in ("Attacker", "Midfielder"):
            if inj.minutes < self.MIN_MEANINGFUL_MINUTES:
                # Too few minutes — raw ratio avoids per-90 inflation
                contribution = (inj.goals + 0.7 * inj.assists) / max(team_goals, 1)
            else:
                g_per_90 = (inj.goals + 0.7 * inj.assists) / (inj.minutes / 90)
                team_g_per_game = max(team_goals / 38, 0.5)
                contribution = g_per_90 / team_g_per_game

            if pos == "Midfielder":
                contribution *= self._midfielder_weight(inj.goals, inj.assists)

            return contribution * factor, 0.0

        # defense_quality_factor: elite defenses (few goals against) → key defender
        # absence matters proportionally more (avg ~1.3 goals/game as baseline)
        defense_quality = min(max(1.3 / max(team_goals_against / 38, 0.3), 0.5), 2.0)

        if pos == "Defender":
            defense = (inj.minutes / 3420) * 0.15 * factor * defense_quality
            return 0.0, defense

        if pos == "Goalkeeper":
            defense = (inj.minutes / 3420) * 0.20 * factor * defense_quality
            return 0.0, defense

        return 0.0, 0.0

    def adjust(
        self,
        lam: float,
        mu: float,
        home_injuries: List[PlayerInjury],
        away_injuries: List[PlayerInjury],
        home_team_goals: int,
        away_team_goals: int,
        home_team_goals_against: int = 50,
        away_team_goals_against: int = 50,
    ) -> Tuple[float, float]:
        """Returns adjusted (λ, μ).

        home_injuries reduce λ (home attack) and/or increase μ (home defense weakened).
        away_injuries reduce μ (away attack) and/or increase λ (away defense weakened).
        """
        home_atk_red = home_def_inc = 0.0
        away_atk_red = away_def_inc = 0.0

        for inj in home_injuries:
            a, d = self.player_impact(inj, home_team_goals, home_team_goals_against)
            home_atk_red += a
            home_def_inc += d

        for inj in away_injuries:
            a, d = self.player_impact(inj, away_team_goals, away_team_goals_against)
            away_atk_red += a
            away_def_inc += d

        # Cap
        home_atk_red = min(home_atk_red, self.MAX_ATTACK_REDUCTION)
        home_def_inc = min(home_def_inc, self.MAX_DEFENSE_INCREASE)
        away_atk_red = min(away_atk_red, self.MAX_ATTACK_REDUCTION)
        away_def_inc = min(away_def_inc, self.MAX_DEFENSE_INCREASE)

        lam_adj = lam * (1 - home_atk_red) * (1 + away_def_inc)
        mu_adj  = mu  * (1 - away_atk_red) * (1 + home_def_inc)

        return round(lam_adj, 4), round(mu_adj, 4)
