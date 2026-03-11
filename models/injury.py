from typing import List, Tuple
from data.models import PlayerInjury


class InjuryAdjuster:
    """Adjusts Dixon-Coles λ/μ based on injured/suspended players.

    Attack impact (→ reduces team's λ):
      Attacker:   (goals + 0.7*assists) / team_goals
      Midfielder: same formula * 0.50 (split role, less goal-centric)

    Defense impact (→ increases opponent's λ):
      Defender:   (minutes / 3420) * 0.15
      Goalkeeper: (minutes / 3420) * 0.20

    3420 = 38 games × 90 min (season maximum).
    Questionable players apply 35% of their full impact.
    Total λ adjustment is capped at ±25%/±20%.
    """

    QUESTIONABLE_FACTOR = 0.35
    MAX_ATTACK_REDUCTION = 0.25   # max λ decrease
    MAX_DEFENSE_INCREASE = 0.20   # max opp_λ increase
    MIDFIELDER_OFFENSIVE_WEIGHT = 0.50

    def player_impact(self, inj: PlayerInjury, team_goals: int) -> Tuple[float, float]:
        """Returns (attack_delta, defense_delta) — fractional adjustments, uncapped.

        attack_delta: how much team's λ should decrease (positive value = reduction)
        defense_delta: how much opponent's λ should increase (positive value = increase)
        """
        factor = self.QUESTIONABLE_FACTOR if inj.status == "Questionable" else 1.0
        pos = inj.position

        if pos == "Attacker":
            attack = (inj.goals + 0.7 * inj.assists) / max(team_goals, 1) * factor
            return attack, 0.0

        if pos == "Midfielder":
            attack = (inj.goals + 0.7 * inj.assists) / max(team_goals, 1)
            attack *= self.MIDFIELDER_OFFENSIVE_WEIGHT * factor
            return attack, 0.0

        if pos == "Defender":
            defense = (inj.minutes / 3420) * 0.15 * factor
            return 0.0, defense

        if pos == "Goalkeeper":
            defense = (inj.minutes / 3420) * 0.20 * factor
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
    ) -> Tuple[float, float]:
        """Returns adjusted (λ, μ).

        home_injuries reduce λ (home attack) and/or increase μ (home defense weakened).
        away_injuries reduce μ (away attack) and/or increase λ (away defense weakened).
        """
        home_atk_red = home_def_inc = 0.0
        away_atk_red = away_def_inc = 0.0

        for inj in home_injuries:
            a, d = self.player_impact(inj, home_team_goals)
            home_atk_red += a
            home_def_inc += d

        for inj in away_injuries:
            a, d = self.player_impact(inj, away_team_goals)
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
