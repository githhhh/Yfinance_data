from typing import Dict, Any, Optional, Tuple


class EPSGrowthCalculator:
    """Calculates Diluted EPS YoY Growth and categorizes special cases."""

    @staticmethod
    def calculate(eps_curr: Optional[float], eps_prior: Optional[float]) -> Tuple[Optional[float], str, bool]:
        """Calculates YoY growth percentage and classifies growth status.
        
        Returns:
            (eps_yoy_growth, growth_status, is_calculable)
        """
        if eps_curr is None or eps_prior is None:
            return None, "MISSING_DATA", False

        # Zero base
        if eps_prior == 0:
            return None, "ZERO_BASE", False

        # Near zero base (< 0.005)
        if abs(eps_prior) < 0.005:
            # Mathematical percentage would be erratic
            val = ((eps_curr - eps_prior) / abs(eps_prior)) * 100.0
            return val, "NEAR_ZERO_BASE", True

        # Normal positive
        if eps_prior > 0 and eps_curr > 0:
            val = ((eps_curr - eps_prior) / eps_prior) * 100.0
            return val, "NORMAL_POSITIVE", True

        # Profit to Loss
        if eps_prior > 0 and eps_curr <= 0:
            val = ((eps_curr - eps_prior) / eps_prior) * 100.0
            return val, "PROFIT_TO_LOSS", True

        # Loss to Profit (Turnaround)
        if eps_prior < 0 and eps_curr >= 0:
            # Standard financial math uses |prior| as denominator
            val = ((eps_curr - eps_prior) / abs(eps_prior)) * 100.0
            return val, "LOSS_TO_PROFIT", True

        # Both negative (Loss narrowing or widening)
        if eps_prior < 0 and eps_curr < 0:
            val = ((eps_curr - eps_prior) / abs(eps_prior)) * 100.0
            if eps_curr > eps_prior:
                return val, "LOSS_NARROWING", True
            else:
                return val, "LOSS_WIDENING", True

        return None, "UNKNOWN_CASE", False
