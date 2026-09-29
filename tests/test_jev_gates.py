"""
Unit tests for Apex JEV integration gate functions.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
import apex_jev as jev

# Import the module under test
import apex_jev as jev


class TestJEVGates:
    """Test JEV gate functions with mocked decisions."""

    @pytest.fixture
    def mock_jev_decisions(self):
        """Base JEV decisions for testing."""
        return {
            "regime": {
                "choice": "bullish",
                "probabilities": {"bullish": 0.6, "bearish": 0.1, "choppy": 0.2, "crisis": 0.1},
                "confidence": 0.8,
            },
            "trend_strength": {
                "score": 2.0,
                "probabilities": {0: 0.0, 1: 0.2, 2: 0.7, 3: 0.1},
                "confidence": 0.75,
                "legend": {0: "No trend", 1: "Weak", 2: "Strong", 3: "Explosive"},
            },
            "news_bullishness": {
                "score": 3.0,
                "probabilities": {0: 0.0, 1: 0.0, 2: 0.1, 3: 0.6, 4: 0.3},
                "confidence": 0.85,
                "legend": {0: "Very bearish", 1: "Bearish", 2: "Neutral", 3: "Bullish", 4: "Very bullish"},
            },
            "portfolio_stress": {
                "score": 0.5,
                "probabilities": {0: 0.7, 1: 0.2, 2: 0.1, 3: 0.0},
                "confidence": 0.9,
                "legend": {0: "Calm", 1: "Cautious", 2: "Stressed", 3: "Critical"},
            },
            "halt_new_buys": {
                "noul": 0.1,
                "confidence": 0.95,
            },
            "position_action": {
                "choice": "buy",
                "probabilities": {"buy": 0.6, "add": 0.2, "hold": 0.15, "trim": 0.03, "exit": 0.02},
                "confidence": 0.7,
            },
        }

    @pytest.fixture
    def base_cfg(self):
        """Base configuration for testing."""
        return {
            "risk_per_trade": 0.02,
            "stop_loss_pct": 0.03,
            "india_max_positions": 4,
            "us_max_positions": 4,
            "daily_loss_limit_pct": 0.05,
            "max_drawdown_pct": 0.08,
            "open_filter_min": 15,
            "jev_enabled": True,
            "jev_regime_enabled": True,
            "jev_trend_enabled": True,
            "jev_news_enabled": True,
            "jev_risk_enabled": True,
            "jev_action_enabled": True,
        }

    # ─── Test should_halt ────────────────────────────────────────────────────

    def test_halt_gate_below_threshold(self, mock_jev_decisions):
        """halt_new_buys.noul = 0.1 < 0.8 → should not halt."""
        assert jev.should_halt(mock_jev_decisions) is False

    def test_halt_gate_above_threshold(self, mock_jev_decisions):
        """halt_new_buys.noul = 0.9 > 0.8 → should halt."""
        decisions = mock_jev_decisions.copy()
        decisions["halt_new_buys"] = {"noul": 0.9, "confidence": 0.9}
        assert jev.should_halt(decisions) is True

    def test_halt_gate_disabled(self, mock_jev_decisions, base_cfg):
        """jev_risk_enabled = False → should not halt even if noul high."""
        saved = dict(jev._jev_flags)
        try:
            jev.configure({**base_cfg, "jev_risk_enabled": False})
            decisions = mock_jev_decisions.copy()
            decisions["halt_new_buys"] = {"noul": 0.9, "confidence": 0.9}
            assert jev.should_halt(decisions) is False
        finally:
            jev.configure(saved)

    # ─── Test get_position_action ────────────────────────────────────────────

    def test_position_action_above_threshold(self, mock_jev_decisions):
        """confidence 0.7 >= 0.65 → returns action."""
        action, conf = jev.get_position_action(mock_jev_decisions)
        assert action == "buy"
        assert conf == 0.7

    def test_position_action_below_threshold(self, mock_jev_decisions):
        """confidence 0.5 < 0.65 → returns None."""
        decisions = mock_jev_decisions.copy()
        decisions["position_action"]["confidence"] = 0.5
        action, conf = jev.get_position_action(decisions)
        assert action is None
        assert conf == 0.0

    def test_position_action_disabled(self, mock_jev_decisions, base_cfg):
        """jev_action_enabled = False → returns None."""
        saved = dict(jev._jev_flags)
        try:
            jev.configure({**base_cfg, "jev_action_enabled": False})
            action, conf = jev.get_position_action(mock_jev_decisions)
            assert action is None
            assert conf == 0.0
        finally:
            jev.configure(saved)

    # ─── Test get_regime_scaling ─────────────────────────────────────────────

    def test_regime_scaling_bullish(self, mock_jev_decisions):
        """Bullish regime → risk_mult=1.0, pos_delta=+2, sl_mult=1.0."""
        scaling = jev.get_regime_scaling(mock_jev_decisions)
        assert scaling["risk_mult"] == 1.0
        assert scaling["pos_delta"] == 2
        assert scaling["sl_mult"] == 1.0

    def test_regime_scaling_bearish(self, mock_jev_decisions):
        """Bearish regime → risk_mult=0.8, pos_delta=-2, sl_mult=1.2."""
        decisions = mock_jev_decisions.copy()
        decisions["regime"]["choice"] = "bearish"
        decisions["regime"]["confidence"] = 0.8
        scaling = jev.get_regime_scaling(decisions)
        assert scaling["risk_mult"] == 0.8
        assert scaling["pos_delta"] == -2
        assert scaling["sl_mult"] == 1.2

    def test_regime_scaling_choppy(self, mock_jev_decisions):
        """Choppy regime → risk_mult=0.7, pos_delta=-1, sl_mult=1.1."""
        decisions = mock_jev_decisions.copy()
        decisions["regime"]["choice"] = "choppy"
        decisions["regime"]["confidence"] = 0.8
        scaling = jev.get_regime_scaling(decisions)
        assert scaling["risk_mult"] == 0.7
        assert scaling["pos_delta"] == -1
        assert scaling["sl_mult"] == 1.1

    def test_regime_scaling_crisis(self, mock_jev_decisions):
        """Crisis regime → risk_mult=0.5, pos_delta=-4, sl_mult=1.5."""
        decisions = mock_jev_decisions.copy()
        decisions["regime"]["choice"] = "crisis"
        decisions["regime"]["confidence"] = 0.8
        scaling = jev.get_regime_scaling(decisions)
        assert scaling["risk_mult"] == 0.5
        assert scaling["pos_delta"] == -4
        assert scaling["sl_mult"] == 1.5

    def test_regime_scaling_low_confidence(self, mock_jev_decisions):
        """Low confidence → no scaling."""
        decisions = mock_jev_decisions.copy()
        decisions["regime"]["confidence"] = 0.5  # Below 0.65 threshold
        scaling = jev.get_regime_scaling(decisions)
        assert scaling == {"risk_mult": 1.0, "pos_delta": 0, "sl_mult": 1.0}

    def test_regime_scaling_disabled(self, mock_jev_decisions, base_cfg):
        """jev_regime_enabled = False → no scaling."""
        saved = dict(jev._jev_flags)
        try:
            jev.configure({**base_cfg, "jev_regime_enabled": False})
            scaling = jev.get_regime_scaling(mock_jev_decisions)
            assert scaling == {"risk_mult": 1.0, "pos_delta": 0, "sl_mult": 1.0}
        finally:
            jev.configure(saved)

    # ─── Test get_portfolio_stress_scaling ──────────────────────────────────

    def test_stress_scaling_calm(self, mock_jev_decisions):
        """Stress 0.5 → 1.0 (no reduction)."""
        assert jev.get_portfolio_stress_scaling(mock_jev_decisions) == 1.0

    def test_stress_scaling_cautious(self, mock_jev_decisions):
        """Stress 1.5 → 0.85."""
        decisions = mock_jev_decisions.copy()
        decisions["portfolio_stress"]["score"] = 1.5
        decisions["portfolio_stress"]["confidence"] = 0.8
        assert jev.get_portfolio_stress_scaling(decisions) == 0.85

    def test_stress_scaling_stressed(self, mock_jev_decisions):
        """Stress 2.5 → 0.7."""
        decisions = mock_jev_decisions.copy()
        decisions["portfolio_stress"]["score"] = 2.5
        decisions["portfolio_stress"]["confidence"] = 0.8
        assert jev.get_portfolio_stress_scaling(decisions) == 0.7

    def test_stress_scaling_critical(self, mock_jev_decisions):
        """Stress 3.5 → 0.5."""
        decisions = mock_jev_decisions.copy()
        decisions["portfolio_stress"]["score"] = 3.5
        decisions["portfolio_stress"]["confidence"] = 0.8
        assert jev.get_portfolio_stress_scaling(decisions) == 0.5

    def test_stress_scaling_low_confidence(self, mock_jev_decisions):
        """Low confidence → no reduction."""
        decisions = mock_jev_decisions.copy()
        decisions["portfolio_stress"]["confidence"] = 0.5  # Below 0.7 threshold
        assert jev.get_portfolio_stress_scaling(decisions) == 1.0

    # ─── Test get_effective_adx_min ──────────────────────────────────────────

    def test_adx_augmentation_no_trend(self, mock_jev_decisions):
        """Trend score 0 → 0.7x multiplier."""
        decisions = mock_jev_decisions.copy()
        decisions["trend_strength"]["score"] = 0.0
        decisions["trend_strength"]["confidence"] = 0.8
        effective = jev.get_effective_adx_min(20.0, decisions)
        assert abs(effective - 14.0) < 0.01  # 20 * 0.7

    def test_adx_augmentation_weak_trend(self, mock_jev_decisions):
        """Trend score 1.0 → 0.9x multiplier."""
        decisions = mock_jev_decisions.copy()
        decisions["trend_strength"]["score"] = 1.0
        decisions["trend_strength"]["confidence"] = 0.8
        effective = jev.get_effective_adx_min(20.0, decisions)
        assert abs(effective - 18.0) < 0.01  # 20 * 0.9

    def test_adx_augmentation_strong_trend(self, mock_jev_decisions):
        """Trend score 2.0 → 1.1x multiplier."""
        effective = jev.get_effective_adx_min(20.0, mock_jev_decisions)
        assert abs(effective - 22.0) < 0.01  # 20 * 1.1

    def test_adx_augmentation_explosive_trend(self, mock_jev_decisions):
        """Trend score 3.0 → 1.3x multiplier."""
        decisions = mock_jev_decisions.copy()
        decisions["trend_strength"]["score"] = 3.0
        decisions["trend_strength"]["confidence"] = 0.8
        effective = jev.get_effective_adx_min(20.0, decisions)
        assert abs(effective - 26.0) < 0.01  # 20 * 1.3

    def test_adx_augmentation_low_confidence(self, mock_jev_decisions):
        """Low confidence → base ADX."""
        decisions = mock_jev_decisions.copy()
        decisions["trend_strength"]["confidence"] = 0.5
        effective = jev.get_effective_adx_min(20.0, decisions)
        assert effective == 20.0

    # ─── Test get_trailing_dist_mult ─────────────────────────────────────────

    def test_trailing_calibration_no_trend(self, mock_jev_decisions):
        """Trend 0 → 2.0x (wider trail)."""
        decisions = mock_jev_decisions.copy()
        decisions["trend_strength"]["score"] = 0.0
        decisions["trend_strength"]["confidence"] = 0.8
        mult = jev.get_trailing_dist_mult(1.5, decisions)
        assert abs(mult - 3.0) < 0.01  # 1.5 * 2.0

    def test_trailing_calibration_strong_trend(self, mock_jev_decisions):
        """Trend 3.0 → 1.0x (tighter trail)."""
        decisions = mock_jev_decisions.copy()
        decisions["trend_strength"]["score"] = 3.0
        decisions["trend_strength"]["confidence"] = 0.8
        mult = jev.get_trailing_dist_mult(1.5, decisions)
        assert abs(mult - 1.5) < 0.01  # 1.5 * 1.0

    # ─── Test get_open_window_minutes ────────────────────────────────────────

    def test_open_window_crisis(self, mock_jev_decisions):
        """Crisis → 30 minutes."""
        decisions = mock_jev_decisions.copy()
        decisions["regime"]["choice"] = "crisis"
        decisions["regime"]["confidence"] = 0.8
        assert jev.get_open_window_minutes(decisions, 15) == 30

    def test_open_window_bullish(self, mock_jev_decisions):
        """Bullish → 5 minutes."""
        decisions = mock_jev_decisions.copy()
        decisions["regime"]["choice"] = "bullish"
        decisions["regime"]["confidence"] = 0.8
        assert jev.get_open_window_minutes(decisions, 15) == 5

    def test_open_window_bearish(self, mock_jev_decisions):
        """Bearish → 15 minutes (default)."""
        decisions = mock_jev_decisions.copy()
        decisions["regime"]["choice"] = "bearish"
        decisions["regime"]["confidence"] = 0.8
        assert jev.get_open_window_minutes(decisions, 15) == 15

    # ─── Test get_news_weight ────────────────────────────────────────────────

    def test_news_weight_high_confidence(self, mock_jev_decisions):
        """High confidence → 1.0 weight."""
        assert jev.get_news_weight(mock_jev_decisions) == 1.0

    def test_news_weight_low_confidence(self, mock_jev_decisions):
        """Low confidence → 0.33 weight."""
        decisions = mock_jev_decisions.copy()
        decisions["news_bullishness"]["confidence"] = 0.5
        assert jev.get_news_weight(decisions) == 0.33

    # ─── Test apply_jev_gates ────────────────────────────────────────────────

    def test_apply_jev_gates_bullish(self, mock_jev_decisions, base_cfg):
        """Bullish regime scales risk, positions, SL."""
        new_cfg = jev.apply_jev_gates(base_cfg, mock_jev_decisions)
        # Risk: 0.02 * 1.0 * 1.0 = 0.02
        assert new_cfg["risk_per_trade"] == 0.02
        # SL: 0.03 * 1.0 = 0.03
        assert new_cfg["stop_loss_pct"] == 0.03
        # Positions: 4 + 2 = 6
        assert new_cfg["india_max_positions"] == 6
        assert new_cfg["us_max_positions"] == 6

    def test_apply_jev_gates_crisis(self, mock_jev_decisions, base_cfg):
        """Crisis regime halves risk, increases SL, cuts positions."""
        decisions = mock_jev_decisions.copy()
        decisions["regime"]["choice"] = "crisis"
        decisions["regime"]["confidence"] = 0.8
        decisions["portfolio_stress"]["score"] = 3.5  # Critical stress
        decisions["portfolio_stress"]["confidence"] = 0.8
        
        new_cfg = jev.apply_jev_gates(base_cfg, decisions)
        # Risk: 0.02 * 0.5 * 0.5 = 0.005
        assert abs(new_cfg["risk_per_trade"] - 0.005) < 0.0001
        # SL: 0.03 * 1.5 = 0.045
        assert new_cfg["stop_loss_pct"] == 0.045
        # Positions: max(1, 4 - 4) = 1
        assert new_cfg["india_max_positions"] == 1
        assert new_cfg["us_max_positions"] == 1
        # Daily loss limit: 0.5%
        assert new_cfg["daily_loss_limit_pct"] == 0.005
        # Max drawdown: 2%
        assert new_cfg["max_drawdown_pct"] == 0.02
        # Open window: 30 min
        assert new_cfg["open_filter_min"] == 30

    def test_apply_jev_gates_does_not_mutate(self, mock_jev_decisions, base_cfg):
        """apply_jev_gates returns new dict, doesn't mutate original."""
        original_risk = base_cfg["risk_per_trade"]
        original_sl = base_cfg["stop_loss_pct"]
        original_pos = base_cfg["india_max_positions"]
        
        new_cfg = jev.apply_jev_gates(base_cfg, mock_jev_decisions)
        
        assert base_cfg["risk_per_trade"] == original_risk
        assert base_cfg["stop_loss_pct"] == original_sl
        assert base_cfg["india_max_positions"] == original_pos
        assert new_cfg is not base_cfg

    # ─── Test extract_jev_features ──────────────────────────────────────────

    def test_extract_jev_features(self, mock_jev_decisions):
        """Extract 8 normalized features for RL observation."""
        features = jev.extract_jev_features(mock_jev_decisions)
        assert len(features) == 8
        # All in [0, 1]
        for f in features:
            assert 0.0 <= f <= 1.0
        # Regime probs sum to ~1
        regime_sum = sum(features[:4])
        assert abs(regime_sum - 1.0) < 0.01

    # ─── Test config field reading ──────────────────────────────────────────

    def test_config_fields_exist(self):
        """Verify all JEV config fields are read from env with defaults."""
        # This tests that the module reads env vars correctly
        assert hasattr(jev, "REGIME_CONF_THRESHOLD")
        assert hasattr(jev, "ACTION_CONF_THRESHOLD")
        assert hasattr(jev, "TREND_CONF_THRESHOLD")
        assert hasattr(jev, "NEWS_CONF_THRESHOLD")
        assert hasattr(jev, "HALT_NOUl_THRESHOLD")
        assert hasattr(jev, "CACHE_TTL_SECONDS")
        assert hasattr(jev, "MAX_CALLS_PER_CYCLE")
        assert hasattr(jev, "JEV_FEATURE_COLUMNS")
        assert len(jev.JEV_FEATURE_COLUMNS) == 8

    # ─── Test JEV question schema ────────────────────────────────────────────

    def test_jev_questions_schema(self):
        """Verify all 6 questions defined with correct types."""
        questions = jev.JEV_QUESTIONS
        assert len(questions) == 6
        assert "regime" in questions
        assert "trend_strength" in questions
        assert "news_bullishness" in questions
        assert "portfolio_stress" in questions
        assert "halt_new_buys" in questions
        assert "position_action" in questions
        
        # Check types
        assert questions["regime"]["type"] == "choice"
        assert questions["trend_strength"]["type"] == "score"
        assert questions["news_bullishness"]["type"] == "score"
        assert questions["portfolio_stress"]["type"] == "score"
        assert questions["halt_new_buys"]["type"] == "noul"
        assert questions["position_action"]["type"] == "choice"
        
        # Check criteria
        assert len(questions["regime"]["criteria"]) == 4
        assert len(questions["trend_strength"]["criteria"]) == 4
        assert len(questions["news_bullishness"]["criteria"]) == 5
        assert len(questions["portfolio_stress"]["criteria"]) == 4
        assert len(questions["position_action"]["criteria"]) == 5


class TestJEVStateBuilder:
    """Test state builder functions."""

    def test_build_market_state(self):
        """Build market state JSON for symbol."""
        indicators = {
            "price": 175.0,
            "rsi": 42.0,
            "macd_hist": 0.34,
            "bb_pos": 0.6,
            "ema_9_21": "bull",
            "adx": 28.0,
            "atr": 2.1,
            "vol_ratio": 1.3,
        }
        news = [
            {"title": "Apple beats earnings", "sentiment": 0.8, "recency_h": 4},
            {"title": "New iPhone launch", "sentiment": 0.6, "recency_h": 12},
        ]
        portfolio = {
            "cash": 18000,
            "drawdown_pct": 0.03,
            "open_positions": 3,
            "daily_pnl_pct": 0.005,
        }
        market = {
            "spy_trend_pct": 0.4,
            "vix": 14.2,
            "breadth": 0.62,
        }
        
        state_json = jev.build_market_state("AAPL", indicators, news, portfolio, market)
        state = json.loads(state_json)
        
        assert state["symbol"] == "AAPL"
        assert state["price"] == 175.0
        assert state["indicators"]["rsi"] == 42.0
        assert len(state["news"]) == 2
        assert state["portfolio"]["cash"] == 18000
        assert state["market"]["vix"] == 14.2

    def test_build_system_state(self):
        """Build system state JSON for market-level decisions."""
        portfolio = {"cash": 18000, "drawdown_pct": 0.03, "open_positions": 3, "daily_pnl_pct": 0.005}
        market = {"spy_trend_pct": 0.4, "vix": 14.2, "breadth": 0.62}
        
        state_json = jev.build_system_state("us", portfolio, market)
        state = json.loads(state_json)
        
        assert state["market"] == "us"
        assert state["portfolio"]["cash"] == 18000
        assert state["market_context"]["vix"] == 14.2


# Run with: pytest tests/test_jev_gates.py -v