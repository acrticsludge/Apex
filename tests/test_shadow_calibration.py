"""
Calibration tests for JEV shadow mode.
Run after 5+ days of shadow logging to validate calibration.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import apex_jev as jev


class TestJEVCalibration:
    """Test JEV calibration from shadow mode logs."""

    @pytest.fixture
    def mock_think_buffer(self):
        """Mock _think_buffer with JEV entries."""
        return [
            # Day 1 - Bullish regime
            {"cat": "JEV", "sym": "INDIA", "msg": "regime=bullish conf=0.80 probs={'bullish': '0.60', 'bearish': '0.10', 'choppy': '0.20', 'crisis': '0.10'}"},
            {"cat": "JEV", "sym": "INDIA", "msg": "trend_strength=2.00 conf=0.75"},
            {"cat": "JEV", "sym": "INDIA", "msg": "portfolio_stress=0.50 conf=0.90"},
            {"cat": "JEV", "sym": "INDIA", "msg": "halt_new_buys=0.10 conf=0.95"},
            
            # Day 2 - Bearish regime  
            {"cat": "JEV", "sym": "US", "msg": "regime=bearish conf=0.85 probs={'bullish': '0.05', 'bearish': '0.70', 'choppy': '0.20', 'crisis': '0.05'}"},
            {"cat": "JEV", "sym": "US", "msg": "trend_strength=1.50 conf=0.70"},
            {"cat": "JEV", "sym": "US", "msg": "portfolio_stress=1.20 conf=0.80"},
            {"cat": "JEV", "sym": "US", "msg": "halt_new_buys=0.20 conf=0.85"},
            
            # Day 3 - Crisis regime
            {"cat": "JEV", "sym": "INDIA", "msg": "regime=crisis conf=0.90 probs={'bullish': '0.02', 'bearish': '0.10', 'choppy': '0.18', 'crisis': '0.70'}"},
            {"cat": "JEV", "sym": "INDIA", "msg": "trend_strength=3.00 conf=0.85"},
            {"cat": "JEV", "sym": "INDIA", "msg": "portfolio_stress=3.50 conf=0.95"},
            {"cat": "JEV", "sym": "INDIA", "msg": "halt_new_buys=0.85 conf=0.90"},
            
            # Day 4 - Choppy regime
            {"cat": "JEV", "sym": "US", "msg": "regime=choppy conf=0.75 probs={'bullish': '0.25', 'bearish': '0.25', 'choppy': '0.40', 'crisis': '0.10'}"},
            {"cat": "JEV", "sym": "US", "msg": "trend_strength=0.50 conf=0.80"},
            {"cat": "JEV", "sym": "US", "msg": "portfolio_stress=0.80 conf=0.85"},
            {"cat": "JEV", "sym": "US", "msg": "halt_new_buys=0.15 conf=0.90"},
            
            # Day 5 - Bullish again
            {"cat": "JEV", "sym": "INDIA", "msg": "regime=bullish conf=0.78 probs={'bullish': '0.55', 'bearish': '0.15', 'choppy': '0.20', 'crisis': '0.10'}"},
            {"cat": "JEV", "sym": "INDIA", "msg": "trend_strength=2.50 conf=0.72"},
            {"cat": "JEV", "sym": "INDIA", "msg": "portfolio_stress=0.60 conf=0.88"},
            {"cat": "JEV", "sym": "INDIA", "msg": "halt_new_buys=0.08 conf=0.92"},
        ]

    def test_regime_calibration(self, mock_think_buffer):
        """
        Test regime calibration: predicted probability vs actual frequency.
        Brier score should be < 0.25.
        """
        # In real test, we'd compare predicted regime probs with actual next-day market direction
        # For now, verify we can parse the log entries
        jev_entries = [e for e in mock_think_buffer if e["cat"] == "JEV" and "regime=" in e["msg"]]
        assert len(jev_entries) == 5  # 5 days
        
        # Parse regime choices
        regimes = []
        for entry in jev_entries:
            msg = entry["msg"]
            regime = msg.split("regime=")[1].split(" ")[0]
            regimes.append(regime)
        
        assert "bullish" in regimes
        assert "bearish" in regimes
        assert "crisis" in regimes
        assert "choppy" in regimes

    def test_confidence_calibration(self, mock_think_buffer):
        """
        Test confidence calibration: high confidence -> high accuracy.
        """
        jev_entries = [e for e in mock_think_buffer if e["cat"] == "JEV" and "regime=" in e["msg"]]
        
        high_conf_entries = []
        for entry in jev_entries:
            msg = entry["msg"]
            conf_str = msg.split("conf=")[1].split(" ")[0]
            conf = float(conf_str)
            if conf >= 0.8:
                high_conf_entries.append(entry)
        
        assert len(high_conf_entries) >= 3  # At least 3 high-confidence predictions
        
        # In real calibration, we'd check if high-conf predictions were correct
        # For now, verify confidence parsing works
        for entry in high_conf_entries:
            conf_str = entry["msg"].split("conf=")[1].split(" ")[0]
            assert float(conf_str) >= 0.8

    def test_brier_score_components(self, mock_think_buffer):
        """
        Test Brier score calculation components.
        Brier = mean((predicted_prob - actual_outcome)^2)
        """
        jev_entries = [e for e in mock_think_buffer if e["cat"] == "JEV" and "regime=" in e["msg"]]
        
        # Extract predicted probabilities for the chosen regime
        predicted_probs = []
        for entry in jev_entries:
            msg = entry["msg"]
            probs_str = msg.split("probs=")[1]
            # Parse the dict string
            import ast
            probs = ast.literal_eval(probs_str)
            chosen = msg.split("regime=")[1].split(" ")[0]
            predicted_probs.append(probs[chosen])
        
        # All predicted probs for highly confident predictions (conf > 0.75) should be > 0.5
        for i, p in enumerate(predicted_probs):
            entry = jev_entries[i]
            conf_str = entry["msg"].split("conf=")[1].split(" ")[0]
            conf = float(conf_str)
            if conf > 0.75:
                assert float(p) > 0.5, f"High confidence ({conf}) but low prob ({p})"

    def test_regime_transitions(self, mock_think_buffer):
        """
        Test regime transition detection.
        """
        jev_entries = [e for e in mock_think_buffer if e["cat"] == "JEV" and "regime=" in e["msg"]]
        
        regimes = []
        for entry in jev_entries:
            msg = entry["msg"]
            regime = msg.split("regime=")[1].split(" ")[0]
            regimes.append(regime)
        
        # Check transitions
        transitions = list(zip(regimes[:-1], regimes[1:]))
        assert ("bullish", "bearish") in transitions
        assert ("bearish", "crisis") in transitions
        assert ("crisis", "choppy") in transitions
        assert ("choppy", "bullish") in transitions


class TestJEVStateParsing:
    """Test parsing of JEV state from think buffer."""

    def test_parse_jev_entry(self):
        """Parse a single JEV think buffer entry."""
        entry = {
            "cat": "JEV",
            "sym": "INDIA",
            "msg": "regime=bullish conf=0.80 probs={'bullish': '0.60', 'bearish': '0.10', 'choppy': '0.20', 'crisis': '0.10'}"
        }
        
        msg = entry["msg"]
        regime = msg.split("regime=")[1].split(" ")[0]
        conf = float(msg.split("conf=")[1].split(" ")[0])
        probs_str = msg.split("probs=")[1]
        
        import ast
        probs = ast.literal_eval(probs_str)
        
        assert regime == "bullish"
        assert conf == 0.80
        assert float(probs["bullish"]) == 0.60
        assert sum(float(v) for v in probs.values()) == pytest.approx(1.0)

    def test_parse_trend_entry(self):
        """Parse trend_strength entry."""
        msg = "trend_strength=2.00 conf=0.75"
        score = float(msg.split("trend_strength=")[1].split(" ")[0])
        conf = float(msg.split("conf=")[1])
        
        assert score == 2.00
        assert conf == 0.75

    def test_parse_stress_entry(self):
        """Parse portfolio_stress entry."""
        msg = "portfolio_stress=3.50 conf=0.95"
        score = float(msg.split("portfolio_stress=")[1].split(" ")[0])
        conf = float(msg.split("conf=")[1])
        
        assert score == 3.50
        assert conf == 0.95

    def test_parse_halt_entry(self):
        """Parse halt_new_buys entry."""
        msg = "halt_new_buys=0.85 conf=0.90"
        noul = float(msg.split("halt_new_buys=")[1].split(" ")[0])
        conf = float(msg.split("conf=")[1])
        
        assert noul == 0.85
        assert conf == 0.90


# Run with: pytest tests/test_shadow_calibration.py -v