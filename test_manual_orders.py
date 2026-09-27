"""Offline checks for the private manual order path; all exchange calls mocked."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import manual_orders as mo
import trading_common as tc


class ManualOrderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        folder = Path(self.temp.name)
        self.patches = [patch.object(mo, "PREVIEWS", folder / "previews.json"),
                        patch.object(tc, "TRADE_LOCK_PATH", str(folder / "trade.lock")),
                        patch.object(tc, "TRADE_LOG_PATH", str(folder / "trades.jsonl")),
                        patch.object(tc, "KILL_SWITCH_FILE", str(folder / "STOP_TRADING"))]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.order = dict(venue="kalshi", env="prod", market="TEST-123", outcome="yes",
                          side="buy", qty="2", limit_price="0.55")
        self.quote = dict(status="active", best_bid=.52, best_ask=.54)

    def test_rejects_invalid_numbers_before_quote(self):
        with patch.object(mo, "_quote") as quote:
            for value in ("0", "-1", "nan", "inf", "1e309", "garbage"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    mo.preview({**self.order, "qty": value}, "owner")
        quote.assert_not_called()

    def test_preview_is_owner_bound_single_use_and_real_send_is_mocked(self):
        trader = Mock(create_order=Mock(return_value={"order_id": "exchange-1", "fill_count": "1"}))
        with patch.object(mo, "_quote", return_value=self.quote), patch.object(mo, "_trader", return_value=trader), \
             patch.dict(os.environ, {"MARKETPULSE_ALLOW_LIVE": "1"}):
            preview = mo.preview(self.order, "owner-a")
            with self.assertRaises(ValueError):
                mo.submit(preview["preview_id"], "owner-b")
            # A mismatched owner cannot consume another account's preview.
            result = mo.submit(preview["preview_id"], "owner-a")
            self.assertEqual(result["status"], "submitted")
            self.assertEqual(result["order_id"], "exchange-1")
            self.assertEqual(trader.create_order.call_count, 1)
            with self.assertRaises(ValueError):
                mo.submit(preview["preview_id"], "owner-a")

    def test_local_opt_in_blocks_send(self):
        with patch.object(mo, "_quote", return_value=self.quote), \
             patch.dict(os.environ, {"MARKETPULSE_ALLOW_LIVE": ""}):
            preview = mo.preview(self.order, "owner")
            with self.assertRaisesRegex(RuntimeError, "local|Local"):
                mo.submit(preview["preview_id"], "owner")

    def test_sell_requires_verified_position(self):
        trader = Mock(positions=Mock(return_value={"market_positions": [
            {"ticker": "TEST-123", "position_fp": "1.00"}]}))
        with patch.object(mo, "_quote", return_value=self.quote), patch.object(mo, "_trader", return_value=trader):
            with self.assertRaisesRegex(ValueError, "exceeds the verified position"):
                mo.preview({**self.order, "side": "sell", "qty": "2", "limit_price": "0.52"}, "owner")

    def test_unknown_exchange_result_stays_reserved_and_is_not_retried(self):
        trader = Mock(create_order=Mock(side_effect=TimeoutError("no receipt")))
        with patch.object(mo, "_quote", return_value=self.quote), patch.object(mo, "_trader", return_value=trader), \
             patch.dict(os.environ, {"MARKETPULSE_ALLOW_LIVE": "1"}):
            preview = mo.preview(self.order, "owner")
            with self.assertRaisesRegex(RuntimeError, "outcome is unknown"):
                mo.submit(preview["preview_id"], "owner")
            self.assertEqual(trader.create_order.call_count, 1)
            self.assertGreater(tc.spent_today(), 0)
            rows = [json.loads(line) for line in Path(tc.TRADE_LOG_PATH).read_text().splitlines()]
            self.assertEqual([r["status"] for r in rows], ["pending", "unknown"])
            with self.assertRaises(ValueError):
                mo.submit(preview["preview_id"], "owner")


if __name__ == "__main__":
    unittest.main()
