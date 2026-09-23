from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from paper_digest.arxiv_client import Paper
from paper_digest.config import AppConfig, FeedConfig, StateConfig
from paper_digest.digest import _window_label
from paper_digest.feedback import FeedbackState
from paper_digest.service import generate_digest
from paper_digest.state import DigestState, load_state, save_state


class IncrementalWindowTests(unittest.TestCase):
    def test_exact_boundaries_long_and_short_gaps_and_empty_run(self):
        for gap in (timedelta(days=3, seconds=17), timedelta(minutes=15)):
            with self.subTest(gap=gap), TemporaryDirectory() as tmp:
                config = AppConfig(
                    timezone="Asia/Shanghai", lookback_hours=24,
                    output_dir=Path(tmp), request_delay_seconds=0,
                    feeds=[FeedConfig(name="Omni Duplex Core", categories=["cs.AI"],
                                      keywords=["full-duplex"])],
                    state=StateConfig(enabled=True, path=Path(tmp) / "state.json",
                                      retention_days=90),
                    since_last_run=True,
                )
                config = replace(config, analysis=None, output_dir=Path(tmp),
                                 feeds=config.feeds[:1],
                                 feedback=replace(config.feedback, path=Path(tmp) / "feedback.json"),
                                 state=replace(config.state, path=Path(tmp) / 'state.json'))
                start = datetime(2026, 9, 21, 6, 10, 1, 946245, tzinfo=UTC)
                end = start + gap
                state = DigestState(seen_papers={}, last_successful_fetch_at=start)
                papers = []
                for index, timestamp in enumerate((start - timedelta(microseconds=1),
                                                   start, start + timedelta(microseconds=1),
                                                   end, end + timedelta(microseconds=1))):
                    papers.append(Paper(title='Full-duplex agent', summary='full-duplex',
                                        authors=[], categories=['cs.AI'], paper_id=f'2609.{index:05}',
                                        abstract_url=f'https://arxiv.org/abs/2609.{index:05}',
                                        pdf_url=None, published_at=timestamp, updated_at=timestamp))
                with patch('paper_digest.service.fetch_feed_papers', return_value=papers):
                    digest = generate_digest(config, now=end, state=state,
                                             feedback_state=FeedbackState(papers={}))
                self.assertEqual({p.published_at for p in digest.feeds[0].papers},
                                 {start + timedelta(microseconds=1), end})
                self.assertEqual(digest.window_start, start)
                self.assertIn(start.astimezone(digest.generated_at.tzinfo).isoformat(),
                              _window_label(digest, chinese=True))
                save_state(config.state, state)
                self.assertEqual(load_state(config.state).last_successful_fetch_at, end)
                with patch('paper_digest.service.fetch_feed_papers', return_value=[]):
                    generate_digest(config, now=end + gap)
                self.assertEqual(load_state(config.state).last_successful_fetch_at, end + gap)
                with patch('paper_digest.service.fetch_feed_papers', side_effect=RuntimeError('failed')):
                    with self.assertRaises(RuntimeError):
                        generate_digest(config, now=end + gap * 2)
                self.assertEqual(load_state(config.state).last_successful_fetch_at, end + gap)
