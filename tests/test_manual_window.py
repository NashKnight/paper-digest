from argparse import Namespace
from datetime import UTC, datetime, timedelta
from unittest import TestCase
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from paper_digest.arxiv_client import ArxivClientError, fetch_latest_papers
from paper_digest.config import FeedConfig
from tests.test_arxiv_client import FIXTURE_PATH
from tools.run_omni_daily import parse_window


class ManualWindowTests(TestCase):
    def test_defaults_and_local_time_and_validation(self):
        now = datetime(2026, 9, 23, 7, tzinfo=UTC)
        previous = now - timedelta(days=2)
        self.assertEqual(parse_window(Namespace(since=None, until=None), 'Asia/Shanghai', previous, now), (previous, now))
        start, end = parse_window(Namespace(since='2026-09-21T14:00', until='2026-09-22T11:00'), 'Asia/Shanghai', previous, now)
        self.assertEqual(start.hour, 14)
        self.assertEqual(start.utcoffset(), timedelta(hours=8))
        self.assertEqual(end.day, 22)
        for args in (Namespace(since='2026-09-24', until=None), Namespace(since=None, until='2026-09-24')):
            with self.assertRaises(ValueError):
                parse_window(args, 'Asia/Shanghai', previous, now)

    @patch('paper_digest.arxiv_client.fetch_bytes_with_retry')
    def test_paginates_and_sends_utc_range(self, fetch):
        page = FIXTURE_PATH.read_bytes()
        fetch.side_effect = [page, b'<feed xmlns="http://www.w3.org/2005/Atom"/>']
        feed = FeedConfig(name='test', categories=['cs.AI'], max_results=1)
        papers = fetch_latest_papers(feed, window_start=datetime(2026,4,1,tzinfo=UTC),
                                    window_end=datetime(2026,4,9,tzinfo=UTC), request_delay_seconds=0)
        self.assertTrue(papers)
        self.assertEqual(fetch.call_count, 2)
        query = parse_qs(urlsplit(fetch.call_args_list[0].args[0].full_url).query)
        self.assertIn('submittedDate:[202604010000 TO 202604090000]', query['search_query'][0])
        query = parse_qs(urlsplit(fetch.call_args_list[1].args[0].full_url).query)
        self.assertEqual(int(query['start'][0]), len(papers))

    @patch('paper_digest.arxiv_client.fetch_latest_papers_from_rss')
    @patch('paper_digest.arxiv_client.fetch_bytes_with_retry', side_effect=ArxivClientError('offline'))
    def test_range_failure_does_not_silently_use_incomplete_rss(self, fetch, rss):
        with self.assertRaises(ArxivClientError):
            fetch_latest_papers(FeedConfig(name='test', categories=['cs.AI']),
                                window_start=datetime(2026,4,1,tzinfo=UTC),
                                window_end=datetime(2026,4,9,tzinfo=UTC), request_delay_seconds=0)
        rss.assert_not_called()
