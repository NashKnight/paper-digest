from dataclasses import replace
from datetime import timedelta
import json
import unittest

from paper_digest.config import FeishuWebhookConfig
from paper_digest.delivery import build_notification_messages
from paper_digest.feishu_delivery import _build_card_payload
from tests.test_delivery_additional import build_digest


class CompactCardTests(unittest.TestCase):
    def test_window_survives_delivery_filtering_and_card_is_interactive(self):
        digest = build_digest()
        digest.window_start = digest.generated_at - timedelta(days=3, seconds=11)
        delivery = FeishuWebhookConfig(
            webhook_url='https://example.invalid', title_prefix='Omni 论文日报',
            skip_if_empty=False, compact_card=True,
            include_focus=False, include_actions=False,
        )
        message = build_notification_messages(delivery, digest)[0]
        self.assertIn(digest.window_start.strftime('%m/%d %H:%M'), message.body)
        self.assertNotIn('最近 24', message.body)
        payload = _build_card_payload(message.title, message.body)
        self.assertEqual(payload['msg_type'], 'interactive')
        self.assertLess(len(json.dumps(payload).encode()), 20000)
        empty = replace(digest, feeds=[])
        message = build_notification_messages(delivery, empty)[0]
        self.assertIn('暂无新增', message.body)
