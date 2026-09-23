#!/usr/bin/env python3
"""Local daily entry point. Keep the fetch checkpoint until delivery succeeds."""
from __future__ import annotations

import argparse
import fcntl
import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_digest.archive_site import build_archive_site
from paper_digest.config import load_config
from paper_digest.delivery import send_configured_deliveries
from paper_digest.digest import write_outputs
from paper_digest.feedback import load_feedback, save_feedback
from paper_digest.feishu_delivery import send_feishu_message
from paper_digest.service import generate_digest
from paper_digest.state import DigestState, load_state, save_state


def parse_window(args, timezone: str, checkpoint: datetime | None, now: datetime):
    tz = ZoneInfo(timezone)
    def parse(value):
        result = datetime.fromisoformat(value)
        return result.replace(tzinfo=tz) if result.tzinfo is None else result
    start = parse(args.since) if args.since else checkpoint
    end = parse(args.until) if args.until else now
    if start is None:
        raise ValueError("No previous checkpoint; provide --since")
    if start >= end:
        raise ValueError("Start must be earlier than end")
    if end > now:
        raise ValueError("End must not be in the future")
    return start, end


def main() -> int:
    parser = argparse.ArgumentParser(description="Omni 论文抓取并发送飞书；默认从上次成功拉取至今")
    parser.add_argument('--since', help='开始时间，如 2026-09-21T14:10:01；默认北京时间')
    parser.add_argument('--until', help='结束时间；默认当前时刻')
    parser.add_argument('--test-notification', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    config = load_config(ROOT / 'config.omni.toml')
    if args.test_notification:
        send_feishu_message(
            config.deliveries[0],
            title='Omni 论文日报 · 已开启',
            body='**每天 11:00 · 北京时间**\n\n'
                 '追踪全双工语音交互与多模态相关论文。\n'
                 '范围：上次成功拉取时刻 → 本次拉取时刻。\n\n'
                 '日报展示论文链接与简短要点，最多 8 篇；完整结果保存在本地。\n'
                 '暂无新论文时，也会发送简短通知。',
        )
        print('Feishu test card accepted', flush=True)
        return 0
    config.state.path.parent.mkdir(parents=True, exist_ok=True)
    with (config.state.path.parent / 'daily.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another daily run is active; skipping.', flush=True)
            return 0
        print(f'Start: {datetime.now(ZoneInfo(config.timezone)).isoformat()}', flush=True)
        if config.analysis and not os.getenv(config.analysis.api_key_env):
            print('Analysis key unavailable; sending paper links and topic metadata.', flush=True)
            config = replace(config, analysis=None)
        state = load_state(config.state)
        custom_range = bool(args.since or args.until)
        now = datetime.now(ZoneInfo(config.timezone))
        try:
            start, end = parse_window(args, config.timezone, state.last_successful_fetch_at, now)
        except ValueError as exc:
            parser.error(str(exc))
        if custom_range:
            # Historical queries return all matches and never move the daily checkpoint.
            state = DigestState(seen_papers={}, last_successful_fetch_at=start)
            stamp = now.strftime('%Y%m%dT%H%M%S%f')
            config = replace(config, output_dir=config.output_dir / 'manual' / stamp)
        print(f'Window: ({start.isoformat()}, {end.isoformat()}]; custom={custom_range}', flush=True)
        feedback = load_feedback(config.feedback)
        digest = generate_digest(config, now=end, state=state, feedback_state=feedback)
        write_outputs(config, digest)
        build_archive_site(config.output_dir, feedback_state=feedback, digest_state=state)
        receipts = send_configured_deliveries(config, digest)
        if not custom_range:
            save_feedback(config.feedback, feedback)
            save_state(config.state, state)
        print(f'Completed: {len(receipts)} delivery; papers={sum(len(f.papers) for f in digest.feeds)}; checkpoint_updated={not custom_range}; output={config.output_dir}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
