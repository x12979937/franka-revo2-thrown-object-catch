#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys
from video2sim_schema import schema_smoke_test, validate_episode_dir

DEFAULT_REFS = [
    "/root/autodl-tmp/mingyu/video2sim/smoke_remote2_demo1_demo2_20260720_110942/demo1_episode_000212/rolling_tabletop",
    "/root/autodl-tmp/mingyu/video2sim/smoke_remote2_demo1_demo2_20260720_110942/demo2_episode_000197/falling_baton",
]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--episode', action='append', default=[])
    ap.add_argument('--strict-videos', action='store_true')
    args=ap.parse_args()
    episodes=args.episode or DEFAULT_REFS
    reports=[]
    for ep in episodes:
        p=Path(ep)
        if not p.exists():
            reports.append({'episode_dir': ep, 'validation_pass': False, 'failure_reasons': ['missing episode dir']})
            continue
        reports.append(validate_episode_dir(p, task_name=p.name, physics_engine='schema_reference', strict_videos=args.strict_videos, write_validation=False))
    print(json.dumps({'reports': reports}, ensure_ascii=False, indent=2))
    return 0 if all(r.get('validation_pass') for r in reports) else 1

if __name__ == '__main__':
    sys.exit(main())
