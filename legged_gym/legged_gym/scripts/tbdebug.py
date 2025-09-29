import os
import glob
import json
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def merge_events_from_file(acc, out):
    acc.Reload()
    for tag in acc.Tags().get('scalars', []):
        events = acc.Scalars(tag)
        if tag not in out:
            out[tag] = []
        out[tag].extend([{"wall_time": float(e.wall_time), "step": int(e.step), "value": float(e.value)} for e in events])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", "-l", required=True, help="TensorBoard log directory (contains events.* files)")
    p.add_argument("--out", "-o", required=True, help="Output JSON file path")
    p.add_argument("--tags", "-t", default=None, help="Comma separated list of tags to export (default: all)")
    args = p.parse_args()

    event_files = glob.glob(os.path.join(args.logdir, "events.*"))
    if not event_files:
        raise SystemExit("No event files found in " + args.logdir)

    data = {}
    for f in sorted(event_files):
        acc = EventAccumulator(f, size_guidance={ "scalars": 0 })
        merge_events_from_file(acc, data)

    # filter tags if requested
    if args.tags:
        wanted = set([t.strip() for t in args.tags.split(",")])
        data = {k:v for k,v in data.items() if k in wanted}

    # sort entries per tag by step (or wall_time)
    for k in data:
        data[k].sort(key=lambda x: (x["step"], x["wall_time"]))

    # write json
    with open(args.out, "w") as fh:
        json.dump(data, fh, indent=2)

if __name__ == "__main__":
    main()