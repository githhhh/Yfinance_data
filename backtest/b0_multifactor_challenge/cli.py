from __future__ import annotations
import argparse
from pathlib import Path
from .panel import build_panel
from .evaluate import run
from .agent import run_official_rdagent, import_factor_result, replay_factor_code
from .diagnostics import run_diagnostics

def main():
    ap=argparse.ArgumentParser(description='B0 multifactor champion challenge')
    sp=ap.add_subparsers(dest='cmd',required=True)
    sp.add_parser('prepare')
    e=sp.add_parser('evaluate'); e.add_argument('--feature-mode',choices=['f0','f1','agent'],default='f1')
    d=sp.add_parser('diagnostics'); d.add_argument('--feature-mode',choices=['f0','f1','agent'],default='f1')
    a=sp.add_parser('rdagent'); a.add_argument('--steps',type=int,default=2)
    i=sp.add_parser('import-factor'); i.add_argument('--name',required=True); i.add_argument('--result-h5',required=True)
    r=sp.add_parser('replay-factor'); r.add_argument('--name',required=True); r.add_argument('--factor-py',required=True)
    args=ap.parse_args()
    if args.cmd=='prepare': print(build_panel())
    elif args.cmd=='evaluate': run(feature_mode=args.feature_mode)
    elif args.cmd=='diagnostics': run_diagnostics(feature_mode=args.feature_mode)
    elif args.cmd=='rdagent': raise SystemExit(run_official_rdagent(args.steps))
    elif args.cmd=='import-factor': print(import_factor_result(args.name,Path(args.result_h5)))
    elif args.cmd=='replay-factor': print(replay_factor_code(args.name,Path(args.factor_py)))
if __name__=='__main__': main()
