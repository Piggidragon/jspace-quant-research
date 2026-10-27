"""Command line: python -m jspace_quant {capture,analyze,plot,all} [...]

    capture [--model M ...]   capture activations (GPU, starts jlens-server itself);
                              default: all models not captured yet
    analyze [--force]         readouts + metrics (CPU) -> results/tables/; computes only arms
                              without a result, --force recomputes all
    intervene [--model M ...] causal swap/ablation test with lens directions (GPU);
                              default: all models without results yet
    plot [--lang en de]       figures -> results/figures/<lang>/
    all                       everything in sequence

JSPACE_SMOKE=1 writes to results/smoke/ and uses only a few items (pipeline test).
"""

from __future__ import annotations

import argparse
import json
import logging

from jspace_quant import config


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="jspace_quant", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    a = sub.add_parser("all")
    for sp in (c, a):
        sp.add_argument("--model", nargs="+", default=["missing"], choices=[*config.MODELS, "all", "missing"],
                        help="models; 'missing' = all not captured yet (default), 'all' = recapture all")
        sp.add_argument("--gpus", default=",".join(config.SERVER_GPUS),
                        help="GPUs for parallel captures, one model per GPU (default: %(default)s)")
    iv = sub.add_parser("intervene")
    iv.add_argument("--model", nargs="+", default=["missing"], choices=[*config.MODELS, "all", "missing"])
    iv.add_argument("--gpus", default=",".join(config.SERVER_GPUS))
    an = sub.add_parser("analyze")
    pl = sub.add_parser("plot")
    for sp in (an, a):
        sp.add_argument("--force", action="store_true", help="recompute all arms")
    for sp in (pl, a):
        sp.add_argument("--lang", nargs="+", default=["en", "de"], choices=["en", "de"])
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger(__name__)
    if config.SMOKE:
        log.warning("SMOKE mode: writing to %s", config.RESULTS_DIR)

    if args.cmd in ("capture", "all"):
        from jspace_quant import capture

        if "all" in args.model:
            models = list(config.MODELS)
        elif "missing" in args.model:
            models = [m for m in config.MODELS if not capture.is_captured(m)]
        else:
            models = args.model
        missing_files = [m for m in models if not config.MODELS[m].exists()]
        if missing_files:
            log.warning("model file missing, skipped: %s (see scripts/make_quant_variants.sh)", missing_files)
            models = [m for m in models if m not in missing_files]
        log.info("capturing: %s", models or "nothing to do")
        if models:
            capture.run_parallel(models, args.gpus.split(","))
    if args.cmd == "intervene":
        from jspace_quant import capture, intervene

        if "all" in args.model:
            models = list(config.MODELS)
        elif "missing" in args.model:
            models = [m for m in config.MODELS if not intervene.is_done(m)]
        else:
            models = args.model
        models = [m for m in models if config.MODELS[m].exists()]
        log.info("intervening: %s", models or "nothing to do")
        if models:
            capture.run_parallel(models, args.gpus.split(","), fn=intervene.run)
    if args.cmd in ("analyze", "all"):
        from jspace_quant import analysis

        for split in analysis.SPLITS:
            analysis.compute(split, force=args.force)
        print(json.dumps(analysis.summarize(), indent=2))
    if args.cmd in ("plot", "all"):
        from jspace_quant import plots

        plots.all_figures(tuple(args.lang))


if __name__ == "__main__":
    main()
