"""The ``reef`` command line.

    reef serve                                    # $REEF_MODEL or seraphimserapis/reef from the Hub
    reef decide  seraphimserapis/reef request.json
    reef train   --out runs/reef-tiny            # needs the [train] extra
    reef export  runs/reef-tiny artifacts/reef-tiny
    reef eval    artifacts/reef-tiny --device edgetpu
    reef bench   artifacts/reef-tiny
    reef push    artifacts/reef-tiny yourname/reef-tiny
    reef setup-edgetpu && reef doctor
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_MODEL = "seraphimserapis/reef"


def _cmd_train(a: argparse.Namespace) -> int:
    if a.backend:  # must happen before Keras is imported
        os.environ["KERAS_BACKEND"] = a.backend
    if a.device:
        os.environ["KERAS_TORCH_DEVICE"] = a.device
    from reef.model import ReefConfig
    from reef.train import TrainArgs, train

    cfg = ReefConfig()
    for key in ("width", "depth", "expansion", "input_dim", "dropout", "state_len", "schema_len", "grid_width",
                "bands", "mlp_rank"):
        v = getattr(a, key)
        if v is not None:
            setattr(cfg, key, v)
    cfg.global_max = not a.no_global_max
    cfg.hoist_empty_state, cfg.gate, cfg.mlp_norm = a.hoist_empty, a.gate, a.mlp_norm
    cfg.__post_init__()
    packing = {k: v for k, v in (("max_instruction_tokens", a.max_instruction_tokens),
                                 ("max_option_tokens", a.max_option_tokens)) if v is not None}
    if a.no_hoist:
        packing["hoist_instructions"] = False
    elif a.hoist:
        packing["hoist_instructions"] = True
    if a.mlp_leak is not None:  # weightless, so it also applies on top of --init
        packing["mlp_leak"] = a.mlp_leak
    if a.act_clip:  # weightless too: {layer name: [lo, hi]} (see reef.model.QClip)
        packing["act_clip"] = json.loads(Path(a.act_clip).read_text())
    weights = {}
    for w in a.weight or []:
        prefix, _, factor = w.rpartition("=")
        if not prefix:
            raise SystemExit(f"--weight expects PREFIX=FACTOR, got {w!r}")
        weights[prefix] = float(factor)
    args = TrainArgs(out=a.out, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
                     weight_decay=a.weight_decay, val_frac=a.val_frac, seed=a.seed, full=a.full,
                     train_data=a.data or ["typed-decisions:train"],
                     test_data=a.test if a.test is not None else ["typed-decisions:test"],
                     init=a.init, select_on=a.select_on, weights=weights, packing=packing)
    train(args, None if a.init else cfg)
    return 0


def _cmd_export(a: argparse.Namespace) -> int:
    from reef.export import export

    export(a.run, a.out, n_calibration=a.calibration, compile_tpu=not a.no_compile, name=a.name,
           int8_temperatures=not a.no_temperatures, calibrate_on=a.calibrate_on)
    return 0


def _cmd_calibrate(a: argparse.Namespace) -> int:
    from reef.calibrate import calibrate

    res = calibrate(a.artifact, device=a.device, limit=a.limit, data=a.data, write=not a.dry_run, select=a.select)
    print(json.dumps(res, indent=2))
    return 0


def _cmd_eval(a: argparse.Namespace) -> int:
    from reef.evaluate import dump, evaluate, prior_baseline

    if a.prior:
        dump(prior_baseline(a.split), a.json)
        return 0
    if not a.artifact:
        print("eval needs an artifact directory (or --prior)", file=sys.stderr)
        return 2
    res = evaluate(a.artifact, a.device, a.split, a.limit)
    ref = Path(a.artifact) / "train_report.json"
    if ref.exists() and a.split == "test" and a.limit is None:
        res["float_reference"] = json.loads(ref.read_text())["test"]
    dump(res, a.json)
    return 0


def _cmd_bench(a: argparse.Namespace) -> int:
    from reef.evaluate import bench, dump

    dump(bench(a.artifact, n=a.n, threads=a.threads), a.json)
    return 0


def _read_request(src: str) -> dict:
    text = sys.stdin.read() if src == "-" else Path(src).read_text()
    return json.loads(text)


def _cmd_decide(a: argparse.Namespace) -> int:
    from reef.runtime import Reef

    reef = Reef(a.artifact, device=a.device)
    print(json.dumps(reef.systemone(_read_request(a.request)), indent=2))
    return 0


def _cmd_serve(a: argparse.Namespace) -> int:
    from reef.server import serve

    serve(a.artifact, host=a.host, port=a.port, device=a.device, max_state_windows=a.max_state_windows,
          strict=a.strict, threads=a.threads)
    return 0


def _cmd_doctor(a: argparse.Namespace) -> int:
    from reef.edgetpu import doctor

    return doctor()


def _cmd_setup(a: argparse.Namespace) -> int:
    from reef.edgetpu import setup

    return 0 if setup(throttled=a.throttled, force=a.force) else 1


def _cmd_packs(a: argparse.Namespace) -> int:
    from reef import packs

    if a.name:
        p = packs.get(a.name)
        print(json.dumps({"name": p.name, "description": p.description, "questions": p.canonical_questions()},
                         indent=2))
        return 0
    for n in packs.names():
        print(f"{n:16s} {packs.get(n).description}")
    return 0


def _cmd_label(a: argparse.Namespace) -> int:
    from reef import packs
    from reef.distill import label
    from reef.teacher import from_spec

    extra = json.loads(a.extra_body) if a.extra_body else {}
    teacher = from_spec(a.teacher, model=a.model, permutations=a.permutations, extra_body=extra,
                        display_name=a.name) \
        if not a.teacher.rstrip("/").endswith("/v1/systemone") else from_spec(a.teacher)
    for name in a.pack:
        out = Path(a.out_dir) / f"{name}.jsonl"
        label(packs.get(name), teacher, a.n, out, seed=a.seed, workers=a.workers)
    return 0


def _cmd_merge(a: argparse.Namespace) -> int:
    from reef.distill import merge

    s = merge(a.inputs, a.out, a.weights)
    print(f"{a.out}: {s['cases']} cases, teachers agree on {s['teachers_agree']:.1%} of {s['questions']} questions")
    return 0


def _cmd_pull(a: argparse.Namespace) -> int:
    from reef.hub import pull

    print(pull(a.repo, a.dest, with_float=a.with_float))
    return 0


def _cmd_policy(a: argparse.Namespace) -> int:
    from reef import policy

    if a.teacher:
        answer = policy.teacher_answerer(a.teacher, a.model)
    elif a.artifact:
        answer = policy.model_answerer(a.artifact, a.device)
    else:
        raise SystemExit("give a model (dir or Hub id) or --teacher URL")
    res = policy.run(answer, policy.load(a.file))
    for f in res["failures"]:
        print(f"FAIL {f['id']:10s} {f['ask']:20s} got {f['top']} ({f['p_top']:.2f}), expected {'/'.join(f['expect'])} "
              f"(p={f['p_expected']:.2f})")
    print(f"passed {res['passed']}/{res['total']}  " + "  ".join(f"{k} {v}" for k, v in res["by_pack"].items()))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2))
    return 0 if res["passed"] == res["total"] or a.no_fail else 1


def _cmd_push(a: argparse.Namespace) -> int:
    from reef.hub import model_card, push

    if a.card_only:
        print(model_card(a.artifact, a.repo, teacher=a.teacher))
        return 0
    print(push(a.artifact, a.repo, private=a.private, teacher=a.teacher, card=a.card, message=a.message))
    return 0


def _cmd_push_data(a: argparse.Namespace) -> int:
    from reef.hub import push_dataset

    splits = {"train": a.train, **({"test": a.test} if a.test else {})}
    print(push_dataset(a.repo, splits, private=not a.public, teachers=a.teacher, message=a.message,
                       out_dir=a.dry_run, upload=not a.dry_run))
    return 0


def build_parser() -> argparse.ArgumentParser:
    from reef import __version__

    p = argparse.ArgumentParser(prog="reef", description="Tiny typed-decision models for the Coral Edge TPU.")
    p.add_argument("--version", action="version", version=f"reef {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train on LocalLLaMA/typed-decisions (needs reef-decide[train])")
    t.add_argument("--out", default="runs/reef-tiny")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--weight-decay", type=float, default=0.05)
    t.add_argument("--val-frac", type=float, default=0.2)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--full", action="store_true", help="no validation split: train on all of train")
    t.add_argument("--data", nargs="+", default=None,
                   help="training sources: typed-decisions:train, open-jev:train[?cap=N], org/dataset:split, "
                        "or `reef label` .jsonl files")
    t.add_argument("--test", nargs="*", default=None,
                   help="test sources (default typed-decisions:test); the first is the headline number")
    t.add_argument("--select-on", nargs="+", default=None, metavar="PREFIX",
                   help="only validation cases whose workflow starts with PREFIX pick the best epoch and fit "
                        "the temperatures (default: all)")
    t.add_argument("--weight", action="append", default=None, metavar="PREFIX=FACTOR",
                   help="sampling weight for training cases by workflow prefix, e.g. open-jev/=0.5 (repeatable)")
    t.add_argument("--no-global-max", action="store_true", help="drop the max-pool state summary")
    t.add_argument("--max-instruction-tokens", type=int, help="instruction tokens per question block (default 128)")
    t.add_argument("--max-option-tokens", type=int, help="tokens per option (default 48)")
    h = t.add_mutually_exclusive_group()
    h.add_argument("--hoist", action="store_true", help="move over-long instructions into the state (default)")
    h.add_argument("--no-hoist", action="store_true", help="truncate over-long instructions instead")
    t.add_argument("--hoist-empty", action="store_true",
                   help="new models: move the input of empty-state requests into the state")
    t.add_argument("--gate", action="store_true", help="new models: multiplicative state/band mixing")
    t.add_argument("--mlp-leak", type=float, default=None,
                   help="leaky MLP activation max(x, a*x) with slope a (0 = ReLU); also applies after --init")
    t.add_argument("--mlp-norm", action="store_true", help="new models: BatchNorm in front of each block MLP")
    t.add_argument("--act-clip", default=None, metavar="JSON",
                   help="int8-aware training: {layer: [lo, hi]} activation clips (also after --init)")
    t.add_argument("--init", default=None,
                   help="fine-tune: start from this artifact dir or Hub repo (its shape wins over shape flags)")
    t.add_argument("--backend", choices=["tensorflow", "torch"], default=None,
                   help="Keras backend for training; torch uses an Apple GPU (MPS) or CUDA when available")
    t.add_argument("--device", default=None, help="torch backend only: mps, cuda or cpu (default: auto)")
    g = t.add_argument_group("model shape (defaults: ReefConfig)")
    for name, typ in (("width", int), ("depth", int), ("expansion", int), ("input-dim", int),
                      ("dropout", float), ("state-len", int), ("schema-len", int), ("grid-width", int),
                      ("bands", int), ("mlp-rank", int)):
        g.add_argument(f"--{name}", type=typ, default=None)
    t.set_defaults(fn=_cmd_train)

    e = sub.add_parser("export", help="checkpoint -> int8 TFLite (+ Edge TPU compile) artifact")
    e.add_argument("run")
    e.add_argument("out")
    e.add_argument("--calibration", type=int, default=256, help="representative windows for int8 PTQ")
    e.add_argument("--no-compile", action="store_true", help="skip edgetpu_compiler")
    e.add_argument("--name", default=None)
    e.add_argument("--no-temperatures", action="store_true",
                   help="keep the float model's temperatures (skip refitting them on the int8 model)")
    e.add_argument("--calibrate-on", nargs="+", default=None, metavar="PREFIX",
                   help="workflows whose validation cases fit the int8 temperatures (default: --select-on)")
    e.set_defaults(fn=_cmd_export)

    ca = sub.add_parser("calibrate", help="refit an artifact's temperatures on its int8 outputs")
    ca.add_argument("artifact")
    ca.add_argument("--device", choices=["auto", "edgetpu", "cpu"], default="cpu")
    ca.add_argument("--limit", type=int, default=4000, help="validation cases to use (default 4000)")
    ca.add_argument("--data", nargs="+", default=None,
                    help="calibrate on these sources instead of the run's own validation split")
    ca.add_argument("--select", nargs="+", default=None, metavar="PREFIX",
                    help="fit on validation cases of these workflows (default: the run's --select-on)")
    ca.add_argument("--dry-run", action="store_true", help="report only, don't write config.json")
    ca.set_defaults(fn=_cmd_calibrate)

    ev = sub.add_parser("eval", help="score an artifact on typed-decisions")
    ev.add_argument("artifact", nargs="?")
    ev.add_argument("--device", choices=["auto", "edgetpu", "cpu"], default="auto")
    ev.add_argument("--split", default="test", help="typed-decisions split, or a source (open-jev:test, a .jsonl path)")
    ev.add_argument("--limit", type=int, default=None)
    ev.add_argument("--prior", action="store_true", help="score the train-mean Prior baseline instead")
    ev.add_argument("--json", default=None, help="also write the result here")
    ev.set_defaults(fn=_cmd_eval)

    b = sub.add_parser("bench", help="CPU vs Edge TPU latency")
    b.add_argument("artifact")
    b.add_argument("--n", type=int, default=200)
    b.add_argument("--threads", type=int, default=1, help="CPU interpreter threads")
    b.add_argument("--json", default=None)
    b.set_defaults(fn=_cmd_bench)

    d = sub.add_parser("decide", help="answer one SystemOne request ({state, questions}) from a file or stdin")
    d.add_argument("artifact")
    d.add_argument("request", nargs="?", default="-")
    d.add_argument("--device", choices=["auto", "edgetpu", "cpu"], default="auto")
    d.set_defaults(fn=_cmd_decide)

    s = sub.add_parser("serve", help="HTTP server: POST /v1/systemone, GET /healthz")
    s.add_argument("artifact", nargs="?", default=os.environ.get("REEF_MODEL", DEFAULT_MODEL),
                   help=f"model directory or Hub repo id (default: $REEF_MODEL or {DEFAULT_MODEL})")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--device", choices=["auto", "edgetpu", "cpu"], default="auto")
    s.add_argument("--max-state-windows", type=int, default=8,
                   help="overlapping chunks read from a long state (default 8, 1 = truncate)")
    s.add_argument("--strict", action="store_true",
                   help="refuse requests that do not fit (HTTP 413) instead of truncating them")
    s.add_argument("--threads", type=int, default=None, help="CPU interpreter threads (default: TFLite's choice)")
    s.set_defaults(fn=_cmd_serve)

    sub.add_parser("doctor", help="check the Edge TPU runtime and device").set_defaults(fn=_cmd_doctor)

    su = sub.add_parser("setup-edgetpu", help="install libedgetpu (macOS: download + patch into ~/.cache/reef)")
    su.add_argument("--throttled", action="store_true",
                    help="install the reduced-clock ('throttled') build instead of max clock ('direct')")
    su.add_argument("--force", action="store_true")
    su.set_defaults(fn=_cmd_setup)

    pk = sub.add_parser("packs", help="list built-in packs, or show one pack's questions")
    pk.add_argument("name", nargs="?")
    pk.set_defaults(fn=_cmd_packs)

    lb = sub.add_parser("label", help="generate pack states and label them with a teacher model")
    lb.add_argument("pack", nargs="+", help="pack name(s), see `reef packs`")
    lb.add_argument("--teacher", required=True,
                    help="OpenAI-compatible base URL with logprobs (llama.cpp, Ollama, MLX, vLLM), "
                         "or a full .../v1/systemone URL")
    lb.add_argument("--model", default=None, help="model name for chat endpoints")
    lb.add_argument("--name", default=None,
                    help='teacher name recorded in the data, e.g. "Gemma 4 E4B" (default: --model)')
    lb.add_argument("--n", type=int, default=2000, help="cases per pack")
    lb.add_argument("--out-dir", default="data")
    lb.add_argument("--seed", type=int, default=0)
    lb.add_argument("--workers", type=int, default=2, help="parallel requests (match the server's slots)")
    lb.add_argument("--permutations", type=int, default=1, help="re-ask with shuffled options and average")
    lb.add_argument("--extra-body", default=None,
                    help="JSON merged into each request (thinking is already turned off via "
                         "chat_template_kwargs.enable_thinking=false)")
    lb.set_defaults(fn=_cmd_label)

    mg = sub.add_parser("merge", help="ensemble teachers: average soft labels of the same cases from several files")
    mg.add_argument("inputs", nargs="+", help="labelled .jsonl files (same pack, seed and n)")
    mg.add_argument("--out", required=True)
    mg.add_argument("--weights", type=float, nargs="+", default=None, help="one weight per input (default equal)")
    mg.set_defaults(fn=_cmd_merge)

    pl = sub.add_parser("pull", help="download a model from the Hugging Face Hub")
    pl.add_argument("repo", help="e.g. seraphimserapis/reef[@revision]")
    pl.add_argument("dest", nargs="?", default=None, help="target directory (default: the HF cache)")
    pl.add_argument("--with-float", action="store_true", help="also fetch the float weights (for fine-tuning)")
    pl.set_defaults(fn=_cmd_pull)

    po = sub.add_parser("policy", help="run the hand-written policy tests against a model or a teacher")
    po.add_argument("artifact", nargs="?", help="model directory or Hub repo id")
    po.add_argument("--teacher", action="append", default=None,
                    help="test a teacher endpoint instead (uses each pack's notes); repeat for an averaged ensemble")
    po.add_argument("--model", action="append", default=None, help="teacher model name (repeat per --teacher)")
    po.add_argument("--device", choices=["auto", "edgetpu", "cpu"], default="auto")
    po.add_argument("--file", default=None, help="policy cases (default: the built-in reef/evals/policy.jsonl)")
    po.add_argument("--json", default=None)
    po.add_argument("--no-fail", action="store_true", help="exit 0 even when cases fail")
    po.set_defaults(fn=_cmd_policy)

    ps = sub.add_parser("push", help="upload an exported artifact to the Hugging Face Hub, with a model card")
    ps.add_argument("artifact")
    ps.add_argument("repo", help="e.g. yourname/reef-mine")
    ps.add_argument("--private", action="store_true")
    ps.add_argument("--teacher", default=None, help='credit the teacher model in the card, e.g. "Gemma 4 E4B"')
    ps.add_argument("--card", default=None, help="use this README.md instead of the generated card")
    ps.add_argument("--card-only", action="store_true", help="write the generated card to stdout, upload nothing")
    ps.add_argument("--message", default=None, help="commit message")
    ps.set_defaults(fn=_cmd_push)

    pd_ = sub.add_parser("push-data", help="upload labelled .jsonl files as a Hub dataset, with a dataset card")
    pd_.add_argument("repo", help="e.g. seraphimserapis/reef-distill")
    pd_.add_argument("--train", nargs="+", required=True)
    pd_.add_argument("--test", nargs="+", default=None)
    pd_.add_argument("--teacher", action="append", default=None,
                     help="teacher name(s) for the card (default: from the data)")
    pd_.add_argument("--public", action="store_true", help="create the repo public (default: private)")
    pd_.add_argument("--message", default=None)
    pd_.add_argument("--dry-run", default=None, metavar="DIR", help="only write parquet + card to DIR")
    pd_.set_defaults(fn=_cmd_push_data)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
