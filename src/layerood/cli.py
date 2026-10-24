"""CLI for extraction, ID-only fitting, local inference and reproducible evaluation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

from .artifacts import load_embeddings, read_jsonl, write_json
from .detector import DetectorConfig, LayerOOD
from .embeddings import EncoderConfig, LocalEncoder
from .evaluation import evaluate, markdown_report


def _encoder_args(p):
    p.add_argument("--model",default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--revision",default="main")
    p.add_argument("--layers",help="Comma-separated transformer-output indices; default 1/3,2/3,final")
    p.add_argument("--pooling",choices=["mean","last"],default="mean")
    p.add_argument("--max-length",type=int,default=128)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--device",default="cpu")
    p.add_argument("--local-files-only",action="store_true")
    p.add_argument("--cache-dir",default=".cache/embeddings")


def _encoder(a):
    return EncoderConfig(model=a.model,revision=a.revision,
        layers=tuple(int(v) for v in a.layers.split(",")) if a.layers else None,
        pooling=a.pooling,max_length=a.max_length,batch_size=a.batch_size,threads=a.threads,
        device=a.device,local_files_only=a.local_files_only,cache_dir=a.cache_dir)


def _detector_args(p):
    p.add_argument("--rank",type=int,default=32)
    p.add_argument("--neighbors",type=int,default=10)
    p.add_argument("--ridge-alpha",type=float,default=10.)
    p.add_argument("--variance-shrinkage",type=float,default=.5)


def _detector(a):
    return DetectorConfig(rank=a.rank,neighbors=a.neighbors,ridge_alpha=a.ridge_alpha,
                          variance_shrinkage=a.variance_shrinkage,threads=getattr(a,"threads",4))


def _output(p):
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--overwrite",action="store_true")


def _report(detector,batch,alpha,seed=42):
    start=time.perf_counter()
    predictions=detector.score(batch,alpha,reject_seen=True)
    elapsed=time.perf_counter()-start
    report=evaluate(batch,predictions,alpha,seed)
    report["detector_scoring_seconds_all_methods"]=elapsed
    report["embedding_metadata"]=batch.metadata
    report["detector_config"]=detector.metadata["config"]
    return report,predictions


def main(argv=None):
    parser=argparse.ArgumentParser(prog="layerood",description="Local SLM embeddings and calibrated OOD detection")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("embed",help="Extract pooled hidden states to portable NPZ")
    p.add_argument("--input",required=True,type=Path);_output(p);_encoder_args(p)
    p=sub.add_parser("fit",help="Fit using three disjoint ID-only embedding splits")
    for split in ("train","reference","calibration"):p.add_argument("--"+split,required=True,type=Path)
    _output(p);_detector_args(p)
    p=sub.add_parser("score",help="Score embeddings with a saved detector")
    p.add_argument("--detector",required=True,type=Path);p.add_argument("--input",required=True,type=Path)
    p.add_argument("--alpha",type=float,default=.05);_output(p)
    p=sub.add_parser("predict",help="Embed and score raw JSONL using a saved detector")
    p.add_argument("--detector",required=True,type=Path);p.add_argument("--input",required=True,type=Path)
    p.add_argument("--alpha",type=float,default=.05);p.add_argument("--local-files-only",action="store_true");_output(p)
    p=sub.add_parser("evaluate",help="Evaluate a disjoint labeled embedding artifact")
    p.add_argument("--detector",required=True,type=Path);p.add_argument("--input",required=True,type=Path)
    p.add_argument("--alpha",type=float,default=.05);p.add_argument("--seed",type=int,default=42);_output(p)
    p=sub.add_parser("prepare-clinc",help="Download official CLINC source and prepare isolated splits")
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--profile",choices=["mini","quick","full"],default="quick")
    p.add_argument("--protocol",choices=["official","heldout-intents"],default="official")
    p.add_argument("--seed",type=int,default=42);p.add_argument("--id-fraction",type=float,default=.75)
    p.add_argument("--source",type=Path,help="Use an already downloaded official data_full.json")
    p=sub.add_parser("run",help="Embed four prepared splits, fit, and write evaluation artifacts")
    p.add_argument("--data",required=True,type=Path);p.add_argument("--output",required=True,type=Path)
    p.add_argument("--alpha",type=float,default=.05);p.add_argument("--seed",type=int,default=42)
    _encoder_args(p);_detector_args(p)
    p=sub.add_parser("inspect",help="Show artifact metadata without loading a language model")
    p.add_argument("path",type=Path)
    p=sub.add_parser("serve",help="Serve local embedding and OOD HTTP endpoints")
    p.add_argument("--detector",required=True,type=Path);p.add_argument("--port",type=int,default=8765)
    p.add_argument("--host",default="127.0.0.1");p.add_argument("--local-files-only",action="store_true")
    a=parser.parse_args(argv)
    created_run = False
    try:
        if hasattr(a,"alpha") and not 0 < a.alpha < 1:
            raise ValueError("alpha must be strictly between 0 and 1")
        if hasattr(a,"output") and a.output.exists() and not getattr(a,"overwrite",False):
            raise FileExistsError(f"Output exists: {a.output}; use a new path")
        if a.command=="embed":
            batch=LocalEncoder(_encoder(a)).encode(read_jsonl(a.input));batch.save(a.output,a.overwrite)
            result={"output":str(a.output),"shape":list(batch.values.shape),"metadata":batch.metadata}
        elif a.command=="fit":
            detector=LayerOOD(_detector(a)).fit(*(load_embeddings(getattr(a,k)) for k in ("train","reference","calibration")))
            detector.save(a.output,a.overwrite);result={"output":str(a.output),"methods":detector.metadata["methods"],"split_sizes":detector.metadata["split_sizes"]}
        elif a.command in ("score","evaluate"):
            detector=LayerOOD.load(a.detector);batch=load_embeddings(a.input)
            if a.command=="score":
                result={"alpha":a.alpha,"primary_method":"local_transport","results":detector.score(batch,a.alpha)}
            else:result,_=_report(detector,batch,a.alpha,a.seed)
            write_json(a.output,result,a.overwrite)
        elif a.command=="predict":
            from .pipeline import OODPipeline
            pipeline=OODPipeline.load(a.detector,a.local_files_only)
            batch=pipeline.encoder.encode(read_jsonl(a.input))
            result={"alpha":a.alpha,"primary_method":"local_transport","results":pipeline.detector.score(batch,a.alpha)}
            write_json(a.output,result,a.overwrite)
        elif a.command=="prepare-clinc":
            from .datasets import prepare_clinc
            result=prepare_clinc(a.output,a.profile,a.protocol,a.seed,a.id_fraction,a.source)
        elif a.command=="run":
            started=time.perf_counter();a.output.mkdir(parents=True)
            config={"command":sys.argv,"encoder":asdict(_encoder(a)),"detector":asdict(_detector(a)),
                    "python":platform.python_version(),"platform":platform.platform(),"alpha":a.alpha,"seed":a.seed,
                    "versions":{p:importlib.metadata.version(p) for p in ("layerood","torch","transformers","numpy","scipy","scikit-learn")},
                    "status":"started","scope":"Package execution; inspect data manifest for benchmark scope",
                    "code_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}}
            if (a.data/"manifest.json").exists():config["dataset"]=json.loads((a.data/"manifest.json").read_text())
            write_json(a.output/"run.json",config)
            created_run = True
            encoder=LocalEncoder(_encoder(a));batches={}
            for split in ("train","reference","calibration","evaluation"):
                print(f"Embedding {split}…",file=sys.stderr,flush=True)
                batches[split]=encoder.encode(read_jsonl(a.data/(split+".jsonl")))
                batches[split].save(a.output/(split+".npz"))
            detector=LayerOOD(_detector(a)).fit(*(batches[k] for k in ("train","reference","calibration")))
            detector.save(a.output/"detector.npz")
            result,predictions=_report(detector,batches["evaluation"],a.alpha,a.seed)
            write_json(a.output/"metrics.json",result);write_json(a.output/"predictions.json",predictions)
            (a.output/"report.md").write_text(markdown_report(result))
            config.update(status="completed",elapsed_seconds=time.perf_counter()-started,
                          resolved_encoder=encoder.identity,fit_seconds=detector.metadata["fit_seconds"])
            write_json(a.output/"run.json",config,overwrite=True)
            result={"output":str(a.output),"elapsed_seconds":config["elapsed_seconds"],"methods":list(result["methods"]),
                    "local_transport":result["methods"]["local_transport"]["all"]}
        elif a.command=="inspect":
            import numpy as np
            with np.load(a.path,allow_pickle=False) as stored:
                key="metadata_json" if "metadata_json" in stored.files else "metadata"
                result=json.loads(str(stored[key]))
                result.pop("used_text_hashes",None)
        elif a.command=="serve":
            try:
                import uvicorn
                from .server import create_app
            except ImportError as error:
                raise ValueError("Install the local service extra: uv sync --extra serve") from error
            uvicorn.run(create_app(a.detector,a.local_files_only),host=a.host,port=a.port)
            return 0
        else:
            raise ValueError("Unknown command")
        print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
        return 0
    except (ValueError,OSError,KeyError,RuntimeError) as error:
        if a.command=="run" and created_run and (a.output/"run.json").exists():
            previous=json.loads((a.output/"run.json").read_text())
            if previous.get("status")=="started":
                previous.update(status="failed",error=str(error))
                write_json(a.output/"run.json",previous,overwrite=True)
        parser.exit(2,f"layerood: {error}\n")


if __name__=="__main__":
    main()
