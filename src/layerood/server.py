"""Optional loopback HTTP interface. Install with `uv sync --extra serve`."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .pipeline import OODPipeline


class TextRequest(BaseModel):
    texts: list[str] = Field(min_length=1,max_length=16)
    alpha: float = Field(default=.05,gt=0,lt=1)


def create_app(detector_path, local_files_only=False):
    pipeline = OODPipeline.load(detector_path,local_files_only)
    app = FastAPI(title="LayerOOD",version="0.1.0",description="Local embeddings and ID-calibrated OOD scores")
    def validate(request):
        if any(not t.strip() or len(t)>8000 for t in request.texts):
            raise HTTPException(422,"Texts must be nonempty and at most 8,000 characters each")
    @app.get("/healthz")
    def health():
        return {"status":"ready","model":pipeline.encoder.identity,"methods":pipeline.detector.metadata["methods"]}
    @app.post("/v1/ood")
    def score(request: TextRequest):
        validate(request)
        return {"primary_method":"local_transport","alpha":request.alpha,
                "p_id_meaning":"ID-calibrated tail rank, not a probability of being ID or OOD",
                "results":pipeline.predict(request.texts,request.alpha)}
    @app.post("/v1/embeddings")
    def embeddings(request: TextRequest):
        validate(request)
        with pipeline._lock:
            batch=pipeline.encoder.encode([{"id":f"request-{i}","text":t} for i,t in enumerate(request.texts)])
        return {"metadata":batch.metadata,"records":batch.records,"embeddings":batch.values.tolist()}
    return app
