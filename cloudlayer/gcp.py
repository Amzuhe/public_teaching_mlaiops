"""GCP adapter. Implement upload/download/push_image for Lab 1.

SDK:  pip install google-cloud-storage google-cloud-aiplatform
Docs: storage.Client for GCS; Artifact Registry push goes through `docker push` after
      `gcloud auth configure-docker <region>-docker.pkg.dev`.

Hints for Lab 1:
  * BLOB_URI looks like gs://bucket/prefix — parse it here, never in src/.
  * Artifact Registry paths are region-scoped:
        <region>-docker.pkg.dev/<project>/<repo>/<image>
    A common first failure is pushing to gcr.io out of habit; it is a different service.
  * push_image must return the digest reference, not the tag.
  * GCP calls them labels, not tags, and they must be lowercase with no spaces.
    cfg.tags(1) already satisfies that constraint — do not "improve" the values.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from cloudlayer.base import CloudAdapter

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GcpAdapter(CloudAdapter):
    def upload(self, local_path: str, key: str) -> str:
        raise NotImplementedError("TODO Lab 1: blob.upload_from_filename, return the gs:// URI")

    def download(self, uri: str, local_path: str) -> None:
        raise NotImplementedError("TODO Lab 1: blob.download_to_filename, creating parents")

    def push_image(self, local_tag: str) -> str:
        raise NotImplementedError("TODO Lab 1: configure-docker, push, return repo@sha256:...")

    # submit_training / register_model  -> Lab 2 (Vertex custom training + Model Registry)
    # deploy / invoke                   -> Lab 3 (Vertex Endpoint)
    # emit_metric                       -> Lab 4 (Cloud Monitoring time series)
    def generate(self, prompt: str, params: dict[str, Any]) -> dict[str, Any]:
        """Lab 5: one Gemini call. Tokens come from the API's usageMetadata, never counted.

        shortcut: Gemini Developer API with an API key (free tier), not Vertex AI, because
        no GCP project exists; same model family and the same usageMetadata fields.
        """
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set (put it in cloud.env, never in Git)")
        model = params.get("model") or os.environ.get("LLM_MODEL", "gemini-3.5-flash-lite")
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": int(params.get("max_output_tokens", 200)),
                "temperature": float(params.get("temperature", 0.0)),
            },
        }
        if params.get("json"):
            body["generationConfig"]["responseMimeType"] = "application/json"
        if params.get("system"):
            body["systemInstruction"] = {"parts": [{"text": params["system"]}]}
        req = urllib.request.Request(
            GEMINI_URL.format(model=model), data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
        )
        for attempt in range(4):  # the free tier rate-limits (429); back off and retry
            start = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.load(resp)
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 429 or attempt == 3:
                    raise RuntimeError(f"Gemini HTTP {exc.code}: {exc.read()[:300]!r}") from exc
                time.sleep(15 * (attempt + 1))
        latency_ms = round((time.perf_counter() - start) * 1000)
        cand = (data.get("candidates") or [{}])[0]
        text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
        usage = data.get("usageMetadata", {})
        return {
            "response": text,
            "input_tokens": usage.get("promptTokenCount", 0),
            # thinking tokens are billed as output, so they belong in the output count
            "output_tokens": usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0),
            "cached_input_tokens": usage.get("cachedContentTokenCount", 0),
            "latency_ms": latency_ms,
            "finish_reason": cand.get("finishReason"),
        }

    # teardown                          -> Lab 5 (filter resources by label)
