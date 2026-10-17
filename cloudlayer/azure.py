"""Azure adapter. Implement upload/download/push_image for Lab 1.

SDK:  pip install azure-storage-blob azure-identity azure-containerregistry
Docs: BlobServiceClient for storage; ACR push goes through `docker push` after
      `az acr login --name <registry>`.

Hints for Lab 1:
  * BLOB_URI is either abfss://container@account.dfs.core.windows.net/prefix or
    https://account.blob.core.windows.net/container/prefix. Pick one form and parse
    it here, never in src/.
  * Use DefaultAzureCredential rather than a connection string. It picks up your CLI
    login locally and your managed identity in CI, which is what Lab 4 needs.
  * push_image must return the digest reference: registry.azurecr.io/repo@sha256:...
  * Azure tags live on the resource, not the blob. Tag the storage account, the
    registry, and later the workspace with cfg.tags(1).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from urllib.parse import urlparse

from cloudlayer.base import CloudAdapter


def _run(*cmd: str) -> str:
    # shutil.which resolves az -> az.cmd on Windows; a bare "az" can't be launched without a shell there
    exe = shutil.which(cmd[0]) or cmd[0]
    return subprocess.run([exe, *cmd[1:]], check=True, capture_output=True, text=True).stdout.strip()


class AzureAdapter(CloudAdapter):
    def _blob(self, uri: str):
        """BlobClient for https://<account>.blob.core.windows.net/<container>/<blob path>.
        The account is the hostname, the container is the FIRST path segment (the classic swap)."""
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobClient
        u = urlparse(uri)
        container, _, blob = u.path.lstrip("/").partition("/")
        if u.scheme != "https" or not u.netloc.endswith(".blob.core.windows.net") or not blob:
            raise ValueError(f"not an https Azure blob URI with a container and a path: {uri}")
        # credential chain, not a connection string: az login locally, managed identity in CI
        return BlobClient(f"https://{u.netloc}", container, blob, credential=DefaultAzureCredential())

    def upload(self, local_path: str, key: str) -> str:
        uri = f"{self.cfg.blob_uri.rstrip('/')}/{key.lstrip('/')}"
        with open(local_path, "rb") as f:
            self._blob(uri).upload_blob(f, overwrite=True)
        return uri

    def download(self, uri: str, local_path: str) -> None:
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(self._blob(uri).download_blob().readall())


    def push_image(self, local_tag: str) -> str:
        """Tag + push to CONTAINER_REGISTRY; return registry/repo@sha256:... (a tag can be moved
        to other bytes, a digest can't - so 'which code is serving?' has one answer)."""
        registry = self.cfg.container_registry.rstrip("/")          # itcs355<id>.azurecr.io/itcs355
        name, _, tag = local_tag.partition(":")
        remote = f"{registry}/{name.rsplit('/', 1)[-1]}:{tag or 'latest'}"
        _run("az", "acr", "login", "--name", registry.split(".", 1)[0])   # lets docker push
        _run("docker", "tag", local_tag, remote)
        _run("docker", "push", remote)
        digests = _run("docker", "inspect", "--format", "{{json .RepoDigests}}", remote)
        repo = remote.rsplit(":", 1)[0]
        ref = next((d for d in digests.strip("[]").replace('"', "").split(",") if d.startswith(repo + "@")), "")
        if not ref:
            raise RuntimeError(f"pushed {remote} but found no digest for it in {digests}")
        return ref

    # submit_training / register_model  -> Lab 2 (Azure ML command job + model registry)
    # deploy / invoke                   -> Lab 3. On Azure for Students, deploy to
    #                                      Container Apps, NOT an Azure ML managed
    #                                      online endpoint: those need
    #                                      ceil(1.2 x instances) x cores of quota, so
    #                                      one Standard_DS3_v2 asks for 8 vCPU against
    #                                      a student cap of about 3, and the cap cannot
    #                                      be raised. Container Apps has no Azure ML
    #                                      core quota and scales to zero.
    # emit_metric                       -> Lab 4 (Azure Monitor custom metric)
    # generate                          -> Lab 5 (managed LLM endpoint; read the usage block for tokens)
    # teardown                          -> Lab 5 (resource graph query by tag)
