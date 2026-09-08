"""Index the official WSDataset v7 without downloading the 18 GB image corpus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://www.kaggle.com/api/v1/datasets"
DATASET = "gloryvu/wildfire-smoke-detection"
VERSION = 7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=.5, status_forcelist=[404, 429, 500, 502, 503, 504])))
    metadata_response = session.get(f"{BASE}/view/{DATASET}", timeout=30)
    metadata_response.raise_for_status()
    metadata = metadata_response.json()
    if metadata.get("currentVersionNumber") != VERSION or metadata.get("licenseName") != "MIT":
        raise ValueError("Dataset version/license changed; inspect before acquiring")
    (args.output / "source_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    state_path = args.output / "index_state.json"
    index_path = args.output / "file_index.jsonl"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"page": 0, "token": None, "files": 0, "complete": False}
    if state["complete"]:
        print(json.dumps({k: state[k] for k in ("page", "files", "complete")}))
        return
    if state["page"] == 0 and index_path.exists():
        raise ValueError("Uncheckpointed index exists; inspect rather than overwrite")
    with index_path.open("a", encoding="utf-8") as stream:
        while state["page"] < 650:
            response = session.get(f"{BASE}/list/{DATASET}", params={"datasetVersionNumber": VERSION, "pageSize": 200, "pageToken": state["token"]}, timeout=30)
            response.raise_for_status()
            page = response.json()
            if page.get("hasErrorMessage"):
                raise ValueError(page["errorMessage"])
            files = page.get("datasetFiles", [])
            for item in files:
                stream.write(json.dumps({"name": item["name"], "bytes": item["totalBytes"], "creation_date": item.get("creationDate")}) + "\n")
            stream.flush()
            state = {"page": state["page"] + 1, "files": state["files"] + len(files),
                     "token": page.get("nextPageToken"), "complete": not page.get("hasNextPageToken", False)}
            state_path.write_text(json.dumps(state), encoding="utf-8")
            if state["page"] % 20 == 0 or state["complete"]:
                print(json.dumps({"pages": state["page"], "indexed_files": state["files"], "complete": state["complete"], "last_name": files[-1]["name"] if files else None}), flush=True)
            if state["complete"]:
                break
    if not state["complete"]:
        raise RuntimeError("Metadata page budget reached; no image downloaded")


if __name__ == "__main__":
    main()
