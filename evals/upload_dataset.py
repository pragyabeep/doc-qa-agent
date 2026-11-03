"""Optional: copy dataset.json into Langfuse as a Dataset so you can browse it in the UI.
   python -m evals.upload_dataset"""
import json
from pathlib import Path

from langfuse import get_client

lf = get_client()
name = "cloudsync-qa"
lf.create_dataset(name=name, description="CloudSync document Q&A eval set")
for it in json.loads((Path(__file__).parent / "dataset.json").read_text()):
    lf.create_dataset_item(dataset_name=name, input={"question": it["question"]},
                           expected_output={"keywords": it["expected_keywords"], "source": it["expected_source"]},
                           metadata={"id": it["id"], "answerable": it["answerable"]})
lf.flush()
print("Uploaded dataset:", name)
