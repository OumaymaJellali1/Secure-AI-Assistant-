import json
from paddleocr import PaddleOCRVL

# initialize the PaddleOCR-VL pipeline
pipeline = PaddleOCRVL(pipeline_version="v1.5")

# run prediction
result = pipeline.predict("image.png")

# debug: print raw result
print(result)

# since `result` is already a dict, use it directly
output = {"text": []}

for item in result.get("res", {}).get("rec_texts", []):
    output["text"].append(item)

with open("output.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=4)

print("Saved to output.json")