import json
from rapidocr_onnxruntime import RapidOCR

engine = RapidOCR()
result, elapse = engine("image.png")

if result:
    output = {"text": [line[1] for line in result]}
    
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    
    print("Saved to output.json")
else:
    print("No text detected")