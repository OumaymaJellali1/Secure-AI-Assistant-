import json
import easyocr

reader = easyocr.Reader(['ar'])  # Arabic
result = reader.readtext('image.png', detail=0)  # detail=0 = text only

output = {"text": result}

with open("output.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=4)

print("Saved to easyocr.json")