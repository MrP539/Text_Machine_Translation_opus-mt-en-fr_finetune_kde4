from transformers import pipeline

# Replace this with your own checkpoint
model_checkpoint = "MRP101py/model-finetuned-kde4-en-to-fr" # ลองโหลดโมเดลมาใช้
translator = pipeline("translation", model=model_checkpoint,trust_remote_code=True)

result = translator("Default to expanded threads")

print(result)