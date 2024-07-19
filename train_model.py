import transformers
import datasets
import os
import numpy as np
import pandas as pd

##########################################################################  download & setup dataset  ####################################################################################

model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
    #โมเดล Helsinki-NLP/opus-mt-en-fr เป็นโมเดลการแปลภาษาที่สร้างขึ้นโดยทีม Helsinki-NLP และเป็นส่วนหนึ่งของโครงการ OPUS-MT ซึ่งเป็นการรวมตัวของโมเดลการแปลภาษาต่าง ๆ 
    # โมเดลนี้ถูกฝึกเพื่อแปลข้อความจากภาษาอังกฤษ (en) เป็นภาษาฝรั่งเศส (fr)

raw_dataset = datasets.load_dataset(path="kde4",lang1="en",lang2="fr",trust_remote_code=True) ##trust_remote_code=True: ตัวเลือกนี้อนุญาตให้รันโค้ดที่กำหนดเองจากรีโปซิทอรี เพื่อให้สามารถโหลดเมตริกได้อย่างถูกต้อง

tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_checkpoint,return_tensors="tf")
print(raw_dataset)  
print("\t\t\t\t|\n\t\t\t\t|\n\t\t\t\t|\n\t\t\t\t|\n\t\t\t\tV")

#ml-translation ต้องทำการตัดคำทั้ง source(ภาษาต้นทาง) และ tagets(ภาษาที่จะแปล)
max_length = 128
def encoder_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["fr"] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs,max_length=max_length,truncation=True)

    # ใช้ tokenizer ให้ตรงกับภาษานั้นๆ 
    with tokenizer.as_target_tokenizer():
        targets_labels = tokenizer(targets,max_length=max_length,truncation=True)

    # นำค่าที่ได้จากการตดคำที่ตรงกับภาษาปลายทางไปเพิ่มในdataset โดยให้ชื่อว่า labels โดยที่เราไป copyค่า มาจาก targets_labels["input_ids"]
    model_inputs["labels"] = targets_labels["input_ids"]
    return (model_inputs)

tokenized_dataset = raw_dataset.map(encoder_function,batched=True,remove_columns=raw_dataset["train"].column_names) 
    #remove_columns=split_datasets["train"].column_names: เอาคอลัมน์ทั้งหมดในชุดข้อมูล train ออกหลังจากประมวลผลเสร็จ เพื่อให้เก็บเฉพาะข้อมูลที่แปลงแล้ว.
    #การใช้ remove_columns เพื่อเอาคอลัมน์ดั้งเดิมออกเป็นประโยชน์ในหลายกรณี เช่น:
        # ลดขนาดของข้อมูลโดยเอาข้อมูลที่ไม่จำเป็นออก
        # ป้องกันการสับสนระหว่างข้อมูลดั้งเดิมและข้อมูลที่แปลงแล้ว
        # ปรับปรุงประสิทธิภาพการประมวลผลข้อมูล

#split data
#tokenized_dataset = tokenized_dataset["train"].train_test_split(train_size=0.9,seed=20)
train_set_size = 500
val_set_size = int(0.1*train_set_size)

tokenized_dataset = tokenized_dataset["train"].train_test_split(test_size=val_set_size,train_size=train_set_size,seed=20)
tokenized_dataset["validation"] = tokenized_dataset.pop("test")
#print(tokenized_dataset)
###########################################################################  create model  ####################################################################################

# Seq2Seq
# Goal ของ Seq2Seq
# รับ input ส่งเข้า Encoder แล้วอัดข้อมูลทุกอย่างให้เป็น S
# ความคาดหวังคือ "S" จะมี information สำหรับการทำนาย
# จากนั้น Decoder มีหน้าที่รับ S มาทำนาย
# ทำงานแบบ "Order มีผลต่อผลลัพธ์" (บทเรียนที่ 1 RNN) ทั้ง Encoder และ Decoder ทำงานเหมือนกัน (Sequence ทั้งต้นและปลาย)
# ตัว Decoder จะทำนายได้เรื่อยๆ จนกว่าทำนายจนเจอ token บอกให้หยุด (บางครั้งก็กำหนดได้ว่าจะหยุดเมื่อไหร่เช่น ทำนายแค่ 1 ครั้งพอ อารมณ์เดียวกับ next word prediction)

model = transformers.AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path=model_checkpoint)


data_collator = transformers.DataCollatorForSeq2Seq(tokenizer=tokenizer,model=model)
    #Data Collator คือออบเจกต์ที่ช่วยในการเตรียมข้อมูลสำหรับการป้อนเข้าโมเดลระหว่างการฝึก (training) หรือการประเมินผล (evaluation) โดยเฉพาะอย่างยิ่งการจัดการกับ batch ของข้อมูลที่มีความยาวไม่เท่ากัน.
    #transformers.DataCollatorForSeq2Seq เป็น Data Collator ที่มาพร้อมกับการเติมข้อมูล (padding) ให้เท่าข้อมูลมีความเท่ากัน แต่มีความพิเศษที่ ข้อมูลส่วน decoder จะถูก padding คำแรกของประโยคเสมอหลังจากนั้นก็ (padding) ให้เท่าข้อมูลมีความเท่ากัน
    # ex  ['<pad>','▁Par','▁défaut',',','▁développer','▁les','▁fils','▁de','▁discussion','</s>','<pad>','<pad>','<pad>','<pad>'] 

metrics = datasets.load_metric("sacrebleu",trust_remote_code=True) #trust_remote_code=True: ตัวเลือกนี้อนุญาตให้รันโค้ดที่กำหนดเองจากรีโปซิทอรี เพื่อให้สามารถโหลดเมตริกได้อย่างถูกต้อง
    #การโหลด metric "sacrebleu" จาก Hugging Face datasets library ใช้ในการประเมินผลการแปลภาษา (Machine Translation) โดยใช้ BLEU score 
    # ซึ่งเป็น metric ที่ใช้กันอย่างแพร่หลายในการวัดความใกล้เคียงระหว่างการ แปลของโมเดล และ การแปลที่ถูกต้อง (reference translation)
        #แต่มีจุดอ่อนที่ ไม่สนใจความหมายและบริบท
        # สรุป วัดว่าสามารถ pred คำต้นฉบับได้เหมือนขนาดไหนถึงแม้ Oder จะเสียไปก็ตาม

###########################################################################  Train   ################################################################################################

def compute_bleu_score_metrics(eval_pred):
    
    preds,labels = eval_pred
    print(f"Prediction: {preds}")
    print(f"Labels: {labels}")

    # ในกรณีที่แบบจำลองส่งคืนมากกว่าบันทึกการทำนาย
    # เงื่อนไขนี้ใช้ตรวจสอบว่าตัวแปร preds เป็น tuple หรือไม่ และถ้าใช่ มันจะเลือกเอาเฉพาะองค์ประกอบแรกของ tuple นั้นมาใช้ต่อไป:
    if isinstance(preds,tuple):
        preds = preds[0]

    # ใช้สำหรับคำนวณ BLEU score โดยเฉพาะการจัดการกับค่า -100 ใน labels ซึ่งมักถูกใช้เป็นค่ากรอก (padding) ที่ไม่ต้องการ และไม่สามารถถอดรหัสได้โดยตรง
    labels = np.where(labels != -100,labels,tokenizer.pad_token_id)
        # ใช้ np.where เพื่อแทนที่ค่าใน labels
            # ถ้า labels มีค่าไม่เท่ากับ -100, ค่านั้นจะคงอยู่ตามเดิมนั้นคือ labels
            # ถ้า labels มีค่าเท่ากับ -100, ค่านั้นจะถูกแทนที่ด้วย tokenizer.pad_token_id ซึ่งเป็น ID ของ token padding
        # วิธีนี้จะช่วยให้เราสามารถถอดรหัส labels ได้โดยไม่มีข้อผิดพลาด

    # ถอดรหัส token กลับเป็นข้อความ
        #ถอดรหัส token IDs หลาย ๆ ชุด(ถอดรหัส token IDs เป็น batch) (หลายประโยค) กลับเป็นข้อความ (string) ในคราวเดียว โดยข้าม token พิเศษ (special tokens)
    decoded_preds = tokenizer.batch_decode(sequences=preds,skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(sequences=labels,skip_special_tokens=True)

    # จัดการกับข้อความที่ถอดรหัสจาก token IDs โดยการลบช่องว่าง (whitespace) ที่ไม่จำเป็นออกจากต้นและท้ายของข้อความ
    decoded_pred = [pred.strip() for pred in decoded_preds]
    decoded_label = [[label.strip()] for label in decoded_labels]
 
    result= metrics.compute(predictions=decoded_pred,references=decoded_label)

    return {"bleu": result["score"]}

output_dir = "./results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class CSVLogger(transformers.TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.log_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            logs = {}
        self.log_history.append(logs)
        self.save_log()

    def save_log(self):
        df = pd.DataFrame(self.log_history)
        df.to_csv(os.path.join(self.output_dir, "training_log.csv"), index=False)
csv_logger = CSVLogger(output_dir="results") 


batch_size = 16
total_steps = len(tokenized_dataset["train"])//batch_size

training_arg = transformers.Seq2SeqTrainingArguments(
    output_dir="model-finetuned-kde4-en-to-fr",       # ไดเรกทอรีที่ใช้เก็บ checkpoint ของโมเดล
    num_train_epochs=1,             # จำนวนรอบการฝึกทั้งหมด
    eval_strategy="epoch",          # กลยุทธ์ในการประเมินผล ใช้ "epoch" หมายถึงประเมินผลหลังจากจบรอบการฝึกแต่ละรอบ
    save_strategy="epoch",          # กลยุทธ์ในการบันทึกโมเดล ใช้ "epoch" หมายถึงบันทึกโมเดลหลังจากจบรอบการฝึกแต่ละรอบ
    learning_rate=5e-5,             # อัตราการเรียนรู้ (learning rate)
    per_device_train_batch_size=batch_size, # ขนาดของ batch ที่ใช้ในการฝึกต่ออุปกรณ์หนึ่งเครื่อง
    per_device_eval_batch_size=batch_size,  # ขนาดของ batch ที่ใช้ในการประเมินผลต่ออุปกรณ์หนึ่งเครื่อง
    weight_decay=0.01,              # อัตราการลดทอนน้ำหนัก (weight decay) ใช้ในการป้องกันการ overfitting
    save_total_limit=3,             # จำนวน checkpoint สูงสุดที่จะถูกเก็บไว้ หากเกินจำนวนนี้ checkpoint เก่าจะถูกลบ
    fp16=True,                      # ใช้การคำนวณแบบ half-precision (FP16) เพื่อลดการใช้หน่วยความจำและเพิ่มความเร็ว
    predict_with_generate=True,     # ใช้การสร้างข้อความในการทำนาย (prediction)
    push_to_hub=True,               # ส่งโมเดลและข้อมูลไปยัง Hugging Face Hub
    logging_steps=total_steps       # บันทึกข้อมูลการฝึกอบรมทุก ๆ epoch
)

trainer = transformers.Seq2SeqTrainer(
    model=model,
    args=training_arg,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    callbacks=[csv_logger],
    compute_metrics=compute_bleu_score_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator
)
#print(f"Befor Train : {trainer.evaluate(max_length=max_length)} " )
print("\t\t\t\t|\n\t\t\t\t|\n\t\t\t\t|\n\t\t\t\t|\n\t\t\t\tV")
print("\n...Training...\n")

#trainer.train()

print("\t\t\t\t|\n\t\t\t\t|\n\t\t\t\t|\n\t\t\t\t|\n\t\t\t\tV")
print(f"After Train : {trainer.evaluate(max_length=max_length)} " )


