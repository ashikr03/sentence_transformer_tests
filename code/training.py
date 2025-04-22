from datasets import load_dataset
from multitask_learning_model.py import MultiTaskLearning_ClassificationandSentiment

ds = load_dataset("stanfordnlp/imdb")
ds_train = ds["train"]
ds_test = ds["test"]
ds_train = ds_train.rename_column('label', 'sentiment_labels')
ds_test = ds_test.rename_column('label', 'sentiment_labels')
ds_train = ds_train.add_column('classification_labels', np.zeros(len(ds_train), dtype=np.int64))
ds_test = ds_test.add_column('classification_labels', np.zeros(len(ds_test), dtype=np.int64))


tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

def tokenization(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

ds_train = ds_train.map(tokenization, batched=True, batch_size=None)
ds_test = ds_test.map(tokenization, batched=True, batch_size=None)
ds_train.set_format('torch', columns=['input_ids', 'attention_mask', 'sentiment_labels', 'classification_labels'])
ds_test.set_format('torch', columns=['input_ids', 'attention_mask', 'sentiment_labels', 'classification_labels'])

sent_nums = 2

model = MultiTaskLearning_ClassificationandSentiment(2, sent_nums)
model.to(device)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=5e-4,
    weight_decay=0.01,
    optim='adamw_torch',
    lr_scheduler_type='linear',
    warmup_steps=500,
    eval_strategy='epoch',
    save_strategy='epoch',
    save_steps=1000,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_test,
    compute_metrics=compute_metrics,
    data_collator=custom_collate
    )


trainer.train()
