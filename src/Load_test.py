from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load the fine-tuned model
model_path = "./outputs/fine_tuned_distilbert"  # path to the folder where model was saved(output in my case)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Manually set the label mapping
model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
model.config.label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# Create a pipeline for easy inference
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Test it
print(sentiment_pipeline("I really enjoyed using this product!"))
print(sentiment_pipeline("This is the worst thing ever."))
