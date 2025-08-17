import torch


def inference_model(model, tokenizer, input_text, max_length, device):
    model.eval()
    with torch.no_grad():
        input_enc = tokenizer(
            input_text,
            padding='max_length',
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(device)

        output = model.generate(
            input_ids=input_enc['input_ids'],
            attention_mask=input_enc['attention_mask'],
            max_length=128
        )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response