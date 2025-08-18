import tqdm
import torch


def inference_model(model, tokenizer, input_text, max_length_encoder, max_length_decoder, device, batch_size=1):
    model.eval()
    model.to(device)

    if isinstance(input_text, str):
        input_text = [input_text]
    
    input_enc = tokenizer(
        input_text,
        padding='max_length',
        truncation=True,
        max_length=max_length_encoder,
        return_tensors="pt"
    )

    all_responses = []
    
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(input_text), batch_size), desc="Inference"):
            batch_enc = input_enc['input_ids'][i:i + batch_size].to(device)
            batch_mask = input_enc['attention_mask'][i:i + batch_size].to(device)

            output = model.generate(
                input_ids=batch_enc,
                attention_mask=batch_mask,
                max_length=max_length_decoder
            )

            batch_responses = [
                tokenizer.decode(output[j], skip_special_tokens=True)
                for j in range(len(output))
            ]
            
            all_responses.extend(batch_responses)
        
        if len(all_responses) == 1:
            return all_responses[0]
        
        return all_responses