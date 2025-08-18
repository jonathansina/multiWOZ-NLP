import tqdm
from typing import List, Dict, Tuple

from transformers import T5Tokenizer
from torch.utils.data import Dataset
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def get_evaluation_score(predictions: List[str], references: List[str]) -> float:
    smoothing = SmoothingFunction().method1
    
    predictions = [pred.strip().lower() for pred in predictions]
    references = [ref.strip().lower() for ref in references]
    
    bleu_scores = [
        sentence_bleu(
            [ref.split()], 
            pred.split(), 
            smoothing_function=smoothing,
            weights=(1, 0, 0, 0)
        ) 
        for pred, ref in zip(predictions, references)
    ]

    P, R, F1 = bert_score(
        predictions, 
        references, 
        lang='en', 
        rescale_with_baseline=True, 
        verbose=True, 
        batch_size=256
    )
    
    avg_bert_f1 = F1.mean().item()
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    
    return {
        "bleu_score": avg_bleu,
        "bert_f1_score": avg_bert_f1
    }


def preprocess_text_realization(dialogue: Dict) -> List[Tuple[str, str]]:
    samples = []
    turns = dialogue["turns"]
    utterances = turns["utterance"]
    speakers = turns["speaker"]
    acts = turns["dialogue_acts"]

    last_user_utt = ""
    for i, (utt, spk) in enumerate(zip(utterances, speakers)):
        if spk == 0:
            last_user_utt = utt
        
        if spk == 1 and i < len(acts):
            act = acts[i]["dialog_act"]

            label_parts = []
            for act_type, act_slots in zip(act["act_type"], act["act_slots"]):
                slots = [f"{s}={v}" for s, v in zip(act_slots.get("slot_name", []), act_slots.get("slot_value", []))]
                label_parts.append(f"{act_type}({', '.join(slots)})")
            action_label = " | ".join(label_parts)

            input_with_context = f"[USER]: {last_user_utt} [ACTION]: {action_label}"

            samples.append((input_with_context, utt))

    return samples


class ResponseDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: T5Tokenizer, max_output_len: int = 64, max_input_len: int = 64):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

        self.actions = []
        self.responses = []

        for dialogue in tqdm.tqdm(data, desc="Processing dialogues"):
            preprocessed_text = preprocess_text_realization(dialogue)

            for (action, response) in preprocessed_text:
                self.actions.append(action)
                self.responses.append(response)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        action_enc = self.tokenizer(
            self.actions[idx], 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_input_len, 
            return_tensors="pt"
        )

        response_enc = self.tokenizer(
            self.responses[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_output_len,
            return_tensors="pt"
        )

        return {
            "encoder_input_ids": action_enc["input_ids"].squeeze(0),
            "encoder_attention_mask": action_enc["attention_mask"].squeeze(0),
            "decoder_input_ids": response_enc["input_ids"].squeeze(0),
            "decoder_attention_mask": response_enc["attention_mask"].squeeze(0)
        }