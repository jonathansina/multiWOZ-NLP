
import tqdm
from typing import List, Dict, Tuple

from transformers import T5Tokenizer
from torch.utils.data import Dataset


def preprocess_text_realization(dialogue: Dict) -> List[Tuple[str, str]]:
    samples = []
    turns = dialogue["turns"]
    utterances = turns["utterance"]
    speakers = turns["speaker"]
    acts = turns["dialogue_acts"]

    for i, (utt, spk) in enumerate(zip(utterances, speakers)):
        if spk == 1 and i < len(acts):
            act = acts[i]["dialog_act"]
            # flatten as above
            label_parts = []
            for act_type, act_slots in zip(act["act_type"], act["act_slots"]):
                slots = [f"{s}={v}" for s, v in zip(act_slots.get("slot_name", []), act_slots.get("slot_value", []))]
                label_parts.append(f"{act_type}({', '.join(slots)})")
            action_label = " | ".join(label_parts)

            samples.append((action_label, utt))

    return samples


class ResponseDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: T5Tokenizer, max_output_len: int = 64):
        self.tokenizer = tokenizer
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
            max_length=self.max_output_len, 
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