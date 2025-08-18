import tqdm
from typing import List, Dict, Tuple

from transformers import T5Tokenizer
from torch.utils.data import Dataset


def preprocess_action_prediction(dialogue: Dict, max_turns: int = 5) -> List[Tuple[str, str]]:
    samples = []
    turns = dialogue["turns"]
    utterances = turns["utterance"]
    speakers = turns["speaker"]
    acts = turns["dialogue_acts"]

    context = []
    for i, (utt, spk) in enumerate(zip(utterances, speakers)):
        context.append(("USER" if spk == 0 else "SYS") + ": " + utt)
        
        if len(context) > max_turns:
            context = context[-max_turns:]
        
        if spk == 1 and i < len(acts):
            act = acts[i]["dialog_act"]
            
            # flatten action into string label (e.g. "Restaurant-Inform(area=centre, pricerange=expensive)")
            label_parts = []
            for act_type, act_slots in zip(act["act_type"], act["act_slots"]):
                slots = [f"{s}={v}" for s, v in zip(act_slots.get("slot_name", []), act_slots.get("slot_value", []))]
                label_parts.append(f"{act_type}({', '.join(slots)})")
            action_label = " | ".join(label_parts)

            input_text = " ".join([c for c in context])
            samples.append((input_text, action_label))

    return samples


class ActionDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: T5Tokenizer, max_turns: int, max_output_len: int = 64, max_input_len: int = 64):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        
        self.inputs = []
        self.actions = []

        for dialogue in tqdm.tqdm(data, desc="Processing dialogues"):
            preprocessed_text = preprocess_action_prediction(dialogue, max_turns=max_turns)

            for (input, action) in preprocessed_text:
                self.inputs.append(input)
                self.actions.append(action)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        input_enc = self.tokenizer(
            self.inputs[idx], 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_input_len, 
            return_tensors="pt"
        )

        action_enc = self.tokenizer(
            self.actions[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_output_len,
            return_tensors="pt"
        )

        return {
            "encoder_input_ids": input_enc["input_ids"].squeeze(0),
            "encoder_attention_mask": input_enc["attention_mask"].squeeze(0),
            "decoder_input_ids": action_enc["input_ids"].squeeze(0),
            "decoder_attention_mask": action_enc["attention_mask"].squeeze(0)
        }