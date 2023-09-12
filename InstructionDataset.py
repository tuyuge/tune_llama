import torch
import json
import bmtrain as bmt
from time import time
from model_center.tokenizer import LlamaTokenizer
import copy


class InstructionDataset(torch.utils.data.Dataset):
    def __init__(self, mode="train"):
        super().__init__()
        self.data = []
        self.mode = mode

    def make_input(self, tokenizer, prompt, output, max_words):
        raw_prompt = prompt
        raw_output = output
        

        prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.int32) # "Instruction+Input" [1, 3, 4]
        output_ids = torch.tensor(tokenizer.encode(output)[1:]+[tokenizer.eos_token_id], dtype=torch.int32) # "Output" [5,2] # remove bos token (1) and add eos token (2) at the end of the output
        if self.mode == "train":
            total_ids = torch.cat((prompt_ids, output_ids))
        # else:
        #     total_ids= prompt_ids
        
        
        # if self.mode == "train":
        #     total_ids = tokenizer.encode(prompt)+tokenizer.encode(output)[1:]+[tokenizer.eos_token_id] # "Instruction+Input" [1, 3, 4]
        # else:
        #     total_ids= tokenizer.encode(prompt)+[tokenizer.eos_token_id] # "Output" [5,2] # remove bos token (1) and add eos token (2) at the end of the output
        
        # total_ids = torch.tensor(total_ids, dtype=torch.int32)
        
        
        length = torch.tensor(len(total_ids), dtype=torch.int32) 
        

        padding = max_words - total_ids.shape[0]
        if padding > 0:
            total_ids = torch.cat((total_ids, torch.zeros(padding, dtype=torch.int32) - 1))
        elif padding < 0:
            total_ids = total_ids[: max_words] # [1, 3, 4, 5, 2, 0]

        targets = copy.deepcopy(total_ids)
        targets[: len(prompt_ids)] = -1 
        attention_mask = total_ids.ge(0) 
        targets_mask = targets.ge(0) 
        total_ids[~attention_mask] = 0  
        targets[~targets_mask] = -100 
        targets = torch.cat((targets[1:], torch.tensor([-100], dtype=torch.int32))) 
        attention_mask = attention_mask.int()
        targets = targets.long()

        self.data.append({              
            "input_ids": total_ids.cuda(),  # [1,3,4,5,2,0]
            "attention_mask": attention_mask.cuda(),  # [1,1,1,1,1,0]
            "length": length.cuda(), #5
            "targets": targets.cuda(), # [-100,-100,5,2,-100,-100]
            "raw_prompt": raw_prompt,
            "raw_output": raw_output,
        })

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class Alpaca_Dataset(InstructionDataset):
    def __init__(self, data_path, tokenizer, max_words) -> None:
        super().__init__()

        PROMPT_DICT = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:"
            ),
            }
        data_path = f"{data_path}/alpaca_data.json"
        with open(data_path) as f:
            data = json.load(f)

        for count, raw in enumerate(data):
            if raw.get("input", "") == "":
                prompt = PROMPT_DICT["prompt_no_input"].format_map(raw)
            else:
                prompt = PROMPT_DICT["prompt_input"].format_map(raw)


            output = raw["output"]

            self.make_input(tokenizer, prompt, output, max_words)
        bmt.print_rank('Alpaca data loaded, all data num:', len(data))


if __name__ == "__main__":
    # data_path = "/data_new/private/tuyuge/datasets"
    # model_path = f"/data_new/private/tuyuge/results/llama-7b"

    # tokenizer = LlamaTokenizer.from_pretrained(model_path)
    # dataset = Alpaca_Dataset(data_path, tokenizer, max_words=512)
    # from model_center.dataset import DistributedDataLoader
    # dataloader = DistributedDataLoader(dataset, batch_size=16, shuffle=True)


    # from model_center.model.config import LlamaConfig
    # from model_center.model import Llama

    # bmt.init_distributed()

    # for it, data in enumerate(dataloader):
    #     input_ids = data["input_ids"]
    #     length = data["length"]
    #     attention_mask = data["attention_mask"]
    #     targets = data["targets"]
    #     raw = data["raw_prompt"]


        # s = f"{raw}\n{input_ids}\n{length}\n{attention_mask}\n{targets}"
        # with open("results.txt", "w", encoding="utf-8") as f:
        #     f.write(s)


        # plmpath = "/data_new/private/tuyuge/results/llama-7b/pytorch_model.pt"
        # # plmconfig = LlamaConfig()
        # # model = Llama(plmconfig)
        # # bmt.load(model, plmpath, strict=False)

        
        # from AdapterLLaMa import AdapterLLaMa
        # model = AdapterLLaMa(plmpath)
        
        

        # logits = model(input_ids, length, attention_mask).logits
        # batch, seq_len, vocab_out_size = logits.size()
        # loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)
        # loss = loss_func(logits.view(batch * seq_len, vocab_out_size), targets.view(batch * seq_len))

        
        # global_loss = bmt.sum_loss(loss).item()
        # bmt.print_rank(global_loss) 
        # # 原始llama：1.3511120080947876
        # # Adapter: 1.357224941253662

        # break
        data_path = "/data_new/private/tuyuge/datasets"
        data_path+="/alpaca_data.json"
        with open(data_path) as f:
            data = json.load(f)
            data['input']