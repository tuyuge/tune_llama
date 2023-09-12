import bmtrain as bmt
import os
import json
from time import time

from model_center.model.config import LlamaConfig
from model_center.model import Llama
from model_center.tokenizer import LlamaTokenizer
from model_center.generation.llama import LlamaBeamSearch
from model_center.dataset import DistributedDataLoader
from InstructionDataset import Alpaca_Dataset
from LoraLLaMa import LoraLLaMa

def setup_model_and_optimizer():
    model_path = f"/data_new/private/tuyuge/results/llama-7b"
    plmpath = f"/data_new/private/tuyuge/results/finetune-llama-Alpaca-1.pt"

    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    bmt.synchronize()

    st = time()
    model = LoraLLaMa(plmpath)
    bmt.load(model, plmpath, strict=False)
    bmt.synchronize()
    bmt.print_rank('model loading time:', time()-st)

    return tokenizer, model

           
def main():
    bmt.init_distributed(seed=1949)

    tokenizer, model = setup_model_and_optimizer()
    model.eval()
    
    path = "/data_new/private/tuyuge/datasets"
   
    max_words = 512
    dataset = Alpaca_Dataset(path, tokenizer, max_words)
    dataloader = DistributedDataLoader(dataset, batch_size=8, shuffle=False)


    sampler = LlamaBeamSearch(model=model, tokenizer=tokenizer)

    with open("/data_new/private/tuyuge/generated_data/llama-7b/data.json", "w") as f:
        f.write("[\n")

        for iteration, query_data in enumerate(dataloader):
            inference_results = sampler.generate(query_data['raw_prompt'], beam_size=3, max_length=100, repetition_penalty=1.1)
            
            for k in range(len(query_data['raw_prompt'])):
                data = {'prompt': query_data['raw_prompt'][k], 'answer': inference_results[k]}
                bmt.print_rank(data)
                json.dump(data, f)
                if iteration < len(data) - 1:
                    f.write(",\n")
            break
        f.write("\n]")


if __name__ == "__main__":
    main()
    # srun -G 1 python llama_adapter/generate_result.py