import os
import bmtrain as bmt

from arguments import get_args
from model_center.tokenizer import LlamaTokenizer
from model_center.utils import print_inspect
from model_center.dataset import DistributedDataLoader
from InstructionDataset import Alpaca_Dataset
from AdapterLLaMa import AdapterLLaMa
from LoraLLaMa import LoraLLaMa

def get_optimizer(args, model):
    optimizer = bmt.optim.AdamOffloadOptimizer(model.parameters(), weight_decay=args.weight_decay)
    return optimizer

def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters * args.epochs
    if args.lr_decay_style == "noam":
        lr_scheduler = bmt.lr_scheduler.Noam(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "constant":
        lr_scheduler = bmt.lr_scheduler.NoDecay(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = -1,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "linear":
        lr_scheduler = bmt.lr_scheduler.Linear(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "exponential":
        lr_scheduler = bmt.lr_scheduler.Exponential(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "cosine":
        lr_scheduler = bmt.lr_scheduler.Cosine(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler

def setup_model_and_optimizer(args):
    tokenizer = LlamaTokenizer.from_pretrained(args.model_config)
    # change model from here
    model = LoraLLaMa(args.model_config+'/pytorch_model.pt')
    bmt.synchronize()
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    bmt.synchronize()
    # get the memory usage
    # bmt.print_rank("Model mem\n", torch.cuda.memory_summary())
    # bmt.synchronize()
    return tokenizer, model, optimizer, lr_scheduler

def initialize():
    args = get_args()
    bmt.init_distributed(seed = args.seed)
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
    return args


def finetune(args, model, optimizer, lr_scheduler, dataset):
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)

    optim_manager = bmt.optim.OptimManager(loss_scale=args.loss_scale, loss_scale_steps=100)
    optim_manager.add_optimizer(optimizer, lr_scheduler)

    print_inspect(model, '*')

    for epoch in range(args.epochs):
        dataloader = DistributedDataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        # set to training mode
        model.train()

        for it, data in enumerate(dataloader):
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            targets = data["targets"] 


            logits = model(input_ids, attention_mask).logits
            batch, seq_len, vocab_out_size = logits.size()
            loss = loss_func(logits.view(batch * seq_len, vocab_out_size), targets.view(batch * seq_len))

            global_loss = bmt.sum_loss(loss).item()

            optim_manager.zero_grad()

            optim_manager.backward(loss)
            grad_norm = optim_manager.clip_grad_norm(optimizer.param_groups, args.clip_grad, norm_type = 2)

            optim_manager.step()

            bmt.print_rank(
                "train | epoch {:3d} | Iter: {:6d}/{:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | grad_norm: {:.4f} |".format(
                    epoch,
                    it,
                    len(dataloader),
                    global_loss,
                    lr_scheduler.current_lr,
                    int(optim_manager.loss_scale),
                    grad_norm,
                )
            )
            if it % args.inspect_iters == 0: print_inspect(model, "*")
        bmt.save(model, os.path.join(args.save, args.save_name+("-%d.pt" % epoch)))


def main():
    args = initialize()
    tokenizer, model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    path = "/data_new/private/tuyuge/datasets"
    max_words = args.max_length
    dataset = Alpaca_Dataset(path, tokenizer, max_words)
    finetune(args, model, optimizer, lr_scheduler, dataset)

if __name__ == "__main__":
    main()
