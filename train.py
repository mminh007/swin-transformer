import torch
import os
from torch import optim as optim
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from tqdm import tqdm
import logging
import datetime
from data.build import build_dataloader
from transformer.model import build_model
from config import setup_parse, update_config
import time



def main(args):
    
    train_set, test_set = build_dataloader(args)

    log_name = "log" + f"_{args.mode}_{args.model}_{args.data}.log"
    logging.basicConfig(filename=log_name, level=logging.INFO)

    logger = logging.getLogger()
    logger.info(f"Create model: {args.model}_{args.name}_{args.data}")

    model = build_model(args)

    device = torch.device(args.devices if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.optimizer is not None:
        if args.optimizer == "Adam":
            optimizer = optim.Adam(model.parameters(), lr = args.lr, betas= [args.beta1, args.beta2],
                                   weight_decay=args.weight_decay)
        
        elif args.optimizer == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr = args.lr, betas= [args.beta1, args.beta2],
                                    weight_decay= args.weight_decay, eps= args.eps)
         
    else:
        optimizer = optim.AdamW(model.parameters(), lr = args.lr, betas= [args.beta1, args.beta2],
                                    weight_decay= args.weight_decay, eps= args.eps)

    if args.dtype.lower() == "bf16":
        dtype = torch.bfloat16
    elif args.dtype.lower() == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    criterion = torch.nn.CrossEntropyLoss()

    logger.info("Start training")

    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        
        running_loss = 0.
        # last_loss = 0.
        for idx, (x, y) in enumerate(tqdm(train_set)):
            x = x.to(device).to(dtype)
            y = y.to(device).long() #  (,num_cl
            
            optimizer.zero_grad()
            outputs = model(x)

            loss = criterion(outputs, y)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # if idx % 1000 == 999:
            #     last_loss = running_loss / 1000
            #     running_loss = 0.
        
        model.eval()
        with torch.no_grad():
            val_loss = 0.
            acc = 0.
            for idx, (samples, targets) in enumerate(tqdm(test_set)):                
                samples = samples.to(device).to(dtype)           
                targets = targets.to(device).long() #  (,num_cl

                predicted = model(samples)  # (B, num_classes)
                loss = criterion(predicted, targets)

                val_loss += loss.item()
                outputs = torch.argmax(input= predicted, dim=1)  # B, num_classes -> , labels

                accuracy = torch.sum(targets == outputs).item() / targets.size(0)
                acc += accuracy

        avg_loss = running_loss / len(train_set)
        avg_vloss = val_loss / len(test_set)
        vacc = acc / len(test_set)

        logger.info(f"Epoch {epoch + 1}: Training Loss = {avg_loss:.4f}, Validation Loss = {avg_vloss:.4f}, Validation Accuracy = {vacc:.4f}")

    # save model
    save_path = os.path.join(args.outputs_dir, f"{args.model}_{args.name}_{args.data}.pth")
    logger.info(f"{save_path} saving...")
    torch.save(model.state_dict(), save_path)

    total_time = time.time() - start_time
    total_time_str = (str(datetime.timedelta(seconds=int(total_time))))
    logger.info(f"Tranining time {total_time_str}")



if __name__ == "__main__":
    parser = setup_parse()

    args = parser.parse_args()
    args = update_config(args)

    main(args)
