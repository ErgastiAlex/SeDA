import os
from pathlib import Path
import time
import datetime
import argparse
import json
import math
import random
import numpy as np
from ruamel.yaml import YAML
yaml = YAML(typ='safe')
from prettytable import PrettyTable

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer

from transformers import BertTokenizer

import utils
from models.model_search import Search

from dataset import create_dataset, create_sampler, create_loader
from dataset.search_dataset import TextMaskingGenerator
from scheduler import create_scheduler
from optim import create_optimizer

from train import train_model
from eval import evaluation_itm, evaluation_itc, mAP


def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    world_size = utils.get_world_size()

    if args.bs > 0:
        config['batch_size_train'] = args.bs
    if args.epo > 0:
        config['schedular']['epochs'] = args.epo
    if args.lr > 0:
        config['optimizer']['lr'] = args.lr
        config['schedular']['lr'] = args.lr

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    
    print("DECOUPLE TRAINING!!!")
    
    print("### output_dir:", args.output_dir)
    
    print("### Creating model")
    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
    model = Search(config=config)
    if config['load_pretrained']:
        model.load_pretrained(args.checkpoint)
    model = model.to(device)
    print("Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    print("### Creating search dataset")
    train_dataset, test_dataset1, test_dataset2 = create_dataset(config, args.evaluate)

    if config.get('small_dataset', False) and not args.evaluate:
        print("### Using a small subset of the dataset for fast debugging")
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        selected_indices = indices[:indices.__len__() // 10]
        train_dataset = torch.utils.data.Subset(train_dataset, selected_indices)

    start_time = time.time()

    if args.evaluate:
        print("### Start evaluating")
        test_loader1, test_loader2 = create_loader([test_dataset1, test_dataset2], [None, None],
                                    batch_size=[config['batch_size_test']] * 2,
                                    num_workers=[config.get('num_workers', 4)] * 2,
                                    is_trains=[False, False],
                                    collate_fns=[None, None])
        sims_matrix_t2i, image_embeds, text_embeds, text_atts = evaluation_itc(
                model_without_ddp, test_loader1, tokenizer, device, config)
        score_test_t2i_1 = evaluation_itm(model_without_ddp, device, config, args,
                                        sims_matrix_t2i, image_embeds, text_embeds, text_atts,)
        del sims_matrix_t2i, image_embeds, text_embeds, text_atts
        
        if utils.is_main_process():
            print('evaluating result:')
            mAP(score_test_t2i_1, test_loader1.dataset.g_pids, test_loader1.dataset.q_pids)

        #evaluate 2
        sims_matrix_t2i, image_embeds, text_embeds, text_atts = evaluation_itc(
            model_without_ddp, test_loader2, tokenizer, device, config)
        score_test_t2i_2 = evaluation_itm(model_without_ddp, device, config, args,
                                        sims_matrix_t2i, image_embeds, text_embeds, text_atts,)
        del sims_matrix_t2i, image_embeds, text_embeds, text_atts
        
        if utils.is_main_process():
            print('evaluating result:')
            mAP(score_test_t2i_2, test_loader2.dataset.g_pids, test_loader2.dataset.q_pids)
        dist.barrier()

    else:
        print("### Start training")
        train_dataset_size = len(train_dataset)
        if utils.is_main_process():
            print(f"### data {train_dataset_size}, batch size, {config['batch_size_train']} x {world_size}")
            tbs = ["epoch", "R1", "R5", "R10", "mAP", "mINP"]
            table = PrettyTable(tbs)
            for tb in tbs[1:]:
                table.custom_format[tb] = lambda f, v: f"{v:.3f}"

        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
        else:
            samplers = [None, None, None]

        train_loader, test_loader1, test_loader2 = create_loader([train_dataset, test_dataset1, test_dataset2], samplers,
                                                  batch_size=[config['batch_size_train']] + [config['batch_size_test']]*2,
                                                  num_workers=[4, 4, 4], is_trains=[True, False, False], collate_fns=[None, None, None])

        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model_without_ddp)
        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size / (config['batch_size_train'] * world_size))
        lr_scheduler = create_scheduler(arg_sche, optimizer)
        scaler = GradScaler()  # bf16

        mask_generator = TextMaskingGenerator(tokenizer, config['mask_prob'], config['max_masks'],
                                              config['skipgram_prb'], config['skipgram_size'],
                                              config['mask_whole_word'])
        best1 = 0
        best2 = 0
        best_epoch = 0
        max_epoch = config['schedular']['epochs']
        for epoch in range(0, max_epoch):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_stats = train_model(model, train_loader, optimizer, scaler, tokenizer, epoch,
                                       device, lr_scheduler, config, mask_generator)
            #dummy

            #evaluate 1
            sims_matrix_t2i, image_embeds, text_embeds, text_atts = evaluation_itc(
                model_without_ddp, test_loader1, tokenizer, device, config)
            score_test_t2i_1 = evaluation_itm(model_without_ddp, device, config, args,
                                            sims_matrix_t2i, image_embeds, text_embeds, text_atts,)
            del sims_matrix_t2i, image_embeds, text_embeds, text_atts

            #evaluate 2
            sims_matrix_t2i, image_embeds, text_embeds, text_atts = evaluation_itc(
                model_without_ddp, test_loader2, tokenizer, device, config)
            score_test_t2i_2 = evaluation_itm(model_without_ddp, device, config, args,
                                            sims_matrix_t2i, image_embeds, text_embeds, text_atts,)
            del sims_matrix_t2i, image_embeds, text_embeds, text_atts

            if utils.is_main_process():
                test_result1 = mAP(score_test_t2i_1, test_loader1.dataset.g_pids, test_loader1.dataset.q_pids, table)
                table.add_row([f"{epoch} (PAB)", test_result1['R1'], test_result1['R5'], test_result1['R10'],
                               test_result1['mAP'], test_result1['mINP']])
                
                test_result2 = mAP(score_test_t2i_2, test_loader2.dataset.g_pids, test_loader2.dataset.q_pids)
                table.add_row([f"{epoch} (OOD)", test_result2['R1'], test_result2['R5'], test_result2['R10'],
                               test_result2['mAP'], test_result2['mINP']])
            
                print(table)

                logs = {'epo': epoch}
                for k, v in test_result1.items():
                    logs[k] = np.around(v, 3)
                for k, v in train_stats.items():
                    logs[k] = float(v)
                print('logs', logs)

                for k, v in logs.items():
                    logs[k] = str(v)
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(logs) + "\n")

                logs_odd = {'epo': epoch}
                for k, v in test_result2.items():
                    logs_odd[k] = np.around(v, 3)
                for k, v in train_stats.items():
                    logs_odd[k] = float(v)
                print('logs_ood', logs_odd)
                for k, v in logs_odd.items():
                    logs_odd[k] = str(v)
                with open(os.path.join(args.output_dir, "log_ood.txt"), "a") as f:
                    f.write(json.dumps(logs_odd) + "\n")

                result1 = test_result1['R1']
                if result1 > best1:
                    save_obj = {'model': model_without_ddp.state_dict(), 'config': config, }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                    best1 = result1
                    best_epoch = epoch
                
                result2 = test_result2['R1']
                if result2 > best2:
                    save_obj = {'model': model_without_ddp.state_dict(), 'config': config, }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best_ood.pth'))
                    best2 = result2

            dist.barrier()
            torch.cuda.empty_cache()

        if utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write("best epoch: %d" % best_epoch)
            print("### best epoch: %d" % best_epoch)
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('### Time {}'.format(total_time_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--bs', default=0, type=int, help="mini batch size")
    parser.add_argument('--epo', default=0, type=int, help="epoch")
    parser.add_argument('--lr', default=0.0, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config = yaml.load(open(args.config, 'r'))
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
