import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from collections import defaultdict
from utils.metrics import wer_list 
from utils.misc import *
from utils import clean_phoenix_2014_trans,clean_phoenix_2014,clean_csl
import gc


def seq_train(loader, model, optimizer, device, epoch_idx, recoder):
    model.train()
    loss_value = []
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]
    scaler = GradScaler()
    for batch_idx, data in enumerate(tqdm(loader)):
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        optimizer.zero_grad()
        with autocast():
            ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)
            if len(device.gpu_list)>1:
                loss = model.module.criterion_calculation(ret_dict, label, label_lgt)
            else:
                loss = model.criterion_calculation(ret_dict, label, label_lgt)

        if np.isinf(loss.item()) or np.isnan(loss.item()):
            print('loss is nan')
            print(str(data[1])+'  frames', str(data[3])+'  glosses')
            continue
        scaler.scale(loss).backward()
        scaler.step(optimizer.optimizer)
        scaler.update() 
        if len(device.gpu_list)>1:
            torch.cuda.synchronize() 
            torch.distributed.reduce(loss, dst=0)

        loss_value.append(loss.item())
        if batch_idx % recoder.log_interval == 0 and is_main_process():
            recoder.print_log(
                '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.6f}'
                    .format(epoch_idx, batch_idx, len(loader), loss.item(), clr[0]))
        del vid
        del vid_lgt
        del label
        del label_lgt
        del ret_dict
        del loss
    optimizer.scheduler.step()
    if is_main_process():
        recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))
    del loss_value
    del clr
    gc.collect()
    return


def seq_eval(cfg, loader, model, device, mode, epoch, work_dir, recoder, evaluate_tool="python"):
    model.eval()
    results=defaultdict(dict)
    for batch_idx, data in enumerate(tqdm(loader)):
        recoder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        info = [d['fileid'] for d in data[-1]]
        gloss = [d['label'] for d in data[-1]]
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)
            for inf, conv_sents, recognized_sents, gl in zip(info, ret_dict['conv_sents'], ret_dict['recognized_sents'], gloss):
                results[inf]['conv_sents'] = conv_sents
                results[inf]['recognized_sents'] = recognized_sents
                results[inf]['gloss'] = gl
        del vid
        del vid_lgt
        del label
        del label_lgt
        del ret_dict

    if cfg.dataset=='phoenix2014':
        gls_hyp1 = [clean_phoenix_2014(' '.join(results[n]['conv_sents'])) for n in results]
        gls_ref = [clean_phoenix_2014(results[n]['gloss']) for n in results]
        gls_hyp2 = [clean_phoenix_2014(' '.join(results[n]['recognized_sents'])) for n in results]
    elif cfg.dataset=='phoenix2014-T':
        gls_hyp1 = [clean_phoenix_2014_trans(' '.join(results[n]['conv_sents'])) for n in results]
        gls_ref = [clean_phoenix_2014_trans(results[n]['gloss']) for n in results]
        gls_hyp2 = [clean_phoenix_2014_trans(' '.join(results[n]['recognized_sents'])) for n in results]
    else:
        gls_hyp1 = [clean_csl(' '.join(results[n]['conv_sents'])) for n in results]
        gls_ref = [clean_csl(results[n]['gloss']) for n in results]
        gls_hyp2 = [clean_csl(' '.join(results[n]['recognized_sents'])) for n in results]
    wer_results_con = wer_list(hypotheses=gls_hyp1, references=gls_ref)
    wer_results = wer_list(hypotheses=gls_hyp2, references=gls_ref)
    if epoch==6667:
        name = [n for n in results]
        with open(f"{work_dir}/{mode}_visual.txt", "w") as file:
            for item in range(len(gls_ref)):
                file.write('fileid  :  '+name[item] + "\n")
                file.write('GT  :  '+gls_ref[item] + "\n")
                file.write('CNN :  '+gls_hyp1[item] +"   "+ str(wer_list(hypotheses=[gls_hyp1[item]], references=[gls_ref[item]])['wer']) +"\n")
                file.write('LSTM:  '+gls_hyp2[item] +"   "+ str(wer_list(hypotheses=[gls_hyp2[item]], references=[gls_ref[item]])['wer']) +"\n\n")
    if wer_results['wer'] < wer_results_con['wer']:
        reg_per = wer_results
    else:
        reg_per = wer_results_con
    recoder.print_log('\tEpoch: {} {} done. Conv wer: {:.4f}  ins:{:.4f}, del:{:.4f}'.format(
        epoch, mode, wer_results_con['wer'], wer_results_con['ins'], wer_results_con['del']),
        f"{work_dir}/{mode}.txt")
    recoder.print_log('\tEpoch: {} {} done. LSTM wer: {:.4f}  ins:{:.4f}, del:{:.4f}'.format(
        epoch, mode, wer_results['wer'], wer_results['ins'], wer_results['del']), f"{work_dir}/{mode}.txt")
    gc.collect()
    return {"wer":reg_per['wer'], "ins":reg_per['ins'], 'del':reg_per['del']}
 
