import hashlib
import json
import logging
from pathlib import Path
import random
import time
import numpy as np
import torch
from config import get_config
from data import build_loaders
from network import CaDH
from loss import CaDHLoss
from save_mat import Save_mat


@torch.no_grad()
def encode(model,loader,device):
    model.eval()
    result={k:[] for k in ['codes','credibility','labels']}
    for images,labels,_ in loader:
        pred=model(images.to(device))
        for key in ['codes','credibility']:result[key].append(pred[key].cpu())
        result['labels'].append(labels.cpu().clone())
    return {k:torch.cat(v) for k,v in result.items()}


def train(c):
    random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed'])
    if torch.cuda.is_available():torch.cuda.manual_seed_all(c['seed'])
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if c['device']=='auto' else c['device']
    c['device']=device
    out=Path(c['output_root'])/c['backbone']/c['dataset']/str(c['n_bits'])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {out}; choose another --output-root')
    out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',force=True,
        handlers=[logging.FileHandler(out/'train.log',encoding='utf-8'),logging.StreamHandler()])
    # Direct public metric call, including NUS-WIDE @5000 and ImageNet @1000.
    from common.hashzoo_utils import mean_average_precision
    from common.hashzoo_data import get_topk
    topk=None if c['synthetic'] else get_topk(c['dataset'])
    c.update(topk=topk,method='CaDH',modules=['DCL','USM'],
        credibility_method='FUME original competition; multilabel label-guided training FML',
        checkpoint_selection='maximum public Hash-Zoo Hamming mAP',detach_uncertainty=True)
    (out/'config.json').write_text(json.dumps(c,indent=2),encoding='utf-8')
    logging.info('START %s',json.dumps(c))
    train_loader,query,database,classes=build_loaders(c)
    model=CaDH(classes,c['n_bits'],c['backbone'],not c['no_pretrained']).to(device)
    objective=CaDHLoss(c).to(device)
    optimizer=torch.optim.Adam([
        {'params':model.backbone.parameters(),'lr':c['backbone_lr']},
        {'params':list(model.hash_head.parameters())+list(model.credibility_head.parameters()),'lr':c['head_lr']}],
        weight_decay=c['weight_decay'])
    digest=hashlib.sha256()
    for value in model.state_dict().values():digest.update(value.detach().cpu().numpy().tobytes())
    logging.info('initial_model_sha256=%s',digest.hexdigest())
    best=-1.;best_path=None;bad_checks=0;best_epoch=None
    for epoch in range(1,c['n_epochs']+1):
        model.train();start=time.perf_counter();stats={};count=0;loss_sum=0.
        for images,y,_ in train_loader:
            images,y=images.to(device),y.to(device).float()
            loss,values=objective(model(images),y)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite joint loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            n=len(images);count+=n;loss_sum+=float(loss.detach())*n
            for k,v in values.items():
                stats[k]=stats.get(k,0)+v*(1 if k in ['triplets','active_triplets'] else n)
        record={'epoch':epoch,'epochs':c['n_epochs'],'train_seconds':time.perf_counter()-start,'loss':loss_sum/count,
            **{k:v/(1 if k in ['triplets','active_triplets'] else count) for k,v in stats.items()}}
        stop=False
        logging.info('[Training][dataset:%s][bits:%d][epoch:%d/%d][time:%.3fs][loss:%.4f][triplet_loss:%.4f][dcl_loss:%.4f][quant_loss:%.4f][active_triplets:%d]',
            c['dataset'],c['n_bits'],epoch,c['n_epochs'],record['train_seconds'],record['loss'],
            record['triplet_loss'],record['fcl_loss'],record['quant_loss'],int(record['active_triplets']))
        if epoch%c['eval_interval']==0 or epoch==c['n_epochs']:
            logging.info('[Evaluating][dataset:%s][bits:%d][epoch:%d][metric:%s]',
                c['dataset'],c['n_bits'],epoch,'mAP' if topk is None else f'mAP@{topk}')
            tic=time.perf_counter()
            rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,train_loader.generator.get_state())
            q,d=encode(model,query,device),encode(model,database,device)
            score=float(mean_average_precision(q['codes'],d['codes'],q['labels'],d['labels'],topk=topk))
            random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
            if rng[3] is not None:torch.cuda.set_rng_state_all(rng[3])
            train_loader.generator.set_state(rng[4])
            if score>best:
                best=score;bad_checks=0;best_epoch=epoch
                target=out/f'e{epoch}_{score:.6f}.pth'
                payload={'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                    'config':c,'mAP':score,'query':q,'database':d,'initial_model_sha256':digest.hexdigest()}
                temporary=out/'checkpoint.tmp'
                torch.save(payload,temporary);temporary.replace(target)
                if best_path is not None and best_path!=target:best_path.unlink()
                best_path=target
                mat_path=Save_mat(epoch,c['n_bits'],c['dataset'],q['labels'],d['labels'],q['codes'].sign(),d['codes'].sign(),save_dir=str(out),mode_name='CaDH',map=score)
                logging.info('[Best MAT saved] %s',mat_path)
            else:bad_checks+=1
            record.update(mAP=score,topk=topk,best_mAP=best,evaluation_seconds=time.perf_counter()-tic,
                best_checkpoint=str(best_path),no_improvement_checks=bad_checks)
            stop=c['early_stop_patience']>0 and bad_checks>=c['early_stop_patience']
            logging.info('[Validation][dataset:%s][bits:%d][epoch:%d][mAP:%.6f][best_mAP:%.6f][best_epoch:%d][time:%.3fs][early_stop:%d/%d]',
                c['dataset'],c['n_bits'],epoch,score,best,best_epoch,record['evaluation_seconds'],bad_checks,c['early_stop_patience'])
        if stop:
            logging.info('EARLY STOP');break
    logging.info('[Complete][best_mAP:%.6f][best_epoch:%d][checkpoint:%s]',best,best_epoch,best_path)
    return best_path


if __name__=='__main__':
    train(get_config())

