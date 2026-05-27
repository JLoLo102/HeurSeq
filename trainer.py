import gc

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from tools.utils import *
from sklearn.metrics import f1_score, average_precision_score, precision_score
from sklearn.preprocessing import label_binarize



class ModelNetTrainer(object):
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, model_name, log_dir, num_views=6):

        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.log_dir = log_dir
        self.num_views = num_views

        self.model.cuda()
        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir)

    def train(self, n_epochs):

        best_top5 = 0
        i_acc = 0
        self.model.train()
        for epoch in range(n_epochs):

            # plot learning rate
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.writer.add_scalar('params/lr', lr, epoch)

            # train one epoch

            for i, data in enumerate(self.train_loader):
                if self.model_name == 'HeurSeq':
                    
                    N, V, C, H, W = data[1].size()
                    in_data = Variable(data[1]).view(-1, C, H, W).cuda()
                else:
                    in_data = Variable(data[1].cuda())
                target = Variable(data[0]).cuda().long()

                transposed = list(zip(*data[2]))
                # transposed=data[2]

                self.optimizer.zero_grad()


                out_data, loss_cs, loss_clip = self.model(in_data, transposed)
        
                loss_cls = self.loss_fn(out_data, target)

                loss = loss_cls + 0.1 * loss_cs + 0.1 * loss_clip




                self.writer.add_scalar('train/train_loss', loss, i_acc+i+1)

     

                # ======= 计算 Top-5 训练准确率 =======
                top5_vals, top5_idx = out_data.topk(5, dim=1, largest=True, sorted=True)
                correct_top5 = (top5_idx == target.unsqueeze(1)).any(dim=1).float()
                acc = correct_top5.mean()

  
                self.writer.add_scalar('train/train_top5_acc', acc, i_acc + i + 1)


                loss.backward()
                self.optimizer.step()

             
                log_str = f'epoch {epoch + 1}, step {i + 1}: train_loss {loss:.3f}; train_top5_acc {acc:.3f}'
                if (i + 1) % 100 == 0:
                    print(log_str)
                i_acc += i

            # evaluation
            if (epoch + 1) % 1 == 0:
                with torch.no_grad():
                    val_loss, metrics = self.update_validation_accuracy(epoch)
                if metrics['Top-5'] > best_top5:
                    best_top5 = metrics['Top-5']
                    torch.save(self.model.state_dict(), self.log_dir + "/best_model.pth")
                    print(
                        f"[Epoch {epoch + 1}] Best model saved with Top-5: {best_top5:.4f}, "
                        f"Top-10: {metrics['Top-10']:.4f}, Macro-F5: {metrics['Macro-F5']:.4f}, "
                        f"mAP: {metrics['mAP']:.4f}, Precision@5: {metrics['Precision@5']:.4f}")
                    read_output_Data(metrics, 'records.txt')

            # adjust learning rate manually and (epoch + 1) % M10 == 0
            if epoch > 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5

        gc.collect()
        if hasattr(torch, 'cuda'):
            try:
                torch.cuda.empty_cache()
            except:
                pass

        # export scalar entropy to JSON for external processing
        if self.log_dir is not None:
            self.writer.export_scalars_to_json(self.log_dir + "/all_scalars.json")
            self.writer.close()

    def update_validation_accuracy(self, epoch):
        """
                    更新验证集指标：Top-1/5/10 Accuracy, Macro-F1, mAP, Precision@5
                    """

        self.model.eval()

        all_targets = []
        all_preds = []
        all_probs = []

        topk = (5, 10)
        correct_topk = {k: 0 for k in topk}
        total_samples = 0
        all_loss = 0.0


        for _, data in enumerate(self.val_loader, 0):
            if self.model_name == 'HeurSeq':
                N, V, C, H, W = data[1].size()
                in_data = data[1].view(-1, C, H, W).cuda()
            else:
                in_data = data[1].cuda()

            target = data[0].cuda()

            transposed = list(zip(*data[2]))

            out_data,_,_ = self.model(in_data,transposed)



            # softmax 概率
            probs = torch.softmax(out_data, dim=1).detach().cpu().numpy()
            preds = out_data.argmax(dim=1).detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()

            all_targets.extend(target_np)
            all_preds.extend(preds)
            all_probs.append(probs)

            # loss

            loss_cls = self.loss_fn(out_data, target).detach().cpu().numpy()

            all_loss += loss_cls

            # Top-k
            topk_vals, topk_idxs = out_data.topk(max(topk), dim=1, largest=True, sorted=True)
            for k in topk:
                correct_topk[k] += (topk_idxs[:, :k] == target.unsqueeze(1)).any(dim=1).sum().item()
            total_samples += target.size(0)

        # 转 numpy
        all_probs = np.concatenate(all_probs, axis=0)
        all_targets = np.array(all_targets)
        num_classes = all_probs.shape[1]
        one_hot_labels = label_binarize(all_targets, classes=np.arange(num_classes))

        # 计算指标
        topk_acc = {f'Top-{k}': correct_topk[k] / total_samples for k in topk}

        # ==== Macro-F5 (Top-5 F1) ====
        top5_preds = np.argsort(all_probs, axis=1)[:, -5:]
        multi_label_preds = np.zeros_like(all_probs)
        for i, top5 in enumerate(top5_preds):
            multi_label_preds[i, top5] = 1
        macro_f5 = f1_score(one_hot_labels, multi_label_preds, average='macro')

        val_map = average_precision_score(one_hot_labels, all_probs, average='macro')

        # Precision@5
        top5_idx = np.argsort(all_probs, axis=1)[:, -5:]
        # precision_at5 = np.mean([target in top5 for target, top5 in zip(all_targets, top5_idx)])
        hits = np.array([target in top5 for target, top5 in zip(all_targets, top5_idx)])
        precision_at5 = hits.sum() / (len(all_targets) * 5)


        metrics = {**topk_acc, 'Macro-F5': macro_f5, 'mAP': val_map, 'Precision@5': precision_at5}
        val_loss = all_loss / len(self.val_loader)

        # 写入 TensorBoard
        for k, v in metrics.items():
            self.writer.add_scalar(f'val/{k}', float(v), epoch + 1)
        self.writer.add_scalar('val/loss', float(val_loss), epoch + 1)

        print(f"[Epoch {epoch + 1}] Loss: {val_loss:.4f}, "
              f"Top-5: {metrics['Top-5']:.4f}, Top-10: {metrics['Top-10']:.4f}, "
              f"Macro-F5: {metrics['Macro-F5']:.4f}, mAP: {metrics['mAP']:.4f}, Precision@5: {metrics['Precision@5']:.4f}")

        # 清理
        del all_probs, one_hot_labels
        gc.collect()

        self.model.train()

        return val_loss, metrics

