"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import time
import json
import datetime
import math

import torch

from ..misc import dist_utils, stats

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from ..optim.lr_scheduler import FlatCosineLRScheduler


class DetSolver(BaseSolver):
    @staticmethod
    def _per_class_ap50_from_coco(coco_evaluator, hard_class_names=None):
        if coco_evaluator is None or "bbox" not in getattr(coco_evaluator, "coco_eval", {}):
            return {}, None
        coco_eval = coco_evaluator.coco_eval["bbox"]
        if coco_eval.eval is None or "precision" not in coco_eval.eval:
            return {}, None

        precision = coco_eval.eval["precision"]
        params = coco_eval.params
        try:
            iou_idx = int(min(range(len(params.iouThrs)), key=lambda i: abs(float(params.iouThrs[i]) - 0.5)))
            area_idx = list(params.areaRngLbl).index("all") if "all" in params.areaRngLbl else 0
            maxdet_idx = len(params.maxDets) - 1
        except Exception:
            return {}, None

        cat_id_to_name = {
            int(cat_id): coco_eval.cocoGt.cats[int(cat_id)].get("name", str(cat_id))
            for cat_id in params.catIds
            if int(cat_id) in coco_eval.cocoGt.cats
        }
        per_class = {}
        for k, cat_id in enumerate(params.catIds):
            values = precision[iou_idx, :, k, area_idx, maxdet_idx]
            values = values[values > -1]
            if values.size:
                per_class[cat_id_to_name.get(int(cat_id), str(cat_id))] = float(values.mean())

        hard_names = list(hard_class_names or [])
        hard_values = [per_class[name] for name in hard_names if name in per_class]
        hard_mean = float(sum(hard_values) / len(hard_values)) if hard_values else None
        return per_class, hard_mean

    @staticmethod
    def _is_better_v2_1(current_ap50, current_hard, best_ap50, best_hard, tie_threshold):
        if best_ap50 is None:
            return True
        if current_ap50 > best_ap50 + tie_threshold:
            return True
        if abs(current_ap50 - best_ap50) <= tie_threshold:
            return (current_hard if current_hard is not None else -1.0) > (best_hard if best_hard is not None else -1.0)
        return False

    def fit(self, ):
        self.train()
        args = self.cfg

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-"*42 + "Start training" + "-"*43)

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, args.lr_gamma, iter_per_epoch, total_epochs=args.epoches,
                                                warmup_iter=args.warmup_iter, flat_epochs=args.flat_epoch, no_aug_epochs=args.no_aug_epoch)
            self.self_lr_scheduler = True
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        top1 = 0
        best_stat = {'epoch': -1, }
        best_v2_1 = {'epoch': -1, 'ap50': None, 'hard_class_mean_ap50': None}
        best_select_metric = getattr(args, 'best_select_metric', None)
        hard_class_names = getattr(args, 'hard_class_names', [])
        hard_tie_threshold = float(getattr(args, 'hard_class_tie_threshold', 0.003))
        best_ap50_checkpoint_name = getattr(args, 'best_ap50_checkpoint_name', None) or 'best_v2_1.pth'
        best_hardclass_checkpoint_name = getattr(args, 'best_hardclass_checkpoint_name', None)
        best_meta_name = getattr(args, 'best_meta_name', None) or 'best_v2_1.json'
        early_stopping_patience = getattr(args, 'early_stopping_patience', None)
        early_stopping_patience = int(early_stopping_patience) if early_stopping_patience else None
        epochs_since_metric_best = 0
        should_stop_early = False
        best_hardclass = {'epoch': -1, 'hard_class_mean_ap50': None, 'ap50': None}
        # evaluate again before resume training
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )
            for k in test_stats:
                best_stat['epoch'] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]
                print(f'best_stat: {best_stat}')

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            train_stats, grad_percentages = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                teacher_model=self.teacher_model, # NEW: Pass teacher model to train_one_epoch
                accumulation_steps=getattr(args, 'accumulation_steps', 1),
            )

            if not self.self_lr_scheduler:  # update by epoch 
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch = epoch
            if dist_utils.is_main_process() and hasattr(self.criterion, 'distill_adaptive_params') and \
                self.criterion.distill_adaptive_params and self.criterion.distill_adaptive_params.get('enabled', False):

                params = self.criterion.distill_adaptive_params
                default_weight = params.get('default_weight')

                avg_percentage = sum(grad_percentages) / len(grad_percentages) if grad_percentages else 0.0

                current_weight = self.criterion.weight_dict.get('loss_distill', 0.0)
                new_weight = current_weight
                reason = 'unchanged'

                if avg_percentage < 1e-6:
                    if default_weight is not None:
                        new_weight = default_weight
                        reason = 'reset_to_default_zero_grad'
                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    if default_weight is not None:
                        new_weight = default_weight
                        reason = 'ema_phase_default'
                elif 'rho' in params and 'delta' in params:
                    rho = params['rho']
                    delta = params['delta']
                    lower_bound = rho - delta
                    upper_bound = rho + delta
                    if not (lower_bound <= avg_percentage <= upper_bound):
                        target_percentage = upper_bound if avg_percentage < lower_bound else lower_bound
                        if current_weight > 1e-6:
                            p_current = avg_percentage / 100.0
                            p_target = target_percentage / 100.0
                            numerator = p_target * (1.0 - p_current)
                            denominator = p_current * (1.0 - p_target)
                            if abs(denominator) >= 1e-9:
                                ratio = numerator / denominator
                                ratio = max(ratio, 0.1)  # clamp non-positive to 0.1
                                new_weight = current_weight * ratio
                                new_weight = min(max(new_weight, current_weight / 10.0), current_weight * 10.0)
                                reason = f'adjusted_to_{target_percentage:.2f}%'
                else:
                    reason = 'foreground_params_only'

                if abs(new_weight - current_weight) > 0:
                    self.criterion.weight_dict['loss_distill'] = new_weight
                print(f"Epoch {epoch}: avg encoder grad {avg_percentage:.2f}% | distill {current_weight:.6f} -> {new_weight:.6f} ({reason})")

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'last.pth']
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )
            per_class_ap50, hard_class_mean_ap50 = self._per_class_ap50_from_coco(coco_evaluator, hard_class_names)

            # TODO
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat[k] > top1:
                    best_stat_print['epoch'] = epoch
                    top1 = best_stat[k]
                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                        else:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')

                best_stat_print[k] = max(best_stat[k], top1)
                print(f'best_stat: {best_stat_print}')  # global best

                if best_stat['epoch'] == epoch and self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if test_stats[k][0] > top1:
                            top1 = test_stats[k][0]
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                    else:
                        top1 = max(test_stats[k][0], top1)
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')

                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    best_stat = {'epoch': -1, }
                    self.ema.decay -= 0.0001
                    self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                    print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            metric_improved = False
            if (
                best_select_metric in ('ap50_hard_tie', 'ap50_hard_tie_v2_2')
                and 'coco_eval_bbox' in test_stats
                and len(test_stats['coco_eval_bbox']) > 1
            ):
                current_ap50 = float(test_stats['coco_eval_bbox'][1])
                current_map = float(test_stats['coco_eval_bbox'][0])
                current_ap75 = float(test_stats['coco_eval_bbox'][2])
                if self._is_better_v2_1(
                    current_ap50,
                    hard_class_mean_ap50,
                    best_v2_1['ap50'],
                    best_v2_1['hard_class_mean_ap50'],
                    hard_tie_threshold,
                ):
                    best_v2_1 = {
                        'epoch': epoch,
                        'mAP': current_map,
                        'ap50': current_ap50,
                        'ap75': current_ap75,
                        'hard_class_mean_ap50': hard_class_mean_ap50,
                        'per_class_ap50': per_class_ap50,
                        'coco_eval_bbox': test_stats['coco_eval_bbox'],
                    }
                    if self.output_dir:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / best_ap50_checkpoint_name)
                        if dist_utils.is_main_process():
                            with (self.output_dir / best_meta_name).open('w') as f:
                                json.dump(best_v2_1, f, indent=2, ensure_ascii=False)
                    metric_improved = True
                    print(f"best_custom_ap50: {best_v2_1}")

                if (
                    best_hardclass_checkpoint_name
                    and hard_class_mean_ap50 is not None
                    and (
                        best_hardclass['hard_class_mean_ap50'] is None
                        or hard_class_mean_ap50 > best_hardclass['hard_class_mean_ap50']
                    )
                ):
                    best_hardclass = {
                        'epoch': epoch,
                        'mAP': current_map,
                        'ap50': current_ap50,
                        'ap75': current_ap75,
                        'hard_class_mean_ap50': hard_class_mean_ap50,
                        'per_class_ap50': per_class_ap50,
                        'coco_eval_bbox': test_stats['coco_eval_bbox'],
                    }
                    if self.output_dir:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / best_hardclass_checkpoint_name)
                        if dist_utils.is_main_process():
                            hard_meta_name = best_hardclass_checkpoint_name.rsplit('.', 1)[0] + '.json'
                            with (self.output_dir / hard_meta_name).open('w') as f:
                                json.dump(best_hardclass, f, indent=2, ensure_ascii=False)
                    metric_improved = True
                    print(f"best_hardclass: {best_hardclass}")

            if early_stopping_patience is not None:
                epochs_since_metric_best = 0 if metric_improved else epochs_since_metric_best + 1
                if epochs_since_metric_best >= early_stopping_patience:
                    print(
                        f"Early stopping at epoch {epoch}: no custom metric improvement "
                        f"for {epochs_since_metric_best} epochs."
                    )
                    should_stop_early = True

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'test_per_class_ap50': per_class_ap50,
                'test_hard_class_mean_ap50': hard_class_mean_ap50,
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

            if should_stop_early:
                break

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return


    def state_dict(self):
        """State dict, train/eval"""
        state = {}
        state['date'] = datetime.datetime.now().isoformat()

        # For resume
        state['last_epoch'] = self.last_epoch

        for k, v in self.__dict__.items():
            if k == 'teacher_model':
                continue
            if hasattr(v, 'state_dict'):
                v = dist_utils.de_parallel(v)
                state[k] = v.state_dict()

        return state
