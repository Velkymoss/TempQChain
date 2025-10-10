import os
import random
import warnings

import numpy as np
import torch
import tqdm
import transformers
from domiknows.program.model.base import Mode
from sklearn.metrics import accuracy_score, f1_score

from tempQchain.graphs.graph_tb_dense_FR import (
    after,
    before,
    includes,
    is_included,
    simultaneous,
    vague,
)
from tempQchain.logger import get_logger
from tempQchain.programs.program_tb_dense_FR import (
    program_declaration_tb_dense_fr,
    program_declaration_tb_dense_fr_T5,
    program_declaration_tb_dense_fr_T5_v2,
    program_declaration_tb_dense_fr_T5_v3,
)
from tempQchain.readers.temporal_reader import TemporalReader

warnings.filterwarnings("ignore")

logger = get_logger(__name__)


def eval(program, testing_set, cur_device, args):
    all_labels = [
        before,
        after,
        includes,
        is_included,
        simultaneous,
        vague,
    ]

    all_labels_text = [
        "before",
        "after",
        "includes",
        "is_included",
        "simultaneous",
        "vague",
    ]

    # def remove_opposite(ind1, ind2, result_set, result_list):
    #     if ind1 in pred_set and ind2 in pred_set:
    #         if result_list[ind1] > result_list[ind2]:
    #             result_set.remove(ind2)
    #         else:
    #             result_set.remove(ind1)

    pred_list = []
    pred_set = set()

    all_true = []
    all_pred = []

    for datanode in tqdm.tqdm(program.populate(testing_set, device=cur_device), "Checking accuracy"):
        for question in datanode.getChildDataNodes():
            pred_set.clear()
            pred_list.clear()
            # Getting predict label
            for ind, label in enumerate(all_labels):
                pred = question.getAttribute(label, "local/softmax")
                if pred.argmax().item() == 1:
                    pred_set.add(ind)
                pred_list.append(pred[1].item())

            # remove_opposite(0, 1, pred_set, pred_list)
            # remove_opposite(2, 3, pred_set, pred_list)
            # remove_opposite(4, 5, pred_set, pred_list)
            # Getting acutal label
            # if args.model == "t5-adapter":
            #     expected_text = question.getAttribute("text_labels")
            #     pred_text = ""
            #     for i, label in enumerate(all_labels_text):
            #         if multilabel:
            #             pred_text += label + ":" + ("yes" if i in pred_set else "no") + " "
            #         else:
            #             if i in pred_set:
            #                 pred_text += label if not pred_text else (", " + label)
            #     correct += int(expected_text.strip() == pred_text.strip())
            # else:
            true_labels = []
            pred_labels = []
            for ind, label_ind in enumerate(all_labels):
                label = question.getAttribute(label_ind, "label").item()
                pred = 1 if ind in pred_set else 0
                true_labels.append(label)
                pred_labels.append(pred)
                
            all_true.append(true_labels)
            all_pred.append(pred_labels)
        
    accuracy = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average="weighted")

    logger.info(f"Program: {'Primal Dual' if args.pmd else 'Sampling Loss' if args.sampling else 'DomiKnowS'}")
    if not args.loaded:
        logger.info("Training info")
        logger.info(f"Batch Size: {args.batch_size}")
        logger.info(f"Epoch: {args.epoch}")
        logger.info(f"Learning Rate: {args.lr}")
        logger.info(f"Beta: {args.beta}")
        logger.info(f"Sampling Size: {args.sampling_size}")
    else:
        logger.info(f"Loaded Model Name: {args.loaded_file}")
    logger.info(f"Accuracy: {accuracy * 100}%")
    logger.info(f"F1 Score: {f1 * 100}%")

    return f1, accuracy


def train(program, train_set, eval_set, cur_device, limit, lr, check_epoch=1, program_name="DomiKnow", args=None):
    def get_avg_loss():
        if cur_device is not None:
            program.model.to(cur_device)
        program.model.mode(Mode.TEST)
        program.model.reset()
        train_loss = 0
        total_loss = 0
        with torch.no_grad():
            for data_item in tqdm.tqdm(train_set, "Calculating Loss of training"):
                loss, _, *output = program.model(data_item)
                total_loss += 1
                train_loss += loss
        return train_loss / total_loss

    best_accuracy = 0
    best_epoch = 0
    old_file = None
    check_epoch = args.check_epoch
    logger.info("-" * 10)
    logger.info(f"Training by {program_name} of ({args.train_file} FR)")
    logger.info(f"Learning Rate: {args.lr}")

    cur_epoch = 0
    if args.optim != "adamw":
        optimizer = lambda param: transformers.optimization.Adafactor(
            param, lr=lr, scale_parameter=False, relative_step=False
        )
    else:
        optimizer = lambda param: torch.optim.AdamW(param, lr=lr)
    for epoch in range(check_epoch, limit, check_epoch):
        logger.info("Training")
        if args.pmd:
            program.train(train_set, c_warmup_iters=0, train_epoch_num=check_epoch, Optim=optimizer, device=cur_device)
        else:
            program.train(train_set, train_epoch_num=check_epoch, Optim=optimizer, device=cur_device)
        cur_epoch += check_epoch
        loss = get_avg_loss()

        accuracy = eval(program, eval_set, cur_device, args)
        logger.info(f"Epoch: {epoch}")
        logger.info(f"Loss: {loss}")
        logger.info(f"Dev Accuracy: {accuracy * 100}%")
        if accuracy >= best_accuracy:
            best_epoch = epoch
            best_accuracy = accuracy
            # if old_file:
            #     os.remove(old_file)
            program_addition = ""
            if program_name == "PMD":
                program_addition = "_beta_" + str(args.beta)
            else:
                program_addition = "_size_" + str(args.sampling_size)
            new_file = (
                program_name
                + "_"
                + str(epoch)
                + "epoch"
                + "_lr_"
                + str(args.lr)
                + program_addition
                + "_model_"
                + args.model
            )
            program.save(os.path.join(args.results_path, new_file))

    if cur_epoch < limit:
        if args.pmd:
            program.train(train_set, c_warmup_iters=0, train_epoch_num=check_epoch, Optim=optimizer, device=cur_device)
        else:
            program.train(train_set, train_epoch_num=check_epoch, Optim=optimizer, device=cur_device)
        accuracy = eval(program, eval_set, cur_device, args)
        logger.info(f"Epoch: {limit}")
        logger.info(f"Dev Accuracy: {accuracy * 100}%")
        if accuracy >= best_accuracy:
            best_epoch = limit
            # if old_file:
            #     os.remove(old_file)
            if program_name == "PMD":
                program_addition = "_beta_" + str(args.beta)
            else:
                program_addition = "_size_" + str(args.sampling_size)
            new_file = (
                program_name
                + "_"
                + str(limit)
                + "epoch"
                + "_lr_"
                + str(args.lr)
                + program_addition
                + "_model_"
                + args.model
            )

            old_file = new_file
            program.save(os.path.join(args.results_path, new_file))

    logger.info(f"Best epoch {best_epoch}")
    return best_epoch


def main(args):
    SEED = 382
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    cuda_number = args.cuda
    if cuda_number == -1:
        cur_device = "cpu"
    else:
        cur_device = "cuda:" + str(cuda_number) if torch.cuda.is_available() else "cpu"

    if args.model == "t5-adapter":
        logger.info("call T5")
        program_declaration_function = None
        if args.version == 2:
            program_declaration_function = program_declaration_tb_dense_fr_T5_v2
        elif args.version == 3:
            program_declaration_function = program_declaration_tb_dense_fr_T5_v3
        elif args.version == 4:
            # program_declaration_function = program_declaration_tb_dense_fr_T5_v4
            raise NotImplementedError("Version 4 is not implemented yet.")
        elif args.version == 5:
            # program_declaration_function = program_declaration_tb_dense_fr_T5_v5
            raise NotImplementedError("Version 5 is not implemented yet.")
        else:
            program_declaration_function = program_declaration_tb_dense_fr_T5

        program = program_declaration_function(
            cur_device,
            pmd=args.pmd,
            beta=args.beta,
            sampling=args.sampling,
            sampleSize=args.sampling_size,
            dropout=args.dropout,
            constraintts=args.constraints,
        )
    else:
        program = program_declaration_tb_dense_fr(
            cur_device,
            pmd=args.pmd,
            beta=args.beta,
            sampling=args.sampling,
            sampleSize=args.sampling_size,
            dropout=args.dropout,
            constraints=args.constraints,
            model=args.model,
        )

    train_file = "tb_dense_train.json"
    training_set = TemporalReader.from_file(
        file_path=os.path.join(args.data_path, train_file), question_type="FR", batch_size=args.batch_size
    )[:1]

    test_file = "tb_dense_test.json"
    testing_set = TemporalReader.from_file(
        file_path=os.path.join(args.data_path, test_file), question_type="FR", batch_size=args.batch_size
    )[:1]

    eval_file = "tb_dense_dev.json"
    eval_set = TemporalReader.from_file(
        file_path=os.path.join(args.data_path, eval_file), question_type="FR", batch_size=args.batch_size
    )[:1]

    program_name = "PMD" if args.pmd else "Sampling" if args.sampling else "Base"

    if args.loaded:
        if args.model_change:
            pretrain_model = torch.load(
                os.path.join(args.results_path, args.loaded_file),
                map_location={
                    "cuda:0": cur_device,
                    "cuda:1": cur_device,
                    "cuda:2": cur_device,
                    "cuda:3": cur_device,
                    "cuda:4": cur_device,
                    "cuda:5": cur_device,
                },
            )
            pretrain_dict = pretrain_model.state_dict()
            current_dict = program.model.state_dict()
            # Filter out unnecessary keys
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in current_dict}
            program.model.load_state_dict(pretrain_dict)
        else:
            program.load(
                os.path.join(args.results_path, args.loaded_file),
                map_location={
                    "cuda:0": cur_device,
                    "cuda:1": cur_device,
                    "cuda:2": cur_device,
                    "cuda:3": cur_device,
                    "cuda:4": cur_device,
                    "cuda:5": cur_device,
                },
            )

        eval(program, testing_set, cur_device, args)

    elif args.loaded_train:
        if args.model_change:
            pretrain_model = torch.load(
                os.path.join(args.results_path, args.loaded_file),
                map_location={
                    "cuda:0": cur_device,
                    "cuda:1": cur_device,
                    "cuda:2": cur_device,
                    "cuda:3": cur_device,
                    "cuda:4": cur_device,
                    "cuda:5": cur_device,
                },
            )
            pretrain_dict = pretrain_model
            current_dict = program.model.state_dict()
            # Filter out unnecessary keys
            # pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in current_dict}
            # Loaded same parameters
            new_state_dict = {k: v if k not in pretrain_dict else pretrain_dict[k] for k, v in current_dict.items()}
            program.model.load_state_dict(new_state_dict)
        else:
            program.load(
                os.path.join(args.results_path, args.loaded_file),
                map_location={
                    "cuda:0": cur_device,
                    "cuda:1": cur_device,
                    "cuda:2": cur_device,
                    "cuda:3": cur_device,
                    "cuda:4": cur_device,
                    "cuda:5": cur_device,
                },
            )
        train(program, training_set, eval_set, cur_device, args.epoch, args.lr, program_name=program_name, args=args)
    else:
        train(program, training_set, eval_set, cur_device, args.epoch, args.lr, program_name=program_name, args=args)
