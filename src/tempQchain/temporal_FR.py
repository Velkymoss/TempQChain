import os
import random
import warnings
from datetime import datetime
from typing import Any

import mlflow
import numpy as np
import torch
import tqdm
import transformers
from domiknows.program.lossprogram import LearningBasedProgram
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
from tempQchain.utils import get_avg_loss

warnings.filterwarnings("ignore")

logger = get_logger(__name__)


def eval(
    program: LearningBasedProgram,
    test_set: list[dict[str, str]],
    cur_device: str,
    args: Any = None,
    log_hyperparams: bool = False,
    log_metrics: bool = False,
) -> tuple[float, float]:
    if args.loaded:
        logger.info(f"Loaded Model Name: {args.loaded_file}")

    all_labels = [
        before,
        after,
        includes,
        is_included,
        simultaneous,
        vague,
    ]

    pred_list = []
    pred_set = set()

    all_true = []
    all_pred = []

    for datanode in tqdm.tqdm(program.populate(test_set, device=cur_device), "Checking f1/accuracy..."):
        for question in datanode.getChildDataNodes():
            pred_set.clear()
            pred_list.clear()
            # Getting predict label
            for ind, label in enumerate(all_labels):
                pred = question.getAttribute(label, "local/softmax")
                if pred.argmax().item() == 1:
                    pred_set.add(ind)
                pred_list.append(pred[1].item())

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

    if log_hyperparams:
        logger.info(f"Program: {'Primal Dual' if args.pmd else 'Sampling Loss' if args.sampling else 'DomiKnowS'}")
        logger.info(f"Batch Size: {args.batch_size}")
        logger.info(f"Learning Rate: {args.lr}")
        logger.info(f"Beta: {args.beta}")
        logger.info(f"Sampling Size: {args.sampling_size}")

    if log_metrics:
        logger.info(f"Accuracy: {accuracy * 100}%")
        logger.info(f"F1 Score: {f1 * 100}%")

    return f1, accuracy


def train(
    program: LearningBasedProgram,
    train_set: list[dict[str, str]],
    eval_set: list[dict[str, str]],
    test_set: list[dict[str, str]] | None,
    cur_device: str | None,
    lr: float,
    program_name: str = "DomiKnow",
    args: Any = None,
) -> int:
    best_f1 = 0
    best_accuracy = 0
    best_epoch = 0

    logger.info("Starting FR training...")
    logger.info(f"Model: {args.model}")
    logger.info("Using Hyperparameters:")
    logger.info(f"Learning Rate: {args.lr}")
    logger.info(f"Batch Size: {args.batch_size}")
    if args.constraints:
        logger.info("Using Constraints")
    if args.dropout:
        logger.info("Using Dropout")
    if args.pmd:
        logger.info(f"Using Primal Dual Method with Beta: {args.beta}")
    if args.sampling:
        logger.info(f"Using Sampling Loss with Size: {args.sampling_size}")
    mlflow.log_params(
        {
            "model": args.model,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "pmd": args.pmd,
            "beta": args.beta if args.pmd else None,
            "epochs": args.epoch,
            "constraints": args.constraints,
            "sampling": args.sampling if args.sampling else None,
            "sampling_size": args.sampling_size if args.sampling else None,
            "dropout": args.dropout,
            "optimizer": args.optim,
            "version": args.version if args.model == "t5-adapter" else None,
        }
    )

    if args.optim != "adamw":

        def optimizer(param):
            return transformers.optimization.Adafactor(param, lr=lr, scale_parameter=False, relative_step=False)
    else:

        def optimizer(param):
            return torch.optim.AdamW(param, lr=lr)

    for epoch in range(1, args.epoch + 1):
        logger.info(f"Epoch {epoch}/{args.epoch}")
        if args.pmd:
            program.train(train_set, c_warmup_iters=0, train_epoch_num=1, Optim=optimizer, device=cur_device)
        else:
            program.train(train_set, train_epoch_num=1, Optim=optimizer, device=cur_device)

        train_loss = get_avg_loss(program, train_set, cur_device, "train")
        eval_loss = get_avg_loss(program, eval_set, cur_device, "eval")
        f1, accuracy = eval(program=program, test_set=eval_set, cur_device=cur_device, args=args)

        logger.info(f"Epoch: {epoch}")
        logger.info(f"Train Loss: {train_loss}")
        logger.info(f"Eval Loss: {eval_loss}")
        logger.info(f"Dev Accuracy: {accuracy * 100}%")
        logger.info(f"Dev F1: {f1 * 100}%")
        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "eval_f1": f1,
                "eval_accuracy": accuracy,
            },
            step=epoch,
        )

        if f1 >= best_f1:
            best_epoch = epoch
            best_accuracy = accuracy
            best_f1 = f1

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
            model_path = os.path.join(args.results_path, new_file)
            program.save(model_path)
            logger.info(f"New best model saved to: {model_path}")
            mlflow.log_artifact(model_path)

    logger.info(f"Best epoch {best_epoch}")
    logger.info(f"Best eval Accuracy {best_accuracy * 100}%")
    logger.info(f"Best eval F1 {best_f1 * 100}%")
    mlflow.log_metrics(
        {
            "best_eval_f1": best_f1,
            "best_eval_accuracy": best_accuracy,
            "best_epoch": best_epoch,
        }
    )

    if test_set:
        logger.info("Final evaluation on test set")
        f1, accuracy = eval(program=program, test_set=test_set, cur_device=cur_device, args=args)
        logger.info(f"Final test Accuracy {accuracy * 100}%")
        logger.info(f"Final test F1 {f1 * 100}%")
        mlflow.log_metrics(
            {
                "final_test_f1": f1,
                "final_test_accuracy": accuracy,
            }
        )

    return best_epoch


def main(args: Any) -> None:
    SEED = 382
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    run_name = f"{args.model}_{datetime.now().strftime('%Y-%d-%m_%H:%M:%S')}"
    logger.info(f"Starting run with id {run_name}")
    mlflow.set_experiment("Temporal_FR")
    mlflow.start_run(run_name=run_name)

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
    test_set = TemporalReader.from_file(
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

        eval(program=program, test_set=test_set, cur_device=cur_device, args=args)

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
        train(
            program=program,
            train_set=training_set,
            eval_set=eval_set,
            cur_device=cur_device,
            lr=args.lr,
            program_name=program_name,
            test_set=test_set,
            args=args,
        )
    else:
        train(
            program=program,
            train_set=training_set,
            eval_set=eval_set,
            cur_device=cur_device,
            lr=args.lr,
            program_name=program_name,
            test_set=test_set,
            args=args,
        )
    mlflow.end_run()