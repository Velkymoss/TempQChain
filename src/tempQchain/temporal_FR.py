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
from tempQchain.programs.program import (
    program_declaration_tb_dense_fr,
)
from tempQchain.readers.temporal_reader import TemporalReader
from tempQchain.utils import get_avg_loss

warnings.filterwarnings("ignore")

logger = get_logger(__name__)

LABEL_STRINGS = ["after", "before", "includes", "is_included", "simultaneous", "vague"]


def eval(
    program: LearningBasedProgram,
    test_set: list[dict[str, str]],
    cur_device: str,
    dec: int = 2,
    args: Any = None,
) -> tuple[float, float, float | None, list[float]]:
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

    if args.constraints:
        total_constraint_sat = 0
        num_constraints = 0

    for datanode in tqdm.tqdm(program.populate(test_set, device=cur_device), "Checking f1/accuracy..."):
        for question in datanode.getChildDataNodes():
            pred_set.clear()
            pred_list.clear()

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
        if args.constraints:
            verify_constraints = datanode.verifyResultsLC()

            for lc in verify_constraints:
                satisfied_val = verify_constraints[lc].get("satisfied")
                if_satisfied_val = verify_constraints[lc].get("ifSatisfied")
                if (
                    isinstance(satisfied_val, float)
                    and isinstance(if_satisfied_val, float)
                    and not np.isnan(satisfied_val)
                    and not np.isnan(if_satisfied_val)
                ):
                    total_constraint_sat += satisfied_val + if_satisfied_val
                    num_constraints += 2

    accuracy = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average="macro")
    if args.constraints:
        overall_constraint_rate = total_constraint_sat / num_constraints if num_constraints else 100
    f1_per_class = f1_score(all_true, all_pred, average=None)
    return (
        round(f1 * 100, dec),
        round(accuracy * 100, dec),
        round(overall_constraint_rate, dec) if args.constraints else None,
        [round(score * 100, dec) for score in f1_per_class],
    )


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
    best_epoch = 0

    logger.info("Starting FR training...")
    logger.info("Model: Modern Bert Base")
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

    if args.use_mlflow:
        mlflow.log_params(
            {
                "model": "Modern Bert Base",
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
        f1, accuracy, constraint_rate, _ = eval(program=program, test_set=eval_set, cur_device=cur_device, args=args)

        logger.info(f"Epoch: {epoch}")
        logger.info(f"Train Loss: {train_loss}")
        logger.info(f"Eval Loss: {eval_loss}")
        logger.info(f"Dev Accuracy: {accuracy}%")
        logger.info(f"Dev F1: {f1}%")
        if args.constraints:
            logger.info(f"Dev Constraint Rate: {constraint_rate}")
        if args.use_mlflow:
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    "eval_f1": f1,
                    "eval_accuracy": accuracy,
                },
                step=epoch,
            )
            if args.constraints:
                mlflow.log_metric("eval_cosntraint_rate", constraint_rate)

        if f1 >= best_f1:
            best_epoch = epoch
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
                + "modernbert"
            )
            model_path = os.path.join(args.results_path, new_file)
            program.save(model_path)
            logger.info(f"New best model saved to: {model_path}")
            if args.use_mlflow:
                mlflow.log_artifact(model_path)

    logger.info(f"Best epoch {best_epoch}")
    logger.info(f"Best eval F1 {best_f1}%")
    if args.use_mlflow:
        mlflow.log_metrics(
            {
                "best_eval_f1": best_f1,
                "best_epoch": best_epoch,
            }
        )

    if test_set:
        logger.info("Final evaluation on test set")
        f1, accuracy, constraint_rate, f1_per_class = eval(
            program=program, test_set=test_set, cur_device=cur_device, args=args
        )
        for label, score in zip(LABEL_STRINGS, f1_per_class):
            if args.use_mlflow:
                mlflow.log_metric(f"test_f1_class_{label}", score)
            logger.info(f"F1 {label}: {score}%")
        logger.info(f"Test Accuracy {accuracy}%")
        logger.info(f"Test F1 {f1}%")
        logger.info(f"Test Constraint Rate {constraint_rate}%")
        if args.use_mlflow:
            mlflow.log_metrics(
                {
                    "test_f1": f1,
                    "test_accuracy": accuracy,
                }
            )
            if args.constraints:
                mlflow.log_metric("test_cosntraint_rate", constraint_rate)
    return best_epoch


def main(args: Any) -> None:
    SEED = 382
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    if args.use_mlflow:
        run_name = f"modernbert_{datetime.now().strftime('%Y-%d-%m_%H:%M:%S')}"
        logger.info(f"Starting run with id {run_name}")
        mlflow.set_experiment("Temporal_FR")
        mlflow.start_run(run_name=run_name)

    cuda_number = args.cuda
    if cuda_number == -1:
        cur_device = "cpu"
    else:
        cur_device = "cuda:" + str(cuda_number) if torch.cuda.is_available() else "cpu"

    program = program_declaration_tb_dense_fr(
        cur_device,
        pmd=args.pmd,
        beta=args.beta,
        sampling=args.sampling,
        sampleSize=args.sampling_size,
        dropout=args.dropout,
        constraints=args.constraints,
    )

    train_file = "tb_dense_train.json"
    training_set = TemporalReader.from_file(
        file_path=os.path.join(args.data_path, train_file), question_type="FR", batch_size=args.batch_size
    )[:2]

    test_file = "tb_dense_test.json"
    test_set = TemporalReader.from_file(
        file_path=os.path.join(args.data_path, test_file), question_type="FR", batch_size=args.batch_size
    )[:2]

    eval_file = "tb_dense_dev.json"
    eval_set = TemporalReader.from_file(
        file_path=os.path.join(args.data_path, eval_file), question_type="FR", batch_size=args.batch_size
    )[:2]

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
    if args.use_mlflow:
        mlflow.end_run()
