import os
import random

import numpy as np
import torch
import tqdm
from domiknows.program.model.base import Mode
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from tempQchain.logger import get_logger
from tempQchain.programs.Program_tb_dense_YN import program_declaration
from tempQchain.readers.file_loaders import DomiKnowS_reader

logger = get_logger(__name__)


def eval(program, testing_set, cur_device, args):
    from tempQchain.graphs.graph_tb_dense_YN import answer_class

    labels = ["Yes", "No"]
    accuracy_ILP = 0
    accuracy = 0
    count = 0
    count_datanode = 0
    satisfy_constraint_rate = 0
    pred = []
    actual = []
    for datanode in tqdm.tqdm(program.populate(testing_set, device=cur_device), "Manually Testing"):
        count_datanode += 1
        for question in datanode.getChildDataNodes():
            count += 1
            label = labels[int(question.getAttribute(answer_class, "label"))]
            pred_label = int(torch.argmax(question.getAttribute(answer_class, "local/argmax")))
            pred_argmax = labels[pred_label]
            pred.append(pred_label)
            actual.append(int(question.getAttribute(answer_class, "label")))
            accuracy += 1 if pred_argmax == label else 0
        verify_constraints = datanode.verifyResultsLC()
        count_verify = 0
        if verify_constraints:
            for lc in verify_constraints:
                count_verify += verify_constraints[lc]["satisfied"]
        satisfy_constraint_rate += count_verify / len(verify_constraints)
    satisfy_constraint_rate /= count_datanode
    accuracy /= count

    result_file = open("result.txt", "a")
    print(
        "Program:", "Primal Dual" if args.pmd else "Sampling Loss" if args.sampling else "DomiKnowS", file=result_file
    )
    if not args.loaded:
        print("Training info", file=result_file)
        print("Batch Size:", args.batch_size, file=result_file)
        print("Epoch:", args.epoch, file=result_file)
        print("Learning Rate:", args.lr, file=result_file)
        print("Beta:", args.beta, file=result_file)
        print("Sampling Size:", args.sampling_size, file=result_file)
    else:
        print("Loaded Model Name:", args.loaded_file, file=result_file)
    print("Evaluation File:", args.test_file, file=result_file)
    print("Accuracy:", accuracy, file=result_file)
    print("Constraints Satisfied rate:", satisfy_constraint_rate, "%", file=result_file)
    print("Reasoning step:", args.reasoning_steps, file=result_file)
    print("Precious:", precision_score(actual, pred, average=None), file=result_file)
    print("Recall:", recall_score(actual, pred, average=None), file=result_file)
    print("F1:", f1_score(actual, pred, average=None), file=result_file)
    print("F1 Macro:", f1_score(actual, pred, average="macro"), file=result_file)
    print("Confusion Matrix:\n", confusion_matrix(actual, pred), file=result_file)
    result_file.close()

    # df = pd.DataFrame(result_csv)
    # df.to_csv("result.csv")


def train(program, train_set, eval_set, cur_device, limit, lr, program_name="DomiKnow", args=None):
    from tempQchain.graphs.graph_tb_dense_YN import answer_class

    def evaluate():
        labels = ["Yes", "No"]
        count = 0
        actual = []
        pred = []
        for datanode in tqdm.tqdm(program.populate(eval_set, device=cur_device), "Manually Evaluation"):
            for question in datanode.getChildDataNodes():
                count += 1
                actual.append(int(question.getAttribute(answer_class, "label")))
                pred.append(int(torch.argmax(question.getAttribute(answer_class, "local/argmax"))))
        return accuracy_score(actual, pred)

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

    best_loss = float("inf")
    best_acc = 0
    best_epoch = 0
    old_file = None
    training_file = open("training.txt", "a")
    check_epoch = args.check_epoch
    print("-" * 10, file=training_file)
    print("Training by ", program_name, file=training_file)
    print("Learning Rate:", args.lr, file=training_file)
    training_file.close()
    epoch = 0
    for epoch in range(check_epoch, limit, check_epoch):
        training_file = open("training.txt", "a")
        if args.pmd:
            program.train(
                train_set,
                c_warmup_iters=0,
                train_epoch_num=check_epoch,
                Optim=lambda param: torch.optim.Adam(param, lr=lr, amsgrad=True),
                device=cur_device,
            )
        else:
            program.train(
                train_set,
                train_epoch_num=check_epoch,
                Optim=lambda param: torch.optim.Adam(param, lr=lr, amsgrad=True),
                device=cur_device,
            )
        accuracy = evaluate()
        avg_loss = float("inf")
        print("Epoch:", epoch, file=training_file)
        print("Training loss: ", avg_loss, file=training_file)
        print("Dev Accuracy:", accuracy * 100, "%", file=training_file)
        check_condition = avg_loss <= best_loss if args.check_condition == "loss" else accuracy >= best_acc

        if check_condition:
            best_epoch = epoch
            best_acc = accuracy
            best_loss = avg_loss
            # if old_file:
            #     os.remove(old_file)
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
                + "_"
                + str(args.model)
            )
            old_file = new_file
            program.save(os.path.join(args.results_path, new_file))
        training_file.close()

    training_file = open("training.txt", "a")
    if epoch < limit:
        if args.pmd:
            program.train(
                train_set,
                c_warmup_iters=0,
                train_epoch_num=limit - epoch,
                Optim=lambda param: torch.optim.Adam(param, lr=lr, amsgrad=True),
                device=cur_device,
            )
        else:
            program.train(
                train_set,
                train_epoch_num=check_epoch,
                Optim=lambda param: torch.optim.AdamW(param, lr=lr, amsgrad=True),
                device=cur_device,
            )
        accuracy = evaluate()
        avg_loss = float("inf")
        print("Epoch:", limit, file=training_file)
        print("Dev Accuracy:", accuracy * 100, "%", file=training_file)
        check_condition = avg_loss <= best_loss if args.check_condition == "loss" else accuracy >= best_acc

        if check_condition:
            best_epoch = epoch + check_epoch
            best_acc = accuracy
            best_loss = avg_loss
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
                + "_"
                + str(args.model)
            )
            old_file = new_file
            program.save(os.path.join(args.results_path, new_file))
    print("Best epoch ", best_epoch, file=training_file)
    training_file.close()
    return best_epoch


def main(args):
    logger.info("Use chains is %s", args.use_chains)
    SEED = 382
    np.random.seed(SEED)
    random.seed(SEED)
    # pl.seed_everything(SEED)
    torch.manual_seed(SEED)
    cuda_number = args.cuda
    if cuda_number == -1:
        cur_device = "cpu"
    else:
        if torch.cuda.is_available():
            cur_device = "cuda:" + str(cuda_number)
        elif torch.backends.mps.is_available():
            cur_device = "mps"
        else:
            cur_device = "cpu"

    train_file = "tb_dense.json"

    file_path = (
        os.path.join(args.data_path, train_file)
        if isinstance(train_file, str)
        else [os.path.join(args.data_path, file_name) for file_name in train_file]
    )

    training_set = DomiKnowS_reader(
        file_path,
        "YN",
        type_dataset=args.train_file.upper(),
        size=args.train_size,
        upward_level=8,
        augmented=args.use_chains,
        batch_size=args.batch_size,
        rule_text=args.text_rules,
        reasoning_steps=None if args.reasoning_steps == -1 else args.reasoning_steps,
    )

    test_file = "test.json"

    file_path = (
        os.path.join(args.data_path, test_file)
        if isinstance(test_file, str)
        else [os.path.join(args.data_path, file_name) for file_name in test_file]
    )
    testing_set = DomiKnowS_reader(
        file_path,
        "YN",
        type_dataset=args.train_file.upper(),
        size=args.test_size,
        augmented=False,
        batch_size=args.batch_size,
        rule_text=args.text_rules,
        reasoning_steps=None if args.reasoning_steps == -1 else args.reasoning_steps,
    )

    eval_file = "dev.json"

    file_path = (
        os.path.join(args.data_path, eval_file)
        if isinstance(eval_file, str)
        else [os.path.join(args.data_path, file_name) for file_name in eval_file]
    )
    eval_set = DomiKnowS_reader(
        file_path,
        "YN",
        type_dataset=args.train_file.upper(),
        size=args.test_size,
        augmented=False,
        batch_size=args.batch_size,
        rule_text=args.text_rules,
        reasoning_steps=None if args.reasoning_steps == -1 else args.reasoning_steps,
    )

    program_name = "PMD" if args.pmd else "Sampling" if args.sampling else "Base"
    program = program_declaration(
        cur_device,
        pmd=args.pmd,
        beta=args.beta,
        sampling=args.sampling,
        sampleSize=args.sampling_size,
        dropout=args.dropout,
        constraints=args.constraints,
        model=args.model.lower(),
    )

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    if args.loaded:
        logger.info(cur_device)
        program.load(
            os.path.join(args.results_path + args.loaded_file),
            map_location={
                "cuda:0": cur_device,
                "cuda:1": cur_device,
                "cuda:2": cur_device,
                "cuda:3": cur_device,
                "cuda:4": cur_device,
                "cuda:5": cur_device,
                "cuda:6": cur_device,
                "cuda:7": cur_device,
            },
        )
        eval(program, testing_set, cur_device, args)
    elif args.loaded_train:
        program.load(
            os.path.join(args.results_path + args.loaded_file),
            map_location={
                "cuda:0": cur_device,
                "cuda:1": cur_device,
                "cuda:2": cur_device,
                "cuda:3": cur_device,
                "cuda:4": cur_device,
                "cuda:5": cur_device,
                "cuda:6": cur_device,
                "cuda:7": cur_device,
            },
        )
        train(program, training_set, eval_set, cur_device, args.epoch, args.lr, program_name=program_name, args=args)
    else:
        train(program, training_set, eval_set, cur_device, args.epoch, args.lr, program_name=program_name, args=args)
