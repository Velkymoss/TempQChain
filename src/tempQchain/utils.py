import torch
import tqdm
from domiknows.program.lossprogram import LearningBasedProgram
from domiknows.program.model.base import Mode


def get_avg_loss(
    program: LearningBasedProgram, dataset: list[dict[str, str]], cur_device: str | None, mode: str
) -> float:
    if cur_device is not None:
        program.model.to(cur_device)
    program.model.mode(Mode.TEST)
    program.model.reset()
    train_loss = 0
    total_loss = 0
    with torch.no_grad():
        for data_item in tqdm.tqdm(dataset, f"Calculating {mode} loss" if mode else "Calculating loss"):
            loss, _, *output = program.model(data_item)
            total_loss += 1
            train_loss += loss
    return train_loss / total_loss
