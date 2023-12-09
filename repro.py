from predict import predict
from prepare_data import prepare_data
from train import train


def repro_stage(stage: str) -> None:
    """
    Args:
        stage: one of simple, stage_1, stage_2
    """
    prepare_data(stage)
    train(stage)
    predict(stage)


def repro(stage) -> None:
    if stage == 'stage_2':
        repro_stage('stage_1')
        repro_stage('stage_2')
    elif stage == 'stage_1':
        repro_stage('stage_1')
    elif stage == 'simple':
        repro_stage('simple')


if __name__ == '__main__':
    repro('stage_2')
    
