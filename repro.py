from predict_stage_1 import predict_stage_1
from predict_stage_2 import predict_stage_2


def main():
    '''
    Reproduce all stages from training to final prediction.
    '''
    predict_stage_1()
    predict_stage_2()


if __name__ == '__main__':
    main()