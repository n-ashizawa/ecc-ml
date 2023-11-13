import commpy.channelcoding.convcode as cc
import numpy as np


TURBO_CODE_RATES = ['1/3', '1/5']


def generate_trellis(turbo_code_rate):
    """
    Parameters
    ----------
s
    Returns
    -------
    """
    # 符号化率により generator_matrix のサイズが変わる
    # 行数k、列数nとする時、rate=k/nである
    if turbo_code_rate == '1/5':
        rsc_code_rate = '1/3'
        # MEMO: 
        # ターボ符号が考案された論文 http://coding.yonsei.ac.kr/firstpaperonturbocode.pdf
        # で、genratorが(37, 21)と記載があったので真似した
        # D^5まで必要となるため、memoryを5に変更した
        memory = np.array([5])
        # 生成多項式 1, 1+D^2+D^5, 1+D^2+D^4
        generator_matrix = np.array([[1, 37, 21]])
        feedback = np.array((63), ndmin=2)
    elif turbo_code_rate == '1/3':
        rsc_code_rate = '1/2'
        memory = np.array([6])
        # # 生成多項式 1, 1+D^2+D^5
        # generator_matrix = np.array([[1, 37]])
        # 生成多項式 1, 1+D^2+D^4+D^6
        generator_matrix = np.array([[1, 1+4+16+64]])
        feedback = np.array((1+2+4+8+16+32+64), ndmin=2)
    else:
        raise ValueError(f'{TURBO_CODE_RATES}から選択してください。')
    code_type = 'rsc'

    trellis = cc.Trellis(memory=memory,
                         g_matrix=generator_matrix,
                         code_type=code_type,
                         feedback=feedback)

    #print(f'memory          : {memory}')
    #print(f'turbo code rate : {turbo_code_rate}')
    #print(f'RSC   code rate : {rsc_code_rate}')
    #print(f'generator matrix: {generator_matrix}')
    #print(f'code_type       : {code_type}')
    #print(f'feedback        : {feedback}')

    return trellis