import commpy.channelcoding.convcode as cc
import commpy.channelcoding.interleavers as ci
import numpy as np
import commpy.utilities as util

from .coding.turbo_code import turbo_encode, turbo_decode, map_decode
from .coding.trellis import generate_trellis

from utils import *


class TurboCode:
    def __init__(self, args):
        torch_fix_seed(args.seed)
        self.msg_len = args.msg_len*args.sum_params
        self.memory = np.array([4])
        self.generator_matrix = np.array([[1, 3, 7]])
        self.code_type = 'rsc'
        self.feedback = np.array((3), ndmin=2)


    def create_double_trellis(self):
        trellis1 = cc.Trellis(memory=self.memory, 
            g_matrix=self.generator_matrix,
            code_type=self.code_type,
            feedback=self.feedback)
        trellis2 = cc.Trellis(memory=self.memory,
            g_matrix=self.generator_matrix,
            code_type=self.code_type,
            feedback=self.feedback)
        return trellis1, trellis2


    def encode(self, msg):
        # インターリーバとトレリスの設定
        L = len(msg)
        interleaver = ci.RandInterlv(length=L, seed=0)
        trellis1, trellis2 = self.create_double_trellis()

        # encode
        [sys_stream, non_sys_stream1, non_sys_stream2] = turbo_encode(msg, trellis1, trellis2, interleaver)
        assert np.all(msg==sys_stream)
        sys_all = np.concatenate([sys_stream, non_sys_stream1, non_sys_stream2])
        return sys_all
            

    def decode(self, encoded_msg):
        trellis_rate3  = generate_trellis("1/3")
        trellis_rate5  = generate_trellis("1/5")

        m = encoded_msg[:self.msg_len]
        e = encoded_msg[self.msg_len:self.msg_len*3]
        e_bar = encoded_msg[self.msg_len*3:]

        interleaver = ci.RandInterlv(length=self.msg_len, seed=0)

        # 誤り訂正される前の m
        m_org = np.array(m)

        e_list = []
        e_bar_list = []

        # RSC code を併用して m を復元する
        start_idx = 0
        for idx in range(2):
            trellis = trellis_rate3 if idx == 0 else trellis_rate5
            e_ = e[start_idx:start_idx+self.msg_len]
            e_list.append(e_)
            # RSC decoding
            # TODO : map_decode でよいのか？
            _, m = map_decode(sys_symbols=m_org,
                non_sys_symbols=np.concatenate(e_list),
                trellis=trellis,
                noise_variance=1.0,
                # L_int が None の時の処理を真似て 0 で初期化する
                L_int=np.zeros(len(m_org)),
                mode='decode')

            # Turbo code を併用して m を復元する
            e_bar_ = e_bar[start_idx:start_idx+self.msg_len]
            e_bar_list.append(e_bar_)
            # Turbo decoding
            m = turbo_decode(sys_symbols=m_org,
                # e を渡す
                non_sys_symbols_1=np.concatenate(e_list),
                # e_bar を渡す
                non_sys_symbols_2=np.concatenate(e_bar_list),
                # 2 つ作っているがどちらも同じなので 1 つだけ渡す
                trellis=trellis,
                noise_variance=1.0,
                number_iterations=10,
                interleaver=interleaver,
                L_int=None)

            start_idx += self.msg_len

        return m
