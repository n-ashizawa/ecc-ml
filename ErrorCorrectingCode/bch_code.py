import bchlib

from utils import *


class BCHCode:
    def __init__(self, args):
        torch_fix_seed(args.seed)
        self.msg_len = args.msg_len
        # ecc_bits = t * m = msg_len(byte)*4 bytes
        #self.t = 16   # when msg_len=32
        #self.m = 8   # when msg_len=32
        #self.t = 8   # when msg_len=8
        #self.m = 4   # when msg_len=8
        self.t = 47   # when msg_len=32, sum_params=5
        self.m = 13   # when msg_len=32, sum_params=5
        self.bch = bchlib.BCH(self.t, m=self.m)


    def encode(self, msg):
        msg_bytes = get_bytes_from_bin(msg)   # 4bytes(32bits)
        ecc = self.bch.encode(msg_bytes)
        encoded_msg = get_bin_from_bytes(msg_bytes+ecc, self.msg_len, reds_len=self.bch.ecc_bytes*8)
        return encoded_msg


    def decode(self, encoded_msg):
        encoded_msg_bytes = get_bytes_from_bin(encoded_msg)   # 4bytes(32bits)
        data = bytearray(encoded_msg_bytes[:-self.bch.ecc_bytes])
        ecc = bytearray(encoded_msg_bytes[-self.bch.ecc_bytes:])

        self.bch.data_len = (self.msg_len+7) // 8   # msg_len=32のとき最大値が15bytes
        nerr = self.bch.decode(data, ecc)
        self.bch.correct(data, ecc)

        decoded_msg_ecc = get_bin_from_bytes(data+ecc, self.msg_len, reds_len=self.bch.ecc_bytes*8)
        decoded_msg = decoded_msg_ecc[:-self.bch.ecc_bytes*8]

        return decoded_msg
