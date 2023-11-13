from reedsolo import RSCodec, ReedSolomonError

from utils import *


class RSCode:
    def __init__(self, args):
        torch_fix_seed(args.seed)
        self.msg_len = args.msg_len
        # t = ((self.msg_len+7)//8)*4 => 16 bits (msg_len=32)
        self.t = ((self.msg_len+7)//8)*4
        self.rs = RSCodec(self.t)


    def encode(self, msg):
        msg_bytes = get_bytes_from_bin(msg)
        encoded_msg_bytes = self.rs.encode(msg_bytes)
        encoded_msg = get_bin_from_bytes(encoded_msg_bytes, self.msg_len, self.t*8)
        return encoded_msg


    def decode(self, encoded_msg):
        try:
            encoded_msg_bytes = get_bytes_from_bin(encoded_msg)
            decoded_msg_bytes = self.rs.decode(encoded_msg_bytes)
            decoded_msg_all = get_bin_from_bytes(decoded_msg_bytes[1], self.msg_len, reds_len=self.t*8)
            decoded_msg = decoded_msg_all[:-self.t*8]
            return decoded_msg
        except ReedSolomonError:
            return encoded_msg[:-self.t*8]
