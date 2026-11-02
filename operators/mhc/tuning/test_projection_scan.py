import unittest
from mhc_projection_scan import split_config


class ProjectionScanContract(unittest.TestCase):
    def test_partition_covers_hidden_with_valid_core_mapping(self):
        base = dict(num_aic_cores=4, num_l0c_stages=16)
        for tokens in (64,128,129,256,512):
            for splits in (1,2,4):
                c = split_config(base,tokens,20480,splits)
                self.assertEqual(c['num_hidden_blocks_per_split']*splits*256,20480)
                self.assertEqual(c['cores_per_split']*splits,c['num_aic_cores'])
                self.assertLessEqual(c['num_aic_cores'],4)
                self.assertEqual(c['num_token_blocks_per_chunk'],c['cores_per_split']*16)
        self.assertEqual(base,dict(num_aic_cores=4,num_l0c_stages=16))

    def test_invalid_partitions_rejected(self):
        for splits in (0,3,8):
            with self.assertRaises(ValueError):
                split_config(dict(num_aic_cores=4,num_l0c_stages=16),128,20480,splits)
