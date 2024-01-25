import os
from typing import Dict, Union

import numpy as np
import pandas as pd
from scipy.io import loadmat

from vibdata.raw.base import DownloadableDataset, RawVibrationDataset
from vibdata.raw.utils import _get_package_resource_dataframe
from vibdata.definitions import LABELS_PATH


class CWRU_raw(RawVibrationDataset, DownloadableDataset):
    DATAFILE_NAMES = [
        # normal
        ["97.mat"],
        ["98.mat"],
        ["99.mat"],
        ["100.mat"],
        # For 12k Drive End Bearing Fault Data
        [
            "105.mat",
            "118.mat",
            "130.mat",
            "169.mat",
            "185.mat",
            "197.mat",
            "209.mat",
            "222.mat",
            "234.mat",
        ],  # 1797rpm
        [
            "106.mat",
            "119.mat",
            "131.mat",
            "170.mat",
            "186.mat",
            "198.mat",
            "210.mat",
            "223.mat",
            "235.mat",
        ],  # 1772rpm
        [
            "107.mat",
            "120.mat",
            "132.mat",
            "171.mat",
            "187.mat",
            "199.mat",
            "211.mat",
            "224.mat",
            "236.mat",
        ],  # 1750rpm
        [
            "108.mat",
            "121.mat",
            "133.mat",
            "172.mat",
            "188.mat",
            "200.mat",
            "212.mat",
            "225.mat",
            "237.mat",
        ],  # 1730rpm
        # For 12k Fan End Bearing Fault Data
        [
            "278.mat",
            "282.mat",
            "294.mat",
            "274.mat",
            "286.mat",
            "310.mat",
            "270.mat",
            "290.mat",
            "315.mat",
        ],  # 1797rpm
        [
            "279.mat",
            "283.mat",
            "295.mat",
            "275.mat",
            "287.mat",
            "309.mat",
            "271.mat",
            "291.mat",
            "316.mat",
        ],  # 1772rpm
        [
            "280.mat",
            "284.mat",
            "296.mat",
            "276.mat",
            "288.mat",
            "311.mat",
            "272.mat",
            "292.mat",
            "317.mat",
        ],  # 1750rpm
        [
            "281.mat",
            "285.mat",
            "297.mat",
            "277.mat",
            "289.mat",
            "312.mat",
            "273.mat",
            "293.mat",
            "318.mat",
        ],  # 1730rpm
        # For 48k Drive End Bearing Fault Data
        [
            "109.mat",
            "122.mat",
            "135.mat",
            "174.mat",
            "189.mat",
            "201.mat",
            "213.mat",
            "250.mat",
            "262.mat",
        ],  # 1797rpm
        [
            "110.mat",
            "123.mat",
            "136.mat",
            "175.mat",
            "190.mat",
            "202.mat",
            "214.mat",
            "251.mat",
            "263.mat",
        ],  # 1772rpm
        [
            "111.mat",
            "124.mat",
            "137.mat",
            "176.mat",
            "191.mat",
            "203.mat",
            "215.mat",
            "252.mat",
            "264.mat",
        ],  # 1750rpm
        [
            "112.mat",
            "125.mat",
            "138.mat",
            "177.mat",
            "192.mat",
            "204.mat",
            "217.mat",
            "253.mat",
            "265.mat",
        ],  # 1730rpm
    ]
    ALLNAMES = [item for sublist in DATAFILE_NAMES for item in sublist]

    MD5SUMS = [
        "ee410e7243aefcd8b7120876556464e7",
        "3983dcddead0d4b910ee1e8d3da164b9",
        "edc080a0b4fbc0c2ec8b7f1403372b73",
        "cc57a511b95785c805055d99281ea2bb",
        "f14821b40412f33018279198da7f9cdc",
        "ff0f20588edc64140ae33888fcf1114a",
        "e45f73f65cc3de4482d28d758503f2c1",
        "09e41b26825d87e03908add7f01d21aa",
        "e9121cec5988ffd8b33b855371e03ba5",
        "9109c26832e82a904fd8545eebe4c05c",
        "a6c8deded8d592ded19818d001687abc",
        "9e572db7bcdd0690fed770fcd29d07ca",
        "e95df8c8a6fc455d9cea5c2ca36fdc38",
        "4abe03d02739813eebc879b20b591968",
        "ba784233ebbe424e31bb35c5523413cd",
        "8053f13c15bb8891dd7586466e72fb34",
        "3b6eb7ba1276995006d2754a821cbe54",
        "29d343340eba96cc3ce4777a3952a30b",
        "376e134d452b25fcd95c2cf4f5c5b93d",
        "fd00ce7a23af6324b1f8aa7784eda37a",
        "df865596a366f545a4b6f5869bdd1531",
        "6d19ac8a0a37e74a19ed6c6dbd150f79",
        "2bf7a32b6b59aa379d5a4a3c345418f5",
        "89fbf0d771512d848972fd574d3923e8",
        "cf1a7ab3e14564be490c7c27bfafed78",
        "22a48a1c32a58fe320cef89119b8b0d9",
        "4f290b5d8f16d466c96f665ccdb08b08",
        "bfe060099052eeb4f3081c80194cc40e",
        "49e63eb2b41faf93ff8452d4fffe8773",
        "b710aabecd66dc21b4f28db4ac968a0a",
        "92947302ba4198102894daf7613e5fff",
        "ccec995fde6f8865632972804dadfc55",
        "7715250229a6d47910d12be7e1970e70",
        "796a8db507869acf5e0b880d0925f693",
        "5acfb2d067d71a084272b832246c3ae9",
        "04728be1ecfd77e106e0371dedc460bb",
        "e57c5e90335671a2302ebb8f3c626509",
        "bf7c1d5702c18f85f54934bbcbb468c0",
        "523ac0bfb09515c73001d3fbb97a4ffd",
        "cafb20bc696fc025ef42cf12eaf986e3",
        "65bb377a01906b61be2f9d82027abffb",
        "b3815a01a53b7f98bd0162726409c593",
        "bd3b8c05a01b46b5114c582023771f0a",
        "8fec873f01e7645f8e47112f101c0b3d",
        "8b41b5e5278b263b7206b8ecd44cb764",
        "8c1c4582dc4c98cc9f41b7aa6479cde6",
        "e7f5a5fdee6e2899628b6c0c7088c832",
        "5200dc9a0353fcf76d58d2ccee3bccfb",
        "309e812a53453f79a90b4a24b6f2f2bd",
        "3f4a16cd10c7522c0c5b9b51addf4a67",
        "12347b9168e4eda4bc133d42e3b7edae",
        "fa66c5abf44505a65b574f0e623f547a",
        "9eb259384b0b08b034e1de5c751aab6e",
        "9fb29f6ecb2608ffa81cf8acbef983a8",
        "06c623a1dfffaf611c463ba1e7d2afd0",
        "3595226d250799a4ff43ce14d895cd70",
        "cf66bae88f59bc35dde92c79f8f42d1f",
        "e3d5f99d7b9af9aa69f7ff6e6aca9d13",
        "95353b9ec4866802f2d64fbbbeeff34f",
        "f78b1e4a33e3c9e9ee28c2a352009a44",
        "2118a5b58c4307b48dc272ea40056db7",
        "76b0097ed4c68b3fb3109eed4510803a",
        "1f6c80e4cf1ef217a0a079b8ed8af50b",
        "314ca0b85eb36b7d94506a4e9a0a3a74",
        "f95a1937ee1bc820524c7b96cee8331a",
        "a9c2778b4c48d7203afb28365ecbe7bd",
        "7a85bf46f4f83079cee966a2ed127208",
        "f1f971229e9825c34a1958221276f584",
        "604d2e29cf8d1c0ef6897d0d251a76de",
        "3c84c249a6ea33b86da4d406b3952715",
        "ea62aa12752e4e57b956264f80c661d2",
        "02efbf0bb53a747ac72803ac6e332f67",
        "aea99437e60f8d90af3c72469c8dcf0c",
        "9a8f61648d7322904d960442fcea0375",
        "f25eea5929eec6cfbed80dedd1402890",
        "38939f3cc8ff11147436a894729aeeb3",
        "21f95b20e7c98609fc66c8652a86a6b8",
        "e16dd62a706188f6a839bc085d7de929",
        "7cf96872e6156106809e8d454a58ff1d",
        "e61c1d83e8242bc8c1bfa8778f383cdc",
        "b2c173a0879e88bb279e3e2d2b7f8344",
        "d18d2247d53d4a6c281c4a9417004baf",
        "b8c574146addd5538d925e190f43d649",
        "ac3e2c039f761fb5a87a342d0c0bcb5b",
        "b9f45a4bc2ca6b7845b48217abc18b61",
        "a5234140e59c48d01ba9d513e5ddc59e",
        "91ab77778e80438ac943ecff08cb8fac",
        "4f0f21eadd53dc1a95cab990c1e63b6b",
        "b230bae4eaaddebc1d90dfaca334b71c",
        "eb0175694bef59f9576e5f9e928cf4bf",
        "3e97426841e285f44105f818f70761b9",
        "6c062833c33031566433a90027266215",
        "a9d0b7fbb1bfe438d276983a45510afe",
        "ebe4ec3c1e634e338aae955684984f8e",
        "56e41a611051b3e68d21edf68b957a5f",
        "602f3dbdc77fdd01f25e67807e1b6bc3",
        "5e11e867a1c3d16b6f95e96725ee9147",
        "0770082acc38b496faf09760fe3e32e9",
        "4c17fb20756b6f4c876502d1e8593f18",
        "60fc7c36c91e6e00a422fa0a89b6fe4e",
        "26889a42df8f4c0da5aec81a23407449",
        "b89d82da441a47ea2cf272c93afe321f",
        "092a7f792fc4682650cd41f06c33ba13",
        "57fbd6cc885600cfb3c16ff24ec77be4",
        "2ce3d980588e453e30d1cb24dce48973",
        "a00915a4b25bd93c1430f9506c1fab42",
        "a73513cd1c97a39c16d7ef16e5acc193",
        "e282d5813c40d2cc4ecb5dd718e07bee",
        "120fe20ba5923de1863f3a6a520eabed",
        "5ff8eff57cf75b7245e010012d66d3b9",
        "134de2b4de2a562ed8d788b10b67039b",
        "d05f3df2271c3e091b306a78bd5d071b",
    ]

    # mirrors = ["https://engineering.case.edu/sites/default/files"]
    # resources = [(name, md5_value) for name, md5_value in zip(ALLNAMES, MD5SUMS)]
    # https://drive.google.com/file/d/1G2vfms1QDlkdzqL_LAQdMIQAoludxBNj/view?usp=sharing
    mirrors = ["1G2vfms1QDlkdzqL_LAQdMIQAoludxBNj"]
    resources = [("CWRU.zip", "d7d3042161080fc82e99d78464fa2914")]

    # resources = [('97.mat', 'ee410e7243aefcd8b7120876556464e7')]

    def __init__(self, root_dir: str, download=False):
        if download:
            # print(f"Downloading {self.name()}")
            super().__init__(
                root_dir=root_dir,
                download_resources=CWRU_raw.resources,
                download_urls=CWRU_raw.mirrors,
                extract_files=True,
            )
        else:
            super().__init__(root_dir=root_dir, download_resources=CWRU_raw.resources)

    def __getitem__(self, i) -> dict:
        if not hasattr(i, "__len__") and not isinstance(i, slice):
            ret = self.__getitem__([i])
            return ret
        df = self.getMetaInfo()
        if isinstance(i, slice):
            rows = df.iloc[i.start : i.stop : i.step]
        else:
            rows = df.iloc[i]
        file_name = rows["file_name"]
        var_name = rows["variable_name"]
        signal_datas = np.empty(len(var_name), dtype=object)
        for i, (f, v) in enumerate(zip(file_name, var_name)):
            data = loadmat(
                os.path.join(self.raw_folder, f),
                simplify_cells=True,
                variable_names=[v],
            )
            signal_datas[i] = data[v]
        signal_datas = signal_datas
        return {"signal": signal_datas, "metainfo": rows}

    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)

    def getMetaInfo(self, labels_as_str=False) -> pd.DataFrame:
        df = _get_package_resource_dataframe(__package__, "CWRU.csv")
        if labels_as_str:
            # Create a dict with the relation between the centralized label with the actually label name
            all_labels = pd.read_csv(LABELS_PATH)
            dataset_labels: pd.DataFrame = all_labels.loc[all_labels["dataset"] == self.name()]
            dict_labels = {id_label: labels_name for id_label, labels_name, _ in dataset_labels.itertuples(index=False)}
            df["label"] = df["label"].apply(lambda id_label: dict_labels[id_label])
        return df

    def name(self):
        return "CWRU"
