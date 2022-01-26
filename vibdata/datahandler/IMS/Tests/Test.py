# Code made in Pycharm by Igor Varejao
from abc import abstractmethod

class BaseTest():

    # Labels:
    # 0 -> Normal
    # 1 -> Degraded Inner Race
    # 2 -> Inner Race Fault
    # 3 -> Degraded Roller Race
    # 4 -> Roller Race Fault
    # 5 -> Degraded Outer Race
    # 6 -> Outer Race Fault
    LABELS = {
        "Normal": 0,
        "Degraded Inner Race": 1,
        "Inner Race Fault": 2,
        "Degraded Roller Race": 3,
        "Roller Race Fault": 4,
        "Degraded Outer Race": 5,
        "Outer Race Fault": 6,
    }
    INVERT_LABELS = {val: key for key, val in LABELS.items()}

    file_names = NotImplemented
    sample_rate = 20000  # Hz
    num_bearings = NotImplemented
    num_test = NotImplemented

    def __init__(self):
        pass

    @abstractmethod
    def return_label(self, file_name: str, colunm: int) -> int:
        idx = self.file_names.index(file_name)
        pass

    @staticmethod
    def bearing(colunm: int) -> int:
        return (colunm // 2) + 1

    # Method auxiliar to create metadata of the signals
    # file name / sample_rate / bearing / label / test
    def _add(self, meta):
        for file in self.file_names:
            for b in range(self.num_bearings * 2):
                meta.append([file,
                             self.sample_rate,
                             "{}.{}".format(self.bearing(b), 'x' if b % 2 == 0 else 'y'),
                             self.INVERT_LABELS[self.return_label(file, b)],
                             self.num_test])