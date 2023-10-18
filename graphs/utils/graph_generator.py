from dataclasses import dataclass, field
import numpy as np

@dataclass(frozen=False, unsafe_hash=True)
class Generation:
    cVarAdj: np.ndarray = field(init=True, default_factory=lambda: np.ndarray, repr=False, compare=False)
    variablesOrdered: np.ndarray = field(init=True, default_factory=lambda: np.ndarray, repr=False, compare=False)

    def __post_init__(self) -> None:
        childs = Generation.softmax(np.clip(self.cVarAdj.sum(1), 0 , 2))
        parents = Generation.softmax(np.clip(self.cVarAdj.sum(0), 0 , 2))
        generalArr = np.arange(len(childs))

        object.__setattr__(self, 'childs', childs)
        object.__setattr__(self, 'parents', parents)
        object.__setattr__(self, 'generalArr', generalArr)

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        return np.exp(logits) / np.exp(logits).sum()

    @staticmethod
    def normalize(scores: np.ndarray) -> np.ndarray:
        return scores / scores.sum()

    def getPreNormedSum(self, parentNotChild: bool, idx: int) -> np.ndarray:
        if parentNotChild:
            return self.cVarAdj[:, idx].sum()
        else:
            return self.cVarAdj[idx, :].sum()

    def getNorm(self, parentNotChild: bool, idx: int) -> np.ndarray:
        if parentNotChild:
            return Generation.normalize(self.cVarAdj[:, idx])
        else:
            return Generation.normalize(self.cVarAdj[idx, :])

    def getNewPick(self, parentNotChild: bool, norm: np.ndarray) -> tuple:
        getPick = np.random.choice(self.generalArr, replace=False, p=norm)
        return getPick, not parentNotChild

    def generator(self, batch_size: int) -> tuple:
        final_path_x = []
        final_path_y = []

        while len(final_path_x) < batch_size:
            path           = []
            path_x         = []
            path_y         = []
            path_x_static  = 0
            break_out      = False
            parentNotChild = False
            normedSum      = 0

            while normedSum == 0:
                if np.random.uniform() > 0.5:
                    picked = np.random.choice(self.generalArr, replace=False, p=self.childs)
                    parentNotChild = False
                else:
                    picked = np.random.choice(self.generalArr, replace=False, p=self.parents)
                    parentNotChild = True

                normedSum = self.getPreNormedSum(parentNotChild, picked)

            norm = self.getNorm(parentNotChild, picked)
            path_x_static = picked
            path.append(picked)

            while not break_out:
                newPick, parentNotChild = self.getNewPick(parentNotChild, norm)

                if newPick in path:
                    break_out = True
                else:
                    path_y.append([newPick])
                    path_x.append(path_x_static)
                    final_path_x.append(path_x_static)
                    final_path_y.append([newPick])
                    path.append(newPick)

                normedSum = self.getPreNormedSum(parentNotChild, picked)
                norm = self.getNorm(parentNotChild, newPick)

        return np.array(final_path_x), np.array(final_path_y)