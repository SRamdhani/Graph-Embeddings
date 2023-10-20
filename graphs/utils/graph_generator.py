from dataclasses import dataclass, field
import numpy as np

""" A very basic generator just for testing this code template."""
@dataclass(frozen=False, unsafe_hash=True)
class Generation:

    def generator(self, i):
        temp_1_x = []
        temp_1_y = []
        if i == (self.total_batches-1):
            temp_1_x = self.full_data_x[(len(self.full_data_x)-self.batch_size):len(self.full_data_x)]
            temp_1_y = self.full_data_y[(len(self.full_data_y)-self.batch_size):len(self.full_data_y)]
        else:
            temp_1_x = self.full_data_x[(i*self.batch_size):(i*self.batch_size+self.batch_size)]
            temp_1_y = self.full_data_y[(i*self.batch_size):(i*self.batch_size+self.batch_size)]
        return np.array(temp_1_x), np.array(temp_1_y)

    def shuffle(self):
        shuffleIdx = np.random.choice(a=np.arange(len(self.full_data_x)),
                                size=len(self.full_data_x), replace=False)

        self.full_data_x = self.full_data_x[shuffleIdx]
        self.full_data_y = self.full_data_y[shuffleIdx]

