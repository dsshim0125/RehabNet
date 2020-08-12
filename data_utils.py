import numpy as np
import matplotlib.pyplot as plt
import argparse
import pdb

class data_loader:
    def __init__(self, window_size, index):
        self.data1 = np.loadtxt('data/MLdatasample', delimiter=',')
        self.data2 = np.loadtxt('data/MLdatasample2', delimiter=',')
        self.data3 = np.loadtxt('data/MLdatasample3.txt')
        self.data4 = np.loadtxt('data/190509_data_0.25khz', delimiter=',')
        self.data5 = np.loadtxt('data/190509_data_0.12khz', delimiter=',')
        self.data6 = np.loadtxt('data/190509_data_05khz', delimiter=',')
        self.data = [self.data1, self.data2, self.data3, self.data4, self.data5, self.data6]
        self.window_size = window_size
        self.index = index

    def train_test_split(self, data):
        data_train = data[:int(len(data) * 0.8)]
        data_test = data[int(len(data) * 0.8):]

        return data_train, data_test


    def time_flowmeter_chestbelt(self, data):

        time = data[:, 0]
        flowmeter = data[:, 1]
        chestbelt = data[:, 2]

        return time, flowmeter, chestbelt


    def window_split(self, data, window_size):
        temp = np.zeros((len(data)-(window_size-1),window_size))
        for i in range(len(temp)):
            temp[i] = data[i:i+window_size]
        return temp

    def get_train_data(self, *args, window_size, index):
        count = 0
        for data in args:
            train_data1, _ = self.train_test_split(data)
            time, train_label1, train_data1 = self.time_flowmeter_chestbelt(train_data1)
            if index == count:
                train_time = time
            else:
                pass

            train_temp_data = self.window_split(train_data1, window_size=window_size)
            train_temp_label = train_label1[window_size-1:].reshape(-1,1)

            if count >= 1:

                train_data = np.vstack((train_data, train_temp_data))
                train_label = np.vstack((train_label, train_temp_label))
            else:
                train_data, train_label = train_temp_data, train_temp_label
            count += 1

        return train_time[window_size-1:], train_data, train_label


    def get_test_data(self, *args, window_size, index):
        count = 0
        for data in args:
            _, test_data1 = self.train_test_split(data)
            time, test_label1, test_data1 = self.time_flowmeter_chestbelt(test_data1)

            if index == count:
                test_time = time
            test_temp_data = self.window_split(test_data1, window_size=window_size)
            test_temp_label = test_label1[window_size-1:].reshape(-1,1)

            if count>=1:
                test_data = np.vstack((test_data, test_temp_data))
                test_label = np.vstack((test_label, test_temp_label))
            else:
                test_data, test_label = test_temp_data, test_temp_label
            count += 1

        return test_time[window_size-1:], test_data, test_label

    def train_data_loader(self):
        train_time, train_data, train_label = self.get_train_data(self.data1, self.data2, self.data3, self.data4,
                                                                  self.data5, self.data6,
                                                                  window_size=self.window_size,
                                                                  index=self.index)

        return train_time, train_data, train_label

    def test_data_loader(self):
        test_time, test_data, test_label = self.get_test_data(self.data1, self.data2, self.data3, self.data4,
                                                              self.data5, self.data6,
                                                              window_size=self.window_size,
                                                              index=self.index)

        return test_time, test_data, test_label

    def plotting_train_data(self):
        count = 0
        for i in range(len(self.data)):
            if count == self.index:
                plotting_data = self.data[i]
                plotting_data, _ = self.train_test_split(plotting_data)
                plotting_time, plotting_label, plotting_data = self.time_flowmeter_chestbelt(plotting_data)

                plotting_data = self.window_split(plotting_data, self.window_size)
                plotting_time = plotting_time[self.window_size-1:]
                plotting_label = plotting_label[self.window_size-1:]
                break
            count +=1

        return plotting_time, plotting_label, plotting_data

    def plotting_test_data(self):
        count = 0
        for i in range(len(self.data)):
            if count == self.index:
                plotting_data = self.data[i]
                _, plotting_data = self.train_test_split(plotting_data)
                plotting_time, plotting_label, plotting_data = self.time_flowmeter_chestbelt(plotting_data)

                plotting_data = self.window_split(plotting_data, self.window_size)
                plotting_time = plotting_time[self.window_size - 1:]
                plotting_label = plotting_label[self.window_size - 1:]

                break

            count +=1

        return plotting_time, plotting_label, plotting_data

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', default=100, type=int)
    parser.add_argument('--index', required=True, type=int)

    args = parser.parse_args()
    index = args.index
    window_size = args.window_size

    data_load = data_loader(window_size=window_size, index=index)
    data_plot = data_load.data[index]
    time, data, label = data_load.time_flowmeter_chestbelt(data_plot)

    plt.figure(figsize=(12,6))
    plt.subplot(2,1,1)
    plt.plot(time, data)
    plt.title('Flowmeter')
    plt.subplot(2,1,2)
    plt.plot(time, label)
    plt.title('Chestbelt')
    plt.show()