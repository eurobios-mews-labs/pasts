from darts.datasets import AirPassengersDataset
import matplotlib.pyplot as plt

from pasts.operations import Operation

if __name__ == '__main__':
    series = AirPassengersDataset().load()
    dataframe = series.pd_dataframe()
    op = Operation(dataframe)
    fig, ax = plt.subplots()
    det = op.fit_transform(['trend', 'seasonality'])
    det.plot(ax=ax)
    ret = op.transform(det)
    ret.plot(ax=ax)
    plt.legend(['transformed data', 'back-transformed data'])
    plt.show()
