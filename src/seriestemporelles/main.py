import pandas as pd
from darts.datasets import AirPassengersDataset
from darts.models import AutoARIMA, Prophet, ExponentialSmoothing
import sys
sys.path.append("/home/said/Bureau/TimeSeries/series-temporelles/src/")
from seriestemporelles.signal.signal_analysis import SignalAnalysis
from seriestemporelles.signal.visualization import Visualisation

if __name__ == '__main__':
    series = AirPassengersDataset().load()
    dt = pd.DataFrame(series.values())
    dt.rename(columns={0: 'passengers'}, inplace=True)
    dt.index = series.time_index
    Visualisation(dt).plot_signal()
    Visualisation(dt).plot_smoothing()

    signal = SignalAnalysis(dt)
    signal.profiling()
    report = signal.apply_test('stationary', 'kpss')
    
    
    #------ ML -----
    timestamp = '1958'
    signal.split_cv(timestamp=timestamp)
    
    pred_arima = signal.apply_model(AutoARIMA())
    signal.apply_model(Prophet())
    signal.apply_model(ExponentialSmoothing())
    
    
    df_predictions = signal.test_set.copy()
    for model in signal.model_results.keys() : 
        df_predictions.loc[:,model] = signal.model_results[model]['predictions']
        

    # print('model {} obtains MAPE: {:.2f}%'.format(model, mape(data_test, forecast)))




#%% 
import matplotlib.pyplot as plt

plt.plot(signal.train_set)
plt.plot(df_predictions['AutoARIMA()'], c='blue')
plt.plot(df_predictions['Prophet()'], c='green')
plt.plot(df_predictions['ExponentialSmoothing()'], c='red')

cols = df_predictions.columns.drop('passengers')
plt.legend(cols)


