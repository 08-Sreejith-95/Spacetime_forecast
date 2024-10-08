This directory contains scripts to read the raw data from the directory and its processing to sequential Datasets including processing into pytorch Dataset
These are custom made for etth datasets. Use this as a reference

1)sequence.py :- The data pipeline starts here. This module contains reading raw data from directory.(Look at this script for the code for retrieving data from the path)
So all the new data should be stored in the path dataloaders/data/informer. Comments are added regularly through the code by me for proper understanding. >Check those if doubts pop up

2)informer.py :- this code is adapted from the informer model. They used ETTH, ETTM datasets.
     1. here the raw data is processed. The sequence length of raw data is 17420 for each variables
     2. it is split into sequences of dimensions [seq_len, label_len, pred_len]. by default [384, 96, 96]. ie., sequence length of 16 days out of this the labels are for 4 days and prediction horizon is also of 4 days. and the target class by default is 'OT' for forecasting task
        so [24*4*4, 24*4, 24*4]. seq_len is lag_len, with this we predict sequence length of next 24*4 values and we have corresponding labels for the same length.
        Note:- we are feeding subsequences as the input data for the whole sequence length. so a single sub sequence is of length 384 and we predict next 96 from it and the labels are the actual values of this
                   
          The date column in pd data frame are the frequencies, h. This can be minute, seconds, days, hrs and so on. The features are the values of this frequencies for a particular variable.
              The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
                   
                   
                   > `time_features` takes in a `dates` dataframe with a 'dates' column and extracts the date down to `freq` where freq can be any of the following if `timeenc` is 0:
    > * m - [month]
    > * w - [month]
    > * d - [month, day, weekday]
    > * b - [month, day, weekday]
    > * h - [month, day, weekday, hour]
    > * t - [month, day, weekday, hour, *minute]
    >
    > If `timeenc` is 1, a similar, but different list of `freq` values are supported (all encoded between [-0.5 and 0.5]):
    > * Q - [month]
    > * M - [month]
    > * W - [Day of month, week of year]
    > * D - [Day of week, day of month, day of year]
    > * B - [Day of week, day of month, day of year]
    > * H - [Hour of day, day of week, day of month, day of year]
    > * T - [Minute of hour*, hour of day, day of week, day of month, day of year]
    > * S - [Second of minute, minute of hour, hour of day, day of week, day of month, day of year]
    *minute returns a number from 0-3 corresponding to the 15 minute period it falls into.
    """
    doubt:- the values are monitored everyday. so check whether each training data is from one day. and the prediction length is in seconds, minutes or hours timestamps for the same day
    in the code a sample contains x_sequence(lag sequence), y_label(prediction_sequence), time_mask(probably in seconds, minutes, hours, or days), mark and mask(What these represents?)
    1)This doubt is in informer.py, informerdataset class, __get_item__

    doubts:- 1) the difference between informerdataset and informerSequenceDataset classes. bcz all the custom datasets are children of this parent class
             2) If we use audio data; is the above dataset compatible. Note: the spacetime model is good for long term forecasting and classification tasks only