import pandas as pd

class PatternIdentifier:

    def double_top_bottom(self, data, window=30, trend_window=50, threshold=0.015):
        """
        Identify double top and double bottom patterns in stock data, considering the trend direction.

        Parameters:
        - data: DataFrame with columns ['High', 'Low']

        Returns:
        - double_tops: List of tuples containing indices of double top patterns
        - support_resistance_tops: List of dicts with resistance, support, and trough index for double tops
        - double_bottoms: List of tuples containing indices of double bottom patterns
        - support_resistance_bottoms: List of dicts with support, resistance, and peak index for double bottoms
        """
        # Calculate rolling max/min to identify peaks/troughs
        data['rolling_max'] = data['High'].rolling(window=window, center=True).max()
        data['rolling_min'] = data['Low'].rolling(window=window, center=True).min()

        # Determine trend direction
        data['trend'] = data['Close'].rolling(window=trend_window).apply(lambda x: x[-1] - x[0], raw=True)

        # Bullish or bearish trend based on trend_window
        is_bullish = data['trend'].iloc[-1] > 0
        is_bearish = data['trend'].iloc[-1] < 0

        double_tops = []
        support_resistance_tops = []
        double_bottoms = []
        support_resistance_bottoms = []

        if is_bullish:
            # Find local peaks for bullish trend (double top pattern)
            peaks = (data['High'] == data['rolling_max']) & (data['High'].shift(-1) < data['High']) & (
                        data['High'].shift(1) < data['High'])
            peak_indices = data[peaks].index

            for i in range(len(peak_indices) - 1):
                first_peak_idx = peak_indices[i]
                second_peak_idx = peak_indices[i + 1]

                # Check if the second peak is within the threshold of the first peak
                if abs(data.loc[first_peak_idx, 'High'] - data.loc[second_peak_idx, 'High']) / data.loc[
                    first_peak_idx, 'High'] <= threshold:
                    # Find the lowest point between the two peaks
                    trough_idx = data.loc[first_peak_idx:second_peak_idx, 'Low'].idxmin()
                    trough_value = data.loc[trough_idx, 'Low']

                    # Ensure the trough is lower than the peaks
                    if trough_value < data.loc[first_peak_idx, 'High']:
                        double_tops.append((first_peak_idx, second_peak_idx))
                        support_resistance_tops.append({
                            'resistance': data.loc[first_peak_idx, 'High'],
                            'support': trough_value,
                            'trough_idx': trough_idx
                        })

        if is_bearish:
            # Find local troughs for bearish trend (double bottom pattern)
            troughs = (data['Low'] == data['rolling_min']) & (data['Low'].shift(-1) > data['Low']) & (
                        data['Low'].shift(1) > data['Low'])
            trough_indices = data[troughs].index

            for i in range(len(trough_indices) - 1):
                first_trough_idx = trough_indices[i]
                second_trough_idx = trough_indices[i + 1]

                # Check if the second trough is within the threshold of the first trough
                if abs(data.loc[first_trough_idx, 'Low'] - data.loc[second_trough_idx, 'Low']) / data.loc[
                    first_trough_idx, 'Low'] <= threshold:
                    # Find the highest point between the two troughs
                    peak_idx = data.loc[first_trough_idx:second_trough_idx, 'High'].idxmax()
                    peak_value = data.loc[peak_idx, 'High']

                    # Ensure the peak is higher than the troughs
                    if peak_value > data.loc[first_trough_idx, 'Low']:
                        double_bottoms.append((first_trough_idx, second_trough_idx))
                        support_resistance_bottoms.append({
                            'support': data.loc[first_trough_idx, 'Low'],
                            'resistance': peak_value,
                            'peak_idx': peak_idx
                        })

        return double_tops, support_resistance_tops, double_bottoms, support_resistance_bottoms

    def double_top_bottom_orders(self, data, double_tops, support_resistance_tops, double_bottoms, support_resistance_bottoms):
        """
        Place buy/sell orders based on the identified double top and double bottom patterns.
        """
        orders = []

        # Process double top patterns
        for dt, sr in zip(double_tops, support_resistance_tops):
            start_idx, end_idx = dt
            support = sr['support']
            trough_idx = sr['trough_idx']

            # Identify double top pattern completion and bearish trend
            for i in range(end_idx, len(data)):
                if data['Low'].iloc[i] < support:
                    orders.append({'type': 'sell', 'index': i, 'price': data['Low'].iloc[i]})
                    break

        # Process double bottom patterns
        for db, sr in zip(double_bottoms, support_resistance_bottoms):
            start_idx, end_idx = db
            resistance = sr['resistance']
            peak_idx = sr['peak_idx']

            # Identify double bottom pattern completion and bullish trend
            for i in range(end_idx, len(data)):
                if data['High'].iloc[i] > resistance:
                    orders.append({'type': 'buy', 'index': i, 'price': data['High'].iloc[i]})
                    break

        return orders