import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Input

# Set random seed for perfect reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==========================================
# 1. SIMULATE TRAFFIC
# ==========================================
T = 1000
time = np.arange(1, T + 1)

# Generate synthetic network slice traffic streams
eMBB  = 60 + 20 * np.random.randn(T) + 10 * np.sin(0.01 * time)
URLLC = 30 + 5 * np.random.randn(T)
mMTC  = 10 + 2 * np.random.randn(T)

# Relu behavior (clip negative traffic values to 0)
eMBB = np.maximum(eMBB, 0)
URLLC = np.maximum(URLLC, 0)
mMTC = np.maximum(mMTC, 0)

# Shape: (3, T) to match MATLAB's row-major orientation
traffic = np.array([eMBB, URLLC, mMTC])

# ==========================================
# 1.1 SIMULATED GRAPH
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(time, traffic[0, :], 'r', label='eMBB', linewidth=1.5)
plt.plot(time, traffic[1, :], 'b', label='URLLC', linewidth=1.5)
plt.plot(time, traffic[2, :], 'g', label='mMTC', linewidth=1.5)
plt.legend()
plt.title('Simulated Traffic Signals')
plt.xlabel('Time')
plt.ylabel('Traffic (Mbps)')
plt.grid(True)

# ==========================================
# 2. NORMALIZE
# ==========================================
minv = np.min(traffic, axis=1, keepdims=True)
maxv = np.max(traffic, axis=1, keepdims=True)
eps = np.finfo(float).eps

traffic_norm = (traffic - minv) / (maxv - minv + eps)

# X handles steps 0 to T-2; Y handles shifted targets 1 to T-1
X = traffic_norm[:, :-1]  
Y = traffic_norm[:, 1:]   

# Keras expectations: [Batch, TimeSteps, Channels] -> (1, 999, 3)
XTrain = np.expand_dims(X.T, axis=0)
YTrain = np.expand_dims(Y.T, axis=0)

# ==========================================
# 3. MODELS (STABILIZED)
# ==========================================

# LSTM (Basic)
model_lstm = Sequential([
    Input(shape=(None, 3)),
    LSTM(64, return_sequences=True),
    Dense(3)
])

# Seq2Seq (More hidden capacity units)
model_seq = Sequential([
    Input(shape=(None, 3)),
    LSTM(128, return_sequences=True),
    Dense(3)
])

# TCN (Optimized with initialization to prevent flat lines)
model_tcn = Sequential([
    Input(shape=(None, 3)),
    Conv1D(128, kernel_size=3, padding='causal', activation='relu', kernel_initializer='he_normal'),
    Conv1D(128, kernel_size=3, padding='causal', activation='relu', kernel_initializer='he_normal'),
    Conv1D(128, kernel_size=3, padding='causal', activation='relu', kernel_initializer='he_normal'),
    Conv1D(128, kernel_size=3, padding='causal', activation='relu', kernel_initializer='he_normal'),
    Dense(3)
])

# Compile LSTM and Seq2Seq with standard learning rate
model_lstm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
model_seq.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')

# Lower learning rate to 0.0005 so deep TCN filters don't collapse into a mean flatline
model_tcn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse')

# ==========================================
# 4. TRAIN
# ==========================================
model_lstm.fit(XTrain, YTrain, epochs=60, batch_size=1, verbose=0)
model_seq.fit(XTrain, YTrain, epochs=60, batch_size=1, verbose=0)
model_tcn.fit(XTrain, YTrain, epochs=60, batch_size=1, verbose=0)

# ==========================================
# 5. PREDICT
# ==========================================
# FIXED: Added .squeeze() to remove the redundant extra axis wrapper dimension
YP_lstm = np.squeeze(model_lstm.predict(XTrain, verbose=0)).T
YP_seq  = np.squeeze(model_seq.predict(XTrain, verbose=0)).T
YP_tcn  = np.squeeze(model_tcn.predict(XTrain, verbose=0)).T

# ==========================================
# 6. DENORMALIZE
# ==========================================
Y_true = traffic[:, 1:]

Y_lstm = YP_lstm * (maxv - minv) + minv
Y_seq  = YP_seq  * (maxv - minv) + minv
Y_tcn  = YP_tcn  * (maxv - minv) + minv

# ==========================================
# 6.1 RMSE CALCULATION
# ==========================================
def calculate_rmse(y_actual, y_predicted):
    return np.sqrt(np.mean((y_actual - y_predicted) ** 2))

rmse_lstm = calculate_rmse(Y_true, Y_lstm)
rmse_seq  = calculate_rmse(Y_true, Y_seq)
rmse_tcn  = calculate_rmse(Y_true, Y_tcn)

print('\nRMSE Comparison:')
print(f'LSTM     : {rmse_lstm:.4f}')
print(f'Seq2Seq  : {rmse_seq:.4f}')
print(f'TCN      : {rmse_tcn:.4f}')

t = np.arange(2, T + 1)

# ==========================================
# 7. MODEL GRAPHS
# ==========================================
def plot_predictions(title_name, Y_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(t, Y_true[0, :], 'r', label='eMBB A')
    plt.plot(t, Y_pred[0, :], 'k--', label='eMBB P')
    plt.plot(t, Y_true[1, :], 'b', label='URLLC A')
    plt.plot(t, Y_pred[1, :], 'm--', label='URLLC P')
    plt.plot(t, Y_true[2, :], 'g', label='mMTC A')
    plt.plot(t, Y_pred[2, :], 'c--', label='mMTC P')
    plt.title(title_name)
    plt.legend()
    plt.grid(True)

plot_predictions('LSTM Traffic Prediction', Y_lstm)
plot_predictions('Seq2Seq Traffic Prediction', Y_seq)
plot_predictions('TCN Traffic Prediction', Y_tcn)

# ==========================================
# 8. THROUGHPUT
# ==========================================
throughput_actual = np.mean(Y_true, axis=1, keepdims=True)
throughput_lstm   = np.mean(Y_lstm, axis=1, keepdims=True)
throughput_seq    = np.mean(Y_seq, axis=1, keepdims=True)
throughput_tcn    = np.mean(Y_tcn, axis=1, keepdims=True)

# ==========================================
# 9. PRB ALLOCATION (PAPER STYLE)
# ==========================================
total_PRB = 1000
priority = np.array([[2], [5], [1]])  # Properly shaped 2D array matrix matching dimensions

PRB_actual = np.round((throughput_actual / np.sum(throughput_actual)) * total_PRB)

PRB_lstm = np.round(((throughput_lstm * priority) / np.sum(throughput_lstm * priority)) * total_PRB)
PRB_seq  = np.round(((throughput_seq  * priority) / np.sum(throughput_seq  * priority)) * total_PRB)
PRB_tcn  = np.round(((throughput_tcn  * priority) / np.sum(throughput_tcn  * priority)) * total_PRB)

# ==========================================
# 10. LATENCY (QoS)
# ==========================================
lat_actual = 1.0 / (throughput_actual + eps)
lat_lstm   = 1.0 / (throughput_lstm + eps)
lat_seq    = 1.0 / (throughput_seq + eps)
lat_tcn    = 1.0 / (throughput_tcn + eps)

# ==========================================
# 11. GRAPHS (BAR CHARTS)
# ==========================================
labels = ['eMBB', 'URLLC', 'mMTC']
x_indexes = np.arange(len(labels))
width = 0.2

def generate_bar_chart(title_str, ylabel_str, act_data, lstm_data, seq_data, tcn_data):
    plt.figure(figsize=(10, 5))
    plt.bar(x_indexes - 1.5*width, act_data.flatten(), width, label='Actual')
    plt.bar(x_indexes - 0.5*width, lstm_data.flatten(), width, label='LSTM')
    plt.bar(x_indexes + 0.5*width, seq_data.flatten(), width, label='Seq2Seq')
    plt.bar(x_indexes + 1.5*width, tcn_data.flatten(), width, label='TCN')
    
    plt.title(title_str)
    plt.xticks(x_indexes, labels)
    plt.ylabel(ylabel_str)
    plt.legend()
    plt.grid(True)

generate_bar_chart('PRB Allocation Comparison', 'PRBs', PRB_actual, PRB_lstm, PRB_seq, PRB_tcn)
generate_bar_chart('Latency (QoS) Comparison', 'Latency', lat_actual, lat_lstm, lat_seq, lat_tcn)
generate_bar_chart('Throughput Comparison', 'Mbps', throughput_actual, throughput_lstm, throughput_seq, throughput_tcn)

plt.show()
