clc;
clear all;
close all;

% Read Files
base_dir = "../logs/models/TCN_20210210-011650/";
data = load(base_dir + "data_time_series.mat");

inputs = data.inputs(:, end, :);
time = (1:length(inputs)) / 100.0;
slip = data.labels(:,2);
prediction_TCN = data.predictions(:,2);

base_dir = "../logs/models/FREQ_20210212-011047/";
data = load(base_dir + "data_time_series.mat");
prediction_freq = data.predictions(:,2);

% Plot
figure;
plot(time, slip, 'Color', [0.3 0.3 0.3],'LineWidth',2.0); hold on;
plot(time, prediction_TCN, 'red','LineWidth',1.0);
plot(time, prediction_freq, 'blue','LineWidth',1.0);
plot(time([1,end]), 0.5*ones(2), 'black')
ylabel('Probability of Slip');
ylim([-0.1, 1.1]);
xlim([0.5, 3.85])
legend("Label", "TCN Prediction", "Freq CNN Prediction");

xlabel("Time [s]");

pred = prediction_freq > 0.5;
accuracy = slip == pred;
sum(accuracy)/length(accuracy)


pred = prediction_TCN > 0.5;
accuracy = slip == pred;
sum(accuracy)/length(accuracy)