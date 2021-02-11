clc;
clear all;
close all;

% Read Files
base_dir = "../logs/models/TCN_20210210-011650/";
data = load(base_dir + "data_time_series.mat");

inputs = data.inputs(:, end, :);
time = (1:length(inputs)) / 100.0;
slip = data.labels(:,2);
prediction = data.predictions(:,2);

figure;
plot(time, slip,'LineWidth',1.5); hold on;
plot(time, prediction,'LineWidth',1.5);
plot(time([1,end]), 0.5*ones(2), 'black')
ylabel('Probability of Slip');
ylim([-0.1, 1.1]);
legend("Label", "Prediction")

xlabel("Time [s]");

