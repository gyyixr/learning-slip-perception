clc;
clear all;
close all;

% Read Files
base_dir = "./models/TCN_20210209-004628/training summary/";

file_val_acc = dir(base_dir + "*accuracy.csv");
file_val_loss = dir(base_dir + "*loss.csv");

% Extract training data
data_val_acc = readtable(base_dir + string(file_val_acc.name), 'HeaderLines', 1); 
data_val_loss = readtable(base_dir + string(file_val_loss.name), 'HeaderLines', 1); 

epochs = 1:300;

data_val_acc = table2array(data_val_acc(epochs, 3));
data_val_loss = table2array(data_val_loss(epochs, 3));

% Plots
figure;
plot(epochs, data_val_acc, 'black','LineWidth',1.5); hold on;

yyaxis left;
title('Validation Metrics');
xlabel('Epochs');
ylabel('Accuracy');
ylim([0.8, 0.95]);

yyaxis right;
plot(epochs, data_val_loss, 'LineWidth', 1.5); hold off;
ylabel('Loss');
ylim([0.15, 0.4]);

