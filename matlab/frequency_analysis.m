clear all;
close all;
clc;

%% MAT
data_mat = load('data_mat.mat');

SPEED_THRESH = 0.003; % 1 cm/s

pressure_mat = data_mat.pressure(:,end,:);

time_mat = data_mat.time - data_mat.time(1);

slip_mat = data_mat.slip;

% figure;
% a1 = subplot(2,1,1);
% hold on;
% % plot(time_mat, pressure_mat(:,:,1));
% plot(time_mat, pressure_mat(:,:,2));
% plot(time_mat, pressure_mat(:,:,3));
% plot(time_mat, pressure_mat(:,:,4));
% % plot(time_mat, pressure_mat(:,:,5));
% % plot(time_mat, pressure_mat(:,:,6));
% hold off;
% 
% speed = [];
% for i = 1:data_mat.num
%     speed = [speed; sqrt(slip_mat(i,1)^2 + slip_mat(i,2)^2)];
% end
% 
% 
% a2 = subplot(2,1,2);
% plot(time_mat, speed);
% % ylim([-0.1 1.1])
% 
% linkaxes([a1 a2],'x')

%% Felt
data_felt = load('data_felt.mat');

NUM = data_felt.num;
LEN = 20;%length(data_felt.pressure(1,:,1));
FREQ = 100;

pressure_felt = data_felt.pressure(1:NUM, :, :);

time_felt = data_felt.time(1:NUM) - data_felt.time(1);

slip_felt = data_felt.slip(1:NUM, :);

speed = [];
slip = [];
dim = 4;
fft_felt = [];
fft_felt_slip = [];
fft_felt_static = [];
slip_counter = 0;
for i = 1:NUM
    speed = [speed; sqrt(slip_felt(i,1)^2 + slip_felt(i,2)^2)];
    slip = [slip; sqrt(slip_felt(i,1)^2 + slip_felt(i,2)^2) > SPEED_THRESH];
    temp = abs(fft(pressure_felt(i,end-LEN:end,:), [], 2)/LEN);
    temp = temp(1, 1:LEN/2 + 1, :);
    temp(2:end-1, :) = 2*temp(2:end-1, :);
    fft_felt = [fft_felt; temp];
    if slip(end) > 0
        slip_counter  = slip_counter + 1;
    else
        slip_counter = 0;
    end
    if slip_counter > 0
        fft_felt_slip = [fft_felt_slip; temp];
    else
        fft_felt_static = [fft_felt_static; temp];
    end
end

f = FREQ*(0:(LEN/2))/LEN;


%%% Signal Plot
figure;
a1 = subplot(2,1,1);
hold on;
plot(time_felt, pressure_felt(:,end,1));
plot(time_felt, pressure_felt(:,end,2));
plot(time_felt, pressure_felt(:,end,3));
plot(time_felt, pressure_felt(:,end,4));
plot(time_felt, pressure_felt(:,end,5));
plot(time_felt, pressure_felt(:,end,6));
hold off;
legend('Sensor 1','Sensor 2','Sensor 3','Sensor 4','Sensor 5','Sensor 6');
ylabel("Pressure Value")

a2 = subplot(2,1,2);
yyaxis left
plot(time_felt, slip);
ylim([-0.1 1.1])
ylabel("Slip Label")

% a3 = subplot(3,1,3);
yyaxis right
plot(time_felt, sum(fft_felt(:,2:end,:),[2,3]));
% hold on;
% plot(time_felt, sum(fft_felt(:,2:end,1),2));
% plot(time_felt, sum(fft_felt(:,2:end,2),2));
% plot(time_felt, sum(fft_felt(:,2:end,3),2));
% plot(time_felt, sum(fft_felt(:,2:end,4),2));
% plot(time_felt, sum(fft_felt(:,2:end,5),2));
% plot(time_felt, sum(fft_felt(:,2:end,6),2));
% hold off;
title("Vibration Energy 5-50Hz")
ylabel("Area Under Freq. Curve")
xlabel("Time(s)")

linkaxes([a1 a2],'x')

%%% FFT plot
figure;
A=[];
for i = 1:6
    a = subplot(6,1,i);
    A = [A, a];
    hold on;
    plot(f(2:end), mean(fft_felt_slip(:,2:end,i),1));
    plot(f(2:end), mean(fft_felt_static(:,2:end,i),1));
%     bar(f(2:end), [mean(fft_felt_slip(:,2:end,i),1); mean(fft_felt_static(:,2:end,i),1)] );
    legend('slip', 'static');
    hold off;
    grid on;
    title(sprintf('Average DFT of pressure sensor %d', i));
    ylabel("Pressure");
    if i == 6
       xlabel("Frequency (Hz)"); 
    end
%     xlim([-0.5,20.5])
end
linkaxes(A,'x');