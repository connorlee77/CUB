acc = csvread('run_.,tag_acc.csv', 1);
vacc = csvread('run_.,tag_val_acc.csv', 1);
loss = csvread('run_.,tag_loss.csv', 1);
vloss = csvread('run_.,tag_val_loss.csv', 1);

figure
plot(acc(172:end,3));
hold on;
plot(vacc(172:end,3));
xlabel('epoch');
ylabel('accuracy');
legend('training','validation', 'location', 'southeast')
title('Classification Accuracy')

figure
plot(loss(172:end,3));
hold on;
plot(vloss(172:end,3));
xlabel('epoch');
ylabel('cross entropy loss');
legend('training','validation', 'location', 'northeast')
title('Loss')