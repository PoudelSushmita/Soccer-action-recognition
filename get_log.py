import re

def extract_metrics(log_file):
    epochs = []
    acc_top1 = []
    acc_top10 = []
    acc_mean1 = []
    loss_simv2t = []
    loss_simt2v = []

    with open(log_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if 'Epoch(train)' in line:
            epoch_match = re.search(r'Epoch\(train\) \[(\d+)\]', line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                epochs.append(epoch)

            acc_item_top1_match = re.search(r'acc/item_top1: ([\d.]+)', line)
            if acc_item_top1_match:
                acc_top1.append(float(acc_item_top1_match.group(1)))

            acc_item_mean1_match = re.search(r'acc/item_mean1: ([\d.]+)', line)
            if acc_item_mean1_match:
                acc_mean1.append(float(acc_item_mean1_match.group(1)))

            loss_simv2t_match = re.search(r'sim_loss_v2t: ([\d.]+)', line)
            if loss_simv2t_match:
                loss_simv2t.append(float(loss_simv2t_match.group(1)))

            loss_simt2v_match = re.search(r'sim_loss_t2v: ([\d.]+)', line)
            if loss_simt2v_match:
                loss_simt2v.append(float(loss_simt2v_match.group(1)))

        elif 'Epoch(val)' in line:
            acc_item_top10_match = re.search(r'acc/item_top5: ([\d.]+)', line)
            if acc_item_top10_match:
                acc_top10.append(float(acc_item_top10_match.group(1)))

    return epochs, acc_top1, acc_top10, acc_mean1, loss_simv2t, loss_simt2v

log_file = '/home/fm-pc-lt-281/projects/mmaction2/projects/actionclip/work_dirs/actionclip_vit-base-p32-res224-clip-pre_g8xb16_1x1x8_k400-rgb/20240214_143439/20240214_143439.log'  # Replace with the path to your log file
epochs, acc_top1, acc_top10, acc_mean1, loss_simv2t, loss_simt2v = extract_metrics(log_file)
print(acc_top1)
# Print extracted metrics
# print("Epoch\tAcc Top 1\tAcc Top 10\tAcc Mean 1\tLoss Sim V2T\tLoss Sim T2V")
# for i in range(len(epochs)):
#     print(f"{epochs[i]}\t{acc_top1[i]}\t\t{acc_top10[i]}\t\t{acc_mean1[i]}\t\t{loss_simv2t[i]}\t\t{loss_simt2v[i]}")

# for i in range(len(epochs)):
#     print(acc_top1[i])
    # print(f"i: {i}, Length of epochs: {len(epochs)}, Length of acc_top1: {len(acc_top1)}, Length of acc_top10: {len(acc_top10)}, Length of acc_mean1: {len(acc_mean1)}, Length of loss_simv2t: {len(loss_simv2t)}, Length of loss_simt2v: {len(loss_simt2v)}")
    # print(f"{epochs[i]}\t{acc_top1[i]}\t\t{acc_top10[i]}\t\t{acc_mean1[i]}\t\t{loss_simv2t[i]}\t\t{loss_simt2v[i]}")

