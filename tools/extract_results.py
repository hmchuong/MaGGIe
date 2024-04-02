import sys

file_log = sys.argv[1] # "/home/chuongh/vm2m/output/HIM/ours_1103_single-stage_strong-aug/test-log_rank0.log"
results = open("results.txt", "w")

metric_keys = ["MAD", "MAD_fg", "MAD_bg", "MAD_unk", "MSE", "SAD", "Grad", "Conn"]
# metric_keys = ["MAD", "MAD_fg", "MAD_bg", "MAD_unk", "MSE", "SAD"]
# metric_keys = ["MAD", "MSE", "SAD", "dtSSD", "MAD_fg", "MAD_bg", "MAD_unk", "MESSDdt"]
with open(file_log, "r") as f:
    start_idx = -1
    metrics = {}
    flag = False
    for line_idx, line in enumerate(f):
        if line.startswith("  test:"):
            flag = True
        if "mask_dir_name" in line and flag:
            if len(metrics) > 0:
                write_line = ""
                for key in metric_keys:
                    if key in metrics:
                        write_line +="{}\t".format(metrics[key])
                results.write("{}\n".format(write_line.strip()))
            else:
                print("No metrics")
            metrics = {}
            mask_dir_name = line.split(":")[-1].strip()
            print(mask_dir_name)
            flag = False
        if '[INFO ]  Metrics:' in line:
            start_idx = line_idx
        if line_idx < start_idx + 9 and start_idx != -1:
            # print(line)
            for key in metric_keys:
                if key + ":" in line:
                    metrics[key] = float(line.split(":")[-1].strip())
                    # print(key, metrics[key])
    if len(metrics) > 0:
        write_line = ""
        for key in metric_keys:
            if key in metrics:
                write_line +="{}\t".format(metrics[key])
        results.write("{}\n".format(write_line.strip()))
results.close()
