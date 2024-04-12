import os
import sys

file_log = sys.argv[1]
output_dir = sys.argv[2]
results = open(os.path.join(output_dir, "results.csv"), "w")

metric_keys = ["MAD", "MAD_fg", "MAD_unk", "MSE", "SAD", "Grad", "Conn"]
results.write("split,masks,")
results.write("{}\n".format(",".join(metric_keys)))

def write_line(metrics, mask_dir_name, split, f):
    if len(metrics) > 0:
        write_line = "{},{},".format(split, mask_dir_name)
        for key in metric_keys:
            if key in metrics:
                write_line +="{},".format(metrics[key])
            else:
                write_line +=","

        f.write("{}\n".format(write_line[:-1].strip()))

with open(file_log, "r") as f:
    start_idx = -1
    metrics = {}
    flag = False
    mask_dir_name = ''
    split = ''
    for line_idx, line in enumerate(f):
        if line.startswith("  test:"):
            flag = 0
            write_line(metrics, mask_dir_name, split, results)
        if "mask_dir_name:" in line and flag < 2:
            metrics = {}
            mask_dir_name = line.split(":")[-1].strip()
            mask_dir_name = mask_dir_name.replace("masks_matched_", "")
            flag += 1
        if "split:" in line and flag < 2:
            split = line.split(":")[-1].strip()
            flag += 1

        if '[INFO ]  Metrics:' in line:
            start_idx = line_idx
        if line_idx < start_idx + 9 and start_idx != -1:
            # print(line)
            for key in metric_keys:
                if key + ":" in line:
                    metrics[key] = float(line.split(":")[-1].strip())
    write_line(metrics, mask_dir_name, split, results)
results.close()
