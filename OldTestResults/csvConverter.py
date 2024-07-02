# import os

# directories = ["2-layer-chip", "3-layer-chip", "2-layer-chip-3x3", "3-layer-chip-3x3"]


# for directory in directories:
#     for filename in os.listdir(directory):
#         input_file = os.path.join(directory, filename)
#         output_file = os.path.join(directory, filename).split(".")[0] + ".csv"

#         data = [x.strip() for x in open(input_file, "r").readlines()]

#         keywords = ["".join(x.split(" ")[:-1]) for x in data[0].split(",")]

#         file = open(output_file, "w")

#         file.write(",".join(keywords) + "\n")

#         prevline = -1
#         for line in data:
#             datum = [x.split(" ")[-1] for x in line.split(",")]
#             if prevline == datum[0]:
#                 continue
#             prevline = datum[0]
#             file.write(",".join(datum) + "\n")

#         file.write("\n")


import os

directories = ["2-layer-chip", "3-layer-chip", "2-layer-chip-3x3", "3-layer-chip-3x3"]


for directory in directories:
    filename = "output_normal_model_approx.txt"
    input_file = os.path.join(directory, filename)
    output_file = os.path.join(directory, filename).split(".")[0] + ".csv"

    data = [x.strip() for x in open(input_file, "r").readlines()]

    keywords = ["".join(x.strip().split(" ")[:-1]) for x in data[0].split(",")]

    file = open(output_file, "w")

    file.write(",".join(keywords) + "\n")

    prevline = -1
    for line in data:
        datum = [x.strip().split(" ")[-1] for x in line.split(",")]
        if prevline == datum[0]:
            continue
        prevline = datum[0]
        file.write(",".join(datum) + "\n")

    file.write("\n")