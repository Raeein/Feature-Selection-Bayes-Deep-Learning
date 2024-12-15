snp_list = [
    "SNP186781.2", "SNP156524.2", "SNP186781.2", "SNP52175.2",
    "SNP163049.1", "SNP203391.1", "SNP114192.2", "SNP7183.2",
    "SNP156524.1", "SNP126579.2", "SNP48917.2", "SNP96655.2",
    "SNP57159.1", "SNP84195.2", "SNP57099.2", "SNP57159.1",
    "SNP203391.2", "SNP201633.1", "SNP2320.2", "SNP2321.2"
]

# Path to the genotype.map file
map_file_path = "genotype.map"

# Create a dictionary to store SNP mappings
snp_to_chromosome = {}

line_number = 1
# Read the genotype.map file and populate the dictionary
with open(map_file_path, "r") as map_file:
    for line in map_file:
        # Split each line into columns
        columns = line.strip().split()
        chromosome = columns[1]
        snp_id = line_number

        # Add the SNP and its chromosome to the dictionary
        snp_to_chromosome[snp_id] = chromosome
        line_number += 1

# Print the chromosome for each SNP in the list
for snp in snp_list:
    stripped_snp = int(snp.replace("SNP", "").split(".")[0])
    if stripped_snp in snp_to_chromosome:
            print(f"{snp} is on chromosome {snp_to_chromosome[stripped_snp]}")
    else:
        print(f"{snp} not found in the genotype.map file")
